"""Build the public site from manuscript/*.md using Quarto's bundled Pandoc."""
from pathlib import Path
from urllib.parse import urlsplit, unquote
import argparse
import html
import json
import shutil
import subprocess
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '_site'
ESC = html.escape

def render(source):
    result = subprocess.run(
        ['quarto', 'pandoc', str(ROOT / source), '--from=markdown+fenced_divs+bracketed_spans+tex_math_dollars-implicit_figures',
         '--to=html5', '--mathjax', '--wrap=none'],
        capture_output=True, text=True, encoding='utf-8', check=True)
    return BeautifulSoup(result.stdout, 'html.parser')

def verify():
    errors = []
    documents = {p: BeautifulSoup(p.read_text(encoding='utf-8'), 'html.parser') for p in OUT.glob('*.html')}
    for page, soup in documents.items():
        ids = [tag['id'] for tag in soup.select('[id]')]
        if len(ids) != len(set(ids)):
            errors.append(f'{page.name}: duplicate anchors')
        for tag in soup.select('[href], [src]'):
            ref = urlsplit(tag.get('href', tag.get('src', '')))
            if ref.scheme or ref.netloc:
                continue
            target = OUT / unquote(ref.path) if ref.path else page
            if not target.is_file():
                errors.append(f'{page.name}: missing {target.name}')
            elif ref.fragment and target.suffix == '.html':
                other = documents[target]
                if other.find(id=unquote(ref.fragment)) is None:
                    errors.append(f'{page.name}: missing anchor {ref.fragment}')
    if errors:
        raise RuntimeError('\n'.join(errors))
    print('Verified all local links, assets, anchors, and unique IDs.')

def build():
    book = json.loads((ROOT / 'book.json').read_text(encoding='utf-8'))
    chapters = book['chapters']
    OUT.mkdir(exist_ok=True)
    # Copy only public assets, never manuscripts, build tools, or Git metadata.
    for name in ('figures', 'slides', 'vendor'):
        shutil.copytree(ROOT / name, OUT / name, dirs_exist_ok=True)
    for name in ('style.css', 'favicon.svg', 'math-config.js', 'comments.js', '.nojekyll'):
        shutil.copy2(ROOT / name, OUT / name)
    shell = (ROOT / 'templates/page.html').read_text(encoding='utf-8')
    footer = (ROOT / 'templates/footer.html').read_text(encoding='utf-8')
    for number in range(len(chapters) + 1):
        nav = ''.join(f'<a class="chapter-link{" active" if number == c["number"] else ""}" '
                      f'href="chapter-{c["number"]}.html"'
                      + (' aria-current="page"' if number == c['number'] else '')
                      + f'><span>{c["number"]:02}</span>{ESC(c["title"])}</a>' for c in chapters)
        if number:
            c = chapters[number - 1]
            body = render(c['source'])
            headings = body.select('h2[id]')
            links = ''.join(f'<a href="#{ESC(h["id"])}">{ESC(h.get_text(" ", strip=True))}</a>' for h in headings)
            toc = '<p class="nav-label">IN THIS CHAPTER</p>' + links
            content = f'<div class="eyebrow">CHAPTER {number:02}</div><h1 class="chapter-title">{ESC(c["title"])}</h1><p class="lead">{ESC(c["description"])}</p>'
            content += f'<div class="slide-bar"><div><strong>Lecture slides</strong><span>A companion to the chapter · PDF</span></div><a class="button" href="slides/chapter-{number}.pdf">Open slides ↗</a></div>'
            content += f'<details class="chapter-contents"><summary>Contents of this chapter</summary><nav aria-label="Chapter contents">{links}</nav></details><article class="book-content">{body}</article>'
            previous = f'<a href="chapter-{number-1}.html">← Previous chapter</a>' if number > 1 else '<a href="index.html">← Course overview</a>'
            following = f'<a href="chapter-{number+1}.html">Next chapter →</a>' if number < len(chapters) else '<a href="index.html">Course overview →</a>'
            content += f'<nav class="page-turn" aria-label="Adjacent chapters">{previous}{following}</nav>'
            title, filename = c['title'], f'chapter-{number}.html'
        else:
            body = render('manuscript/index.md')
            # Keep the original front-page elements while authoring their text in Markdown.
            for selector in ('.affiliation', '.lead', '.intro'):
                wrapper = body.select_one(selector)
                child = wrapper.find(['p', 'ul'], recursive=False) if wrapper else None
                if child:
                    child['class'] = wrapper.get('class', [])
                    if selector == '.intro':
                        child['aria-label'] = 'Course themes'
                    wrapper.replace_with(child.extract())
            for paragraph in body.select('.hero-actions > p'):
                paragraph.unwrap()
            rows = ''.join(f'<article class="chapter-row"><span class="number">{c["number"]:02}</span><div><h3><a href="chapter-{c["number"]}.html">{ESC(c["title"])}</a></h3><p>{ESC(c["description"])}</p><div class="row-links"><a href="chapter-{c["number"]}.html">Read chapter <span aria-hidden="true">→</span></a><a href="slides/chapter-{c["number"]}.pdf">Slides <span class="filetype">PDF</span></a></div></div></article>' for c in chapters)
            listing = f'<section id="chapters"><div class="section-heading"><h2>Chapters &amp; lecture slides</h2><span>01 — {len(chapters):02}</span></div>{rows}</section>'
            placeholder = body.select_one('#chapter-list-placeholder')
            if placeholder is None:
                raise ValueError('Keep the chapter-list-placeholder marker in manuscript/index.md')
            placeholder.replace_with(BeautifulSoup(listing, 'html.parser'))
            content = str(body)
            toc = '<p class="nav-label">ON THIS PAGE</p><a href="#chapters">Chapters &amp; slides</a>'
            title, filename = 'Course overview', 'index.html'
        page = shell
        for key, value in {'TITLE': ESC(title + ' · ' + book['title']), 'CHAPTER': str(number),
                           'NAV': nav, 'MAIN': content + footer, 'TOC': toc}.items():
            page = page.replace('@@' + key + '@@', value)
        soup = BeautifulSoup(page, 'html.parser')
        for figure in soup.select('.book-figure'):
            figure.name = 'figure'
            caption = figure.select_one('.figure-caption')
            if caption:
                caption.name = 'figcaption'
            for image in figure.select('img'):
                if image.parent.name == 'p':
                    image.parent.unwrap()
                if not image.find_parent('a'):
                    link = soup.new_tag('a', href=image['src'], attrs={'class': 'figure-link', 'aria-label': 'Open figure at full size'})
                    image.wrap(link)
        if not number:
            soup.select_one('.overview')['class'] = ['overview', 'active']
        for math in soup.select('.math'):
            text = math.get_text()
            if text.startswith(('\\(', '\\[')):
                math['data-tex'] = text[2:-2]
        for table in soup.select('table'):
            if not table.find_parent(class_='table-scroll'):
                wrapper = soup.new_tag('div', attrs={'class': 'table-scroll', 'tabindex': '0', 'aria-label': 'Scrollable data table'})
                table.wrap(wrapper)
        for link in soup.select('a[href]'):
            url = urlsplit(link['href'])
            if url.scheme in ('http', 'https') or url.netloc:
                link['target'] = '_blank'
                link['rel'] = list(dict.fromkeys([*link.get('rel', []), 'noopener', 'noreferrer']))
        (OUT / filename).write_text(str(soup), encoding='utf-8')
        print(f'Built {filename}')
    verify()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check the existing generated site only')
    args = parser.parse_args()
    if args.check:
        verify()
    else:
        build()
