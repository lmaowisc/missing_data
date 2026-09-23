# Editing the book

The authoritative book text is in `manuscript/`. Edit these files in VS Code or another text editor. Turn on **View → Word Wrap** in VS Code for comfortable paragraph editing.

| File | What to edit |
| --- | --- |
| `manuscript/index.md` | Front-page text and author information |
| `manuscript/chapter-1.md` … `chapter-7.md` | Chapter prose and mathematics |
| `book.json` | Chapter titles and short descriptions used throughout the site |
| `figures/` | Chapter figures |
| `slides/` | Companion slide PDFs |
| `style.css` | Typography, colors, and layout |

## Daily workflow

Open a terminal in this repository folder and start the preview:

```powershell
python preview.py
```

Open <http://127.0.0.1:8261/>. Keep the terminal running. Save a manuscript file, wait for **Ready — refresh your browser**, and refresh to see the rebuilt page. Press Ctrl+C to stop the preview. The public website is unaffected until you push.

In a second terminal in this folder, publish when ready:

```powershell
git status
git diff
git add .
git commit -m "Revise Chapter 1"
git push
```

The repository uses Windows' certificate store locally. On a new Windows checkout, if Git reports a certificate-chain error, use `git -c http.sslBackend=schannel push`.

GitHub installs the build tools, generates the HTML from Markdown, checks links and anchors, and deploys `_site/`. Follow progress at <https://github.com/lmaowisc/missing_data/actions>. A failed build leaves the previously deployed website in place.

## Writing Markdown and mathematics

Write natural paragraphs separated by blank lines. Use `**bold**`, `*italics*`, and `[link text](https://example.org)`.

```markdown
## 1.2 The observed-data likelihood {#section-2}

Let $Y$ denote the full data and $R$ the observation indicator.
The observed-data likelihood integrates over the missing components:

$$
L(\theta; y_{\mathrm{obs}})
= \int f(y_{\mathrm{obs}}, y_{\mathrm{mis}}; \theta)\,d y_{\mathrm{mis}}.
$$

This representation connects the full-data model to the observed record.
```

Use `$...$` for inline mathematics and `$$` on separate lines for displayed equations. Avoid blank lines inside an equation. Existing `\label`, `\ref`, and `\tag` commands are retained. Section numbers are written explicitly; change them when reorganizing sections. Keep existing `{#section-2}` identifiers when changing heading wording, so old links continue to work. New second-level headings automatically appear in the contents lists.

Blocks marked `::: math-block`, `::: math-passage`, or `::: book-figure` preserve the chapter layout. Edit the prose and equations inside them and retain the matching colon fences. A few complex tables retain HTML to preserve merged cells; ordinary prose and equations use Markdown. Images are relative to the published site root, for example `figures/MARid.png`.

The `chapter-list-placeholder` block in the front-page manuscript is replaced with the chapter listing from `book.json`; keep that marker.

## Build once or set up another computer

Install Python 3.13 and [Quarto 1.6.39](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.6.39), then run:

```powershell
python -m pip install -r requirements.txt
python build.py
```

Quarto supplies the Pandoc converter; the Python build preserves this site's custom layout. This is a Markdown project with a custom build, so use `python build.py` or `python preview.py`, not `quarto render`.

Generated HTML is placed in `_site/` and ignored by Git. Do not edit it: the next build replaces it. The old sibling `prototype/` folder is an archive of the original slide-to-prose development process. Do not copy its output into this repository or rerun the one-time migration over your edited manuscripts. This repository builds independently of that archive.

Comments currently link to this book's GitHub Discussions forum. Embedded Giscus activation remains pending GitHub account re-authentication; this manuscript migration does not change that status.
