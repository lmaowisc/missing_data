# Missing Data: Foundations and Methods

Companion website for *Missing Data: Foundations and Methods* by [Lu Mao, PhD](https://lmaowisc.github.io/), University of Wisconsin–Madison.

The published site is available at <https://lmaowisc.github.io/missing_data/>.

## Contents

- Seven narrative chapters and their accompanying slide PDFs.
- Original figures and tables used in the selected chapter materials.
- Local MathJax assets so mathematical content does not depend on a third-party CDN.
- A repository-scoped comments section, powered by GitHub Discussions through Giscus once enabled.

## Publishing

Edit the book in **`manuscript/chapter-1.md` through `manuscript/chapter-7.md`**. Edit front-page prose in `manuscript/index.md` and chapter titles/descriptions in `book.json`.

Run `python preview.py` and open <http://127.0.0.1:8261/>. Saved changes rebuild automatically; refresh the browser to view them.

```powershell
git add .
git commit -m "Revise book text"
git push
```

GitHub Actions builds and validates the site from Markdown, then deploys only `_site/` to GitHub Pages. Generated HTML is not tracked. See **[EDITING.md](EDITING.md)** for installation and editing instructions.

To activate comments, enable GitHub Discussions for the repository, create a public `Comments` category, install the [Giscus GitHub App](https://github.com/apps/giscus) for this repository, and copy the repository and category identifiers from <https://giscus.app/> into `comments.js`. Comments then remain in the repository's own GitHub Discussions.

## Editorial source

The Markdown files in this repository are the authoritative manuscript. Quarto's bundled Pandoc and `build.py` generate the custom website independently of the original local prototype. Do not copy generated pages from the old prototype over these sources. Third-party notes and unrelated course files remain excluded.
