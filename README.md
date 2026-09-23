# Missing Data: Foundations and Methods

Companion website for *Missing Data: Foundations and Methods* by [Lu Mao, PhD](https://lmaowisc.github.io/), University of Wisconsin–Madison.

The published site is available at <https://lmaowisc.github.io/missing_data/>.

## Contents

- Seven narrative chapters and their accompanying slide PDFs.
- Original figures and tables used in the selected chapter materials.
- Local MathJax assets so mathematical content does not depend on a third-party CDN.
- A repository-scoped comments section, powered by GitHub Discussions through Giscus once enabled.

## Publishing

Edit the book in **`chapter-1.qmd` through `chapter-7.qmd`**. Edit front-page prose in `index.qmd` and chapter titles/descriptions in `book.json`.

Run `quarto preview --port 8261` and open <http://127.0.0.1:8261/>. Saved changes rebuild automatically and refresh the preview. Open `missing_data.Rproj` in RStudio, or open this folder in VS Code with the Quarto extension.

```powershell
git add .
git commit -m "Revise book text"
git push
```

GitHub Actions runs `quarto render` to build and validate the site from `.qmd` files, then deploys only `_site/` to GitHub Pages. Generated HTML is not tracked. See **[EDITING.md](EDITING.md)** for installation and editing instructions.

To activate comments, enable GitHub Discussions for the repository, create a public `Comments` category, install the [Giscus GitHub App](https://github.com/apps/giscus) for this repository, and copy the repository and category identifiers from <https://giscus.app/> into `comments.js`. Comments then remain in the repository's own GitHub Discussions.

## Editorial source

The `.qmd` files in this repository are the authoritative manuscript. `_quarto.yml` configures the native Quarto project; an automatic post-render hook preserves the custom design. The site builds independently of the original local prototype. Do not copy generated pages from the old prototype over these sources. Third-party notes and unrelated course files remain excluded.
