# Liga F website

A static site with two pages:

- `index.html` is the written analysis (the landing page). Every number in its prose is filled in from the data by `js/story.js`.
- `explore.html` is the interactive dashboard (`js/explore.js`).

Both read the CSV files in `app/data/` and share `js/common.js`. The site uses Plotly in the browser and needs no server code or build step.

## Run locally

Browsers block CSV loading from `file://` pages, so serve the folder instead of double-clicking `index.html`:

```bash
python3 -m http.server 8000 --directory app
```

Then open `http://localhost:8000`. Each tab has its own address (for example `#styles`), so you can link straight to a section.

## Refresh the data

From the project root, re-run the pipeline. It overwrites `app/data/` and removes any CSV the dashboard no longer uses:

```bash
python src/build_liga_f_metrics.py
```

The site asks the browser to revalidate data files on every load, so a deployed site shows new numbers as soon as the refreshed CSVs are pushed. If you change the JavaScript or `styles.css`, bump the `?v=` number on their tags in both HTML files. Then run `python src/make_social_card.py` so the preview image matches.

## Deploy with GitHub Pages

1. Push the project to its own GitHub repository (`data/` is ignored; `app/data/` is committed).
2. In GitHub, open **Settings → Pages** and set **Source** to **GitHub Actions**. Branch deployment can only serve `/` or `/docs`, not `/app`.
3. The included workflow, `.github/workflows/pages.yml`, publishes `app/` on every push to `main`. The public `github.io` address appears in the workflow run and under **Settings → Pages**.
