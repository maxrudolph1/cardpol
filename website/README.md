# CARDPol project website

A buildless, responsive research project page modeled after the site in
`../meta_plotting/website`. It uses local assets, system fonts, and vanilla
JavaScript; no package installation or external CDN is required. All core content
and result tables remain available with JavaScript disabled.

## Preview

From the repository root:

```bash
python3 -m http.server 8000 --directory website
```

Open `http://localhost:8000`. Relative asset URLs also work at the GitHub Pages
project subpath `/cardpol/`.

## Publish with GitHub Pages

1. In the GitHub repository, open **Settings → Pages** and choose **GitHub Actions**
   as the build and deployment source.
2. Commit and push `website/` and `.github/workflows/pages.yml` to `main`.
3. The **Deploy project website to GitHub Pages** workflow publishes the site.
   It also supports manual runs from the Actions tab.

With the repository's current name and owner, the default URL will be
`https://maxrudolph1.github.io/cardpol/` once deployment succeeds. No deployment
or repository settings changes are performed by adding these files.

## Content and assets

- `index.html`: paper summary, method, results, and adapted figure captions.
- `static/css/index.css`: responsive styling inspired by the reference site.
- `static/js/index.js`: domain-switchable DMC chart, derived from the HTML table.
- `static/paper/cardpol.pdf`: compiled from `paper/main.tex`, including bibliography
  and appendix, with the source's anonymous submission formatting preserved.
- `static/images/`: WebP exports of the paper's robot illustrations, environment
  examples, state-decoding plots, distractor-variance plot, legend, and EgoDex plot.
  `favicon.svg` is a small code-native site icon.

The paper source contains template author and affiliation placeholders. The site
intentionally omits authors and venue until the final metadata is supplied.
The DMC table comes from `paper/figures/tables/drqbc_downstream_rl_shift.tex`;
the LIBERO table comes from the actively included
`paper/figures/tables/method_env_table.tex`, rather than older result tables.
Bold in the website tables denotes the highest **mean**, not statistical
significance or the paper's overlapping-error-interval formatting.

The public code link describes the repository's DMC reproduction scope; it does
not promise LIBERO or EgoDex training code. No citation is invented for the
unpublished manuscript.
