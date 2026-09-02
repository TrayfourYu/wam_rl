# Dual-Branch RL for World-Action Models — blog assets

Ready-to-publish Jekyll post for a `*.github.io` research blog.

```
wam-dual-branch-rl-blog/
├── _posts/2026-09-02-dual-branch-rl-for-wam.md   ← the blog post
├── assets/fig1_eval_success.png                  ← Figure 1 (1660 px wide, 300 dpi)
├── assets/fig1_eval_success.svg                  ← Figure 1 (vector, crisper on retina screens)
├── assets/references.bib                         ← BibTeX for jekyll-scholar / pandoc users
└── README.md
```

## 1. Copy the files

```bash
cp _posts/2026-09-02-dual-branch-rl-for-wam.md  <your-repo>/_posts/
cp assets/fig1_eval_success.png assets/fig1_eval_success.svg  <your-repo>/assets/
cp assets/references.bib                       <your-repo>/assets/     # optional
```

## 2. Make the math render

The post ships with its own MathJax v3 loader (the two `<script>` tags at the top),
so it works in any Jekyll theme. Two things to check:

**a) kramdown must not pre-process the math** (this is the #1 reason formulas show
up as raw LaTeX on GitHub Pages). Add to `_config.yml`:

```yaml
markdown: kramdown
kramdown:
  math_engine: null
```

**b) If your theme already provides MathJax** (e.g. minimal-mistakes with
`mathjax: true`, or jekyll-theme-chirpy), delete the two `<script>` tags at the
top of the post to avoid loading MathJax twice.

## 3. Front matter

Already set. Adjust `layout` (`post` is correct for most themes; `single` for
chirpy) and the `subtitle` if your theme does not support it.

## 4. Figure

The image is referenced with Liquid so it works both on user sites
(`user.github.io`) and project sites (`user.github.io/repo`):

```markdown
![...]({{ '/assets/fig1_eval_success.png' | relative_url }})
```

If you paste the post into a plain Markdown renderer (GitHub README, Typora,
etc.), replace it with `assets/fig1_eval_success.png`.

## 5. Before you publish

Open the post in your editor and grep for `[[fill:` — there is one placeholder
(your name / URL in the "Cite this post" BibTeX block). Everything else is
written out; see the `AUTHOR NOTES` HTML comment at the very bottom of the post
for the numbers you should double-check against your own logs.
