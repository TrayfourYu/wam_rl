# Research blog

Personal research notes on embodied AI, vision-language-action (VLA) models,
world-action models (WAM), and reinforcement learning for robot manipulation.

Built with [Jekyll](https://jekyllrb.com/) + the Minima theme, hosted on
GitHub Pages at <https://trayfouryu.github.io/wam_rl/>.

## Layout

```
_config.yml      site configuration
index.md         home page (lists all posts)
_posts/          blog posts, one Markdown file each
assets/          images, BibTeX, and other static files
```

Rules: posts go in `_posts/`, images go in `assets/`.

## Writing a new post

Create `_posts/YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "Your title"
date: 2026-09-15 10:00:00 +0800
categories: [embodied-ai]
tags: [VLA, RL]
---
```

Equations render with MathJax; copy the two `<script>` tags from the top of the
existing post if a page needs math. Reference images with
`{{ '/assets/name.png' | relative_url }}` so the path stays correct on
project sites.
