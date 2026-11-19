<!--
SPDX-FileCopyrightText: 2026 The Sanket contributors (Team Winging It)
SPDX-License-Identifier: MIT
-->
# Final research paper

The deliverable is [`Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India.pdf`](Explainable%20AI%20for%20District-Level%20Prediction%20of%20Medium-Range%20Weather%20Forecast%20Busts%20over%20India.pdf). It follows the section requirements of `Research_paper/Reference/Research Paper Template.pdf` while using an IEEE-style two-column layout, Times typography, numbered IEEE citations, numbered tables/figures, declarations and a reproducibility appendix.

## Source and rebuild

`build_research_paper.py` contains the manuscript text and draws all tables/figures as vector graphics; no external image files are required.

```bash
python -m venv .venv
.venv/bin/python -m pip install reportlab pymupdf
.venv/bin/python Research_paper/Final/build_research_paper.py --check
```

The `--check` option compares the finished PDF with the template, earlier reports in `Research_paper/Final`, reference PDFs, README, documentation and survey scripts. It reports contiguous 8-word and 12-word overlap with that source corpus. This is a reproducible local similarity check, not a vendor plagiarism-service score.

## Validation output from the committed build

```text
Abstract: 228 words; keywords: 5
References: 26; all cited in text
Tables and figures: numbered and referenced in text
Template placeholders: none
Wrote Research_paper/Final/Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India.pdf (36,801 bytes, 6 pages)
Local 8-word overlap, main text: 0.354%
Local 8-word overlap, final PDF: 1.885%
Local 12-word overlap, main text: 0.079%
```

The manuscript was written anew from the measured project facts. Factual values, proper names and cited reference titles are retained; prose, structure, captions and figures are original.
