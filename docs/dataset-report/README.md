# Dataset report

The assignment dataset report for the Sanket project, in the format of
[`Reference dataset report.pdf`](../../Reference%20dataset%20report.pdf) in the repository
root: same cover layout, same section heads (Data description, Data cleaning method
selection and justification, Results, Conclusion, References), same typography.

| File | What it is |
|---|---|
| [`../Dataset report - Sanket.pdf`](../../Dataset%20report%20-%20Sanket.pdf) | **The deliverable.** Cover, title, problem statement, data life cycle, Sections 1–5, 15 pages |
| [dataset-report.md](dataset-report.md) | The same content as markdown, written by the build script for review |
| [build_report.py](build_report.py) | Builds both. Edits the reference PDF's cover in place, renders the figures from `backend/data/`, renders the six content pages, merges |
| [assets/](assets/) | The three figures, regenerated on every build from the data files |

## Rebuilding it

```bash
python docs/dataset-report/build_report.py
```

Needs `pymupdf`, `reportlab`, `matplotlib`, `pandas`, `pyarrow` and `pillow`, plus
Liberation Serif (or Tinos) installed — the reference report is set in Liberation Serif,
and the cover edit writes the new title and names in the font already embedded in it.

## Where the numbers come from

- **Measured for this report, from `backend/data/`:** the error statistics in Table 8 and
  Figure 1, the missingness in Table 5, the coverage in Figure 2, the geography in Figure
  3, and the lead-day error growth in Table 9. The build script recomputes all of them
  from `backend/data/samples/*_2019.parquet`, `backend/data/geo/*.parquet` and
  `backend/data/mjo_omi_index.parquet` on every run, so they cannot go stale silently.
- **Taken from the project's own documents, each with the date it was measured where the
  document records one:** `docs/engineering-reference.md`, `docs/known-issues.md`,
  `docs/one-pager.md` and `docs/analysis/verification_product_agreement.csv`. These are the
  figures that come from real runs at full scale — 365 cycles and 12,154,500 rows for 2017,
  the dedupe and IMD findings, the SHAP and baseline results.
- **Not quoted anywhere in the report:** the live model's current scores. Those are read
  from the deployed model by the site's About page, because a number copied into a
  document is correct until the next retrain and quietly wrong afterwards.

Anything that was not measured is not in the report, and every limitation the repository
records is stated with its number rather than omitted.
