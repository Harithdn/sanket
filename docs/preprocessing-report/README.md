# Preprocessing and model-selection report

The assignment report for the Sanket project, in the format of
[`Dataset_Preprocessing_Model_Selection_Report_Template.pdf`](../../Dataset_Preprocessing_Model_Selection_Report_Template.pdf)
in the repository root: same cover block and student-details table, same section heads
(1. Project Overview through 11. Appendix and the Student Declaration), same typography
(Helvetica; 17 pt title and 13 pt section heads in #1F4E79, 8.5 pt / 7.3 pt tables with the
template's #D9EAF7 header fill and 0.5 pt grey rules), same footer on every page.

| File | What it is |
|---|---|
| [`../../Dataset Preprocessing and Model Selection Report - Sanket.pdf`](../../Dataset%20Preprocessing%20and%20Model%20Selection%20Report%20-%20Sanket.pdf) | **The deliverable.** Cover, Sections 1–11, 16 tables, 8 figures, 20 pages |
| [preprocessing-report.md](preprocessing-report.md) | The same content as markdown, written by the build script for review |
| [build_report.py](build_report.py) | Builds both. Measures the data, draws the eight figures, renders the report |
| [assets/](assets/) | The eight figures, regenerated on every build from the data files |

## Rebuilding it

```bash
python docs/preprocessing-report/build_report.py
```

Needs `reportlab`, `matplotlib`, `pandas`, `pyarrow` and `pillow`. Helvetica is a reportlab
built-in, so nothing has to be installed as a font.

## Where the numbers come from

- **Measured for this report, by the build script, from `backend/data/`:** every table and
  figure in Section 5 (dimensions, summary statistics, error statistics and the p90 bust
  thresholds, class distribution by lead and month, spread-versus-error rank correlations,
  missingness and duplicate counts), the data-quality findings in Section 3 (the soil
  saturation at Daman, Kavaratti and Port Blair, the per-station humidity bias, the
  zero-inflation of rainfall) and the geography figures in Section 2. The script recomputes
  all of them from `backend/data/samples/*_2019.parquet`, `backend/data/geo/*.parquet` and
  `backend/data/mjo_omi_index.parquet` on every run, so they cannot go stale silently.
- **Taken from the project's own documents, each with the date it was measured where the
  document records one:** the archive-scale figures (7,288 of 7,305 cycles, 33,300 rows per
  cycle, 12,154,500 rows for 2017), the two corrupt member-days, the IMD date convention, the
  sea-masking effect, the 21,492 paired city-days of ERA5-versus-MERRA-2 agreement, and the
  label and calibration findings — `docs/engineering-reference.md`, `docs/known-issues.md`
  and `docs/analysis/verification_product_agreement.csv`.
- **Not quoted anywhere in the report:** the live model's scores. This stage reports no
  model performance; those numbers are read from the deployed model by the site's About page,
  because a number copied into a document is correct until the next retrain and quietly wrong
  afterwards.

The report is measured on the archived 2019 sample and the geography layer, not on the full
twenty-year store, which is a release asset rather than a repository file. That limitation is
stated in the appendix rather than left implicit.
