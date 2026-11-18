# Methodology and Model Performance Report

Source, figures, and reproducible build script for the assignment **Methodology and Model
Performance Report** on the research paper *"Explainable AI for District-Level Prediction
of Medium-Range Weather Forecast Busts over India"*:

- `../../Research_paper/Final/Methodology and Model Performance Report - Sanket.pdf` — the
  compiled PDF report formatted to match
  `../../Research_paper/Reference/Methodology_and_Model_Performance_Report.pdf`.
- `methodology-report.md` — Markdown companion of the report text and tables.
- `build_report.py` — runs the production ingestion, two-stage XGBoost + Platt training,
  candidate tabular models, statistical baseline ladder, and exact TreeSHAP pipeline in an
  isolated temporary directory against `backend/data/samples/*_2019.parquet` (without
  modifying `backend/data/`), pairs those reproducible sample numbers with the verified
  17-year archive measurements documented in `docs/engineering-reference.md` and
  `docs/known-issues.md`, generates `assets/*.png`, and builds the PDF.
- `assets/` — the three figures embedded in the PDF (`fig1_methodology_pipeline.png`,
  `fig2_performance_comparison.png`, `fig3_shap_and_regressors.png`).

## Rebuilding

From the repository root, with `backend/requirements.txt`, `reportlab`, and `matplotlib`
installed:

```bash
python docs/methodology-report/build_report.py
```
