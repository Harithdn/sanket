# SPDX-FileCopyrightText: 2026 The Sanket contributors (Team Winging It)
# SPDX-License-Identifier: MIT
"""Build the Sanket Methodology and Model Performance Report PDF and figures.

Recomputes the archived 2019 sample run (Stage-1 per-variable XGBoost error
regressors, Stage-2 XGBoost + Platt bust classifier, candidate tabular models,
the eight-rung statistical baseline ladder, and exact TreeSHAP attributions) in
an isolated temporary store without modifying ``backend/data/``, pairs those
reproducible measurements with the verified 17-year district-archive runs
documented in ``docs/engineering-reference.md`` and ``docs/known-issues.md``,
renders three figures under ``docs/methodology-report/assets/``, and compiles
``Research_paper/Final/Methodology and Model Performance Report - Sanket.pdf``
in the exact layout of ``Research_paper/Reference/Methodology_and_Model_Performance_Report.pdf``.

Run from the repository root::

    python docs/methodology-report/build_report.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BACKEND = REPO / "backend"
ASSETS = HERE / "assets"
PDF_OUT = REPO / "Research_paper" / "Final" / "Methodology and Model Performance Report - Sanket.pdf"

PAGE_W, PAGE_H = A4
MARGIN_LR = 72.0
MARGIN_TB = 55.0
BODY_W = PAGE_W - 2 * MARGIN_LR

# Reference palette (exact hex match to Methodology_and_Model_Performance_Report.pdf)
C_TITLE = colors.HexColor("#1F3864")
C_SUBTITLE = colors.HexColor("#2E5597")
C_HEADING = colors.HexColor("#2E74B5")
C_TEXT = colors.HexColor("#000000")
C_TBL_HDR = colors.HexColor("#E7E6E6")
C_TBL_GRID = colors.HexColor("#888888")

# Matplotlib hex palette aligned with the Reference PDF
HEX_NAVY = "#1F3864"
HEX_BLUE = "#2E74B5"
HEX_BLUE_DARK = "#2E5597"
HEX_ORANGE = "#C55A11"
HEX_GREEN = "#548235"
HEX_GRAY = "#7F7F7F"
HEX_LIGHT_GRAY = "#BFBFBF"
HEX_RED = "#C00000"


# ---------------------------------------------------------------------------
# Font registration (Times New Roman / Tinos metric-compatible serif)
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    (
        Path.home() / ".local/share/fonts/Tinos-Regular.ttf",
        Path.home() / ".local/share/fonts/Tinos-Bold.ttf",
        Path.home() / ".local/share/fonts/Tinos-Italic.ttf",
        Path.home() / ".local/share/fonts/Tinos-BoldItalic.ttf",
    ),
    (
        Path("/usr/share/fonts/truetype/croscore/Tinos-Regular.ttf"),
        Path("/usr/share/fonts/truetype/croscore/Tinos-Bold.ttf"),
        Path("/usr/share/fonts/truetype/croscore/Tinos-Italic.ttf"),
        Path("/usr/share/fonts/truetype/croscore/Tinos-BoldItalic.ttf",),
    ),
    (
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf"),
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"),
    ),
]


def register_fonts() -> tuple[str, str, str, str]:
    for reg, bold, ital, bital in FONT_CANDIDATES:
        if all(p.exists() for p in (reg, bold, ital, bital)):
            pdfmetrics.registerFont(TTFont("ReportSerif", str(reg)))
            pdfmetrics.registerFont(TTFont("ReportSerif-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReportSerif-Italic", str(ital)))
            pdfmetrics.registerFont(TTFont("ReportSerif-BoldItalic", str(bital)))
            pdfmetrics.registerFontFamily(
                "ReportSerif",
                normal="ReportSerif",
                bold="ReportSerif-Bold",
                italic="ReportSerif-Italic",
                boldItalic="ReportSerif-BoldItalic",
            )
            for p in (reg, bold, ital, bital):
                fm.fontManager.addfont(str(p))
            family_name = fm.FontProperties(fname=str(reg)).get_name()
            plt.rcParams.update(
                {
                    "font.family": family_name,
                    "mathtext.fontset": "stix",
                    "axes.edgecolor": "#444444",
                    "axes.linewidth": 0.7,
                }
            )
            return ("ReportSerif", "ReportSerif-Bold", "ReportSerif-Italic", "ReportSerif-BoldItalic")
    return ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic")


# ---------------------------------------------------------------------------
# Reproducible archived-sample evaluation (isolated in tempfile)
# ---------------------------------------------------------------------------
def run_archived_sample_eval() -> dict:
    """Run the production ingestion, training, baseline ladder, candidate models,
    and TreeSHAP pipeline against ``backend/data/samples/*_2019.parquet`` in an
    isolated temporary directory so ``backend/data/`` stays untouched.
    """
    tmp = Path(tempfile.mkdtemp(prefix="sanket_methodology_eval_"))
    try:
        import json
        import subprocess

        data_dir = tmp / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(BACKEND / "data" / "geo", data_dir / "geo")
        env = os.environ.copy()
        env.update(
            {
                "DATA_DIR": str(data_dir),
                "DB_PATH": str(tmp / "metadata.db"),
                "MODEL_DIR": str(data_dir / "models"),
                "RAW_UPLOAD_DIR": str(data_dir / "raw_uploads"),
                "CANONICAL_DIR": str(data_dir / "canonical"),
                "GEO_DIR": str(BACKEND / "data" / "geo"),
                "ALLOW_LOCAL_RETRAIN": "true",
            }
        )
        for sample_name in ("gefs_reforecast_india_2019.parquet", "era5_observations_india_2019.parquet"):
            subprocess.run(
                [sys.executable, "-m", "app.ingestion.pipeline", "--file", f"data/samples/{sample_name}", "--confirm-all"],
                cwd=str(BACKEND),
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        subprocess.run(
            [sys.executable, "-m", "app.ml.train_pipeline", "--dry-run", "--emit-eval"],
            cwd=str(BACKEND),
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        run_dir = sorted((data_dir / "models").glob("run_*"))[-1]
        ev_path = sorted((data_dir / "analysis" / "eval_events").glob("*.parquet"))[-1]
        ev = pd.read_parquet(ev_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        metrics_json = json.loads((run_dir / "metrics.json").read_text())
        thresholds_json = json.loads((run_dir / "thresholds.json").read_text())

        shap_df = pd.read_parquet(run_dir / "shap_summary.parquet")
        top_clf_shap = (
            shap_df[(shap_df["model"] == "classifier") & (shap_df["group_region_id"] == "__all__")]
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        reg_top_driver: dict[str, tuple[str, float]] = {}
        for var in manifest["modelled_variables"]:
            sub = (
                shap_df[(shap_df["model"] == f"regressor::{var}") & (shap_df["group_region_id"] == "__all__")]
                .sort_values("mean_abs_shap", ascending=False)
            )
            if not sub.empty:
                reg_top_driver[var] = (str(sub.iloc[0]["feature"]), float(sub.iloc[0]["mean_abs_shap"]))

        fc = json.loads((run_dir / "feature_columns.json").read_text())
        cal = json.loads((run_dir / "calibrator.json").read_text())
        clf_cols = fc["classifier"]
        cat_cols = [c for c in ["state_id", "season"] if c in clf_cols]
        num_cols = [c for c in clf_cols if c not in cat_cols]

        tr = ev[ev["split"] == "train"].copy()
        va = ev[ev["split"] == "val"].copy()
        te = ev[ev["split"] == "test"].copy()
        y_tr = tr["y_bust"].astype(int).to_numpy()
        y_va = va["y_bust"].astype(int).to_numpy()
        y_te = te["y_bust"].astype(int).to_numpy()

        def score_probs(probs: np.ndarray) -> dict[str, float]:
            pred = (probs >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            spec = float(tn / (tn + fp))
            sens = float(recall_score(y_te, pred))
            return {
                "acc": float(accuracy_score(y_te, pred)),
                "bal_acc": float(balanced_accuracy_score(y_te, pred)),
                "sens": sens,
                "spec": spec,
                "prec": float(precision_score(y_te, pred, zero_division=0)),
                "f1": float(f1_score(y_te, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_te, probs)),
                "pr_auc": float(average_precision_score(y_te, probs)),
                "brier": float(brier_score_loss(y_te, probs)),
            }

        candidates: dict[str, dict[str, float]] = {}
        candidates["Two-Stage XGBoost + Platt"] = score_probs(te["model_proba"].to_numpy())

        # Single-stage ablation (without Stage-1 pred_err_* and conf_*)
        raw_cols = [c for c in clf_cols if not c.startswith(("pred_err_", "conf_"))]
        Xtr_raw, Xva_raw, Xte_raw = tr[raw_cols].copy(), va[raw_cols].copy(), te[raw_cols].copy()
        for df_ in (Xtr_raw, Xva_raw, Xte_raw):
            for c in raw_cols:
                if c in cat_cols:
                    df_[c] = df_[c].astype("category")
                else:
                    df_[c] = pd.to_numeric(df_[c], errors="coerce")
        m_ss = xgb.XGBClassifier(
            n_estimators=3000,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=15,
            reg_lambda=3.0,
            reg_alpha=1.0,
            gamma=0.5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=100,
        )
        m_ss.fit(Xtr_raw, y_tr, eval_set=[(Xva_raw, y_va)], verbose=False)
        candidates["Single-Stage XGBoost (no OOF)"] = score_probs(m_ss.predict_proba(Xte_raw)[:, 1])

        prep = ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
                (
                    "cat",
                    Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
                    cat_cols,
                ),
            ]
        )
        for cname, est in [
            (
                "Random Forest (300 trees)",
                RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1),
            ),
            ("L2 Logistic Regression", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]:
            pipe = Pipeline([("prep", prep), ("est", est)])
            pipe.fit(tr[clf_cols], y_tr)
            candidates[cname] = score_probs(pipe.predict_proba(te[clf_cols])[:, 1])

        return {
            "run_id": manifest["run_id"],
            "regressor_metrics": metrics_json["regressors"],
            "classifier_metrics": metrics_json["classifier"],
            "thresholds": thresholds_json["bust_threshold"],
            "calibrator": cal,
            "top_clf_shap": top_clf_shap,
            "reg_top_driver": reg_top_driver,
            "candidates": candidates,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Figures (clean academic 300-DPI style matching the Reference PDF)
# ---------------------------------------------------------------------------
def build_fig1_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.35), dpi=300)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 2.65)
    ax.axis("off")

    boxes = [
        (
            0.12,
            0.92,
            1.76,
            1.42,
            "#DAE3F3",
            HEX_BLUE_DARK,
            "1. Paired Store\n& Weight Table",
            "GEFSv12 (5 mem, D1–10)\n+ ERA5 reanalysis\n666 GADM districts\n10,373 cell weights\nSea-soil mask (18 dist.)",
        ),
        (
            2.12,
            0.92,
            1.76,
            1.42,
            "#DAE3F3",
            HEX_BLUE_DARK,
            "2. Embargo Split\n& Bias Removal",
            "Chronological cycle split\n10-day boundary embargo\nTrain-only monthly bias\nLabel v2: |e − b| > p90\n(7 surface variables)",
        ),
        (
            4.12,
            0.92,
            1.76,
            1.42,
            "#FCE4D6",
            HEX_ORANGE,
            "3. Stage-1 Error\nRegressors (×7)",
            "XGBoost hist, depth 4\n32 issue-time features\n30-day block OOF folds\nOutputs: pred_err_v &\nconf_v = 1 − ê_v / τ_v",
        ),
        (
            6.12,
            0.92,
            1.76,
            1.42,
            "#FCE4D6",
            HEX_ORANGE,
            "4. Stage-2 Bust\nClassifier + Platt",
            "XGBoost hist, depth 3\n66 event-level features\nEarly stop on val year\nPlatt logit calibration\n(a > 0 monotonicity)",
        ),
        (
            8.12,
            0.92,
            1.76,
            1.42,
            "#E2EFDA",
            HEX_GREEN,
            "5. Verification,\nGate & TreeSHAP",
            "Calibrated P(bust) bands\nExact TreeExplainer\n8-rung baseline ladder\nShared-event gate:\nΔAUC ≥ −0.05 to ship",
        ),
    ]

    for x, y, w, h, fc_hex, ec_hex, title, body in boxes:
        patch = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.07",
            facecolor=fc_hex,
            edgecolor=ec_hex,
            linewidth=1.15,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h - 0.23,
            title,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#111111",
            linespacing=1.12,
        )
        ax.plot([x + 0.10, x + w - 0.10], [y + h - 0.45, y + h - 0.45], color=ec_hex, lw=0.6, alpha=0.55)
        ax.text(
            x + w / 2,
            y + 0.47,
            body,
            ha="center",
            va="center",
            fontsize=6.45,
            color="#222222",
            linespacing=1.22,
        )

    for i in range(4):
        x0 = boxes[i][0] + boxes[i][2] + 0.03
        x1 = boxes[i + 1][0] - 0.03
        ym = 1.63
        ax.annotate(
            "",
            xy=(x1, ym),
            xytext=(x0, ym),
            arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color="#333333", lw=1.15),
        )

    # Bottom governance & leakage-firewall bar
    bar = mpatches.FancyBboxPatch(
        (0.12, 0.08),
        9.76,
        0.60,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#F7F9FC",
        edgecolor="#8FAADC",
        linewidth=0.85,
        linestyle="--",
    )
    ax.add_patch(bar)
    ax.text(
        5.0,
        0.38,
        "Issue-Time Firewall (contracts.OBSERVATION_DERIVED): zero post-00 UTC observation inputs at forecast scoring time\n"
        "Spatial challenger: 43,969-param Dilated BustCNN on 0.25° grids  •  "
        "Serving budget: precomputed cycles + standing TreeSHAP (176 / 512 MB)",
        ha="center",
        va="center",
        fontsize=6.35,
        linespacing=1.28,
        color="#1F3864",
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_fig2_performance(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75), dpi=300, gridspec_kw={"width_ratios": [1.28, 1.0, 1.02]})

    # Panel (a): Held-out 2017 ROC-AUC across 8-rung ladder + v1 issue-time + promoted v2
    ax = axes[0]
    labels_a = [
        "Climatology",
        "Lead day only",
        "Isotonic Dist. Reg. (IDR)",
        "Ensemble spread",
        "Lead + spread + season",
        "k-NN analog",
        "EMOS (Gaussian NGR)",
        "Pre-audit v1 (issue-time)",
        "District-season freq.",
        "Two-Stage XGBoost v2",
    ]
    vals_a = [0.5000, 0.5095, 0.5987, 0.5996, 0.6079, 0.6114, 0.6256, 0.6572, 0.6715, 0.7615]
    cols_a = [
        HEX_LIGHT_GRAY,
        HEX_LIGHT_GRAY,
        "#8FAADC",
        "#8FAADC",
        "#8FAADC",
        "#8FAADC",
        "#5B9BD5",
        HEX_ORANGE,
        "#2E74B5",
        HEX_NAVY,
    ]
    y_pos = np.arange(len(labels_a))
    bars = ax.barh(y_pos, vals_a, height=0.66, color=cols_a, edgecolor="#333333", linewidth=0.45)
    ax.axvline(0.50, color="#666666", linestyle="--", linewidth=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_a, fontsize=6.6)
    ax.set_xlim(0.45, 0.825)
    ax.set_xlabel("Held-out 2017 ROC-AUC (n = 2.43M)", fontsize=7.2)
    ax.set_title("(a) 17-Year Ladder & Audit Gain", fontsize=8.0, fontweight="bold", color=HEX_NAVY, pad=5)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, vals_a):
        fw = "bold" if v > 0.75 else "normal"
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=6.3, fontweight=fw)

    # Panel (b): 5-bin reliability curve on 2,430,900 held-out 2017 events
    ax2 = axes[1]
    pred_v2 = [0.1308, 0.2986, 0.4955, 0.6800, 0.8810]
    obs_v2 = [0.1162, 0.2870, 0.4907, 0.6890, 0.9027]
    # Uncalibrated v1 as-of-issue (documented over-confident top bin 0.888 vs 0.619; overall Brier 0.2342)
    pred_v1 = [0.118, 0.294, 0.496, 0.692, 0.888]
    obs_v1 = [0.278, 0.362, 0.448, 0.535, 0.619]

    ax2.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.85, label="Perfect (1:1)")
    ax2.plot(
        pred_v1,
        obs_v1,
        marker="s",
        markersize=3.8,
        linewidth=1.2,
        color=HEX_ORANGE,
        linestyle="-.",
        label="v1 uncal. (Brier 0.234)",
    )
    ax2.plot(
        pred_v2,
        obs_v2,
        marker="o",
        markersize=4.2,
        linewidth=1.5,
        color=HEX_NAVY,
        label="v2 + Platt (Brier 0.190)",
    )
    ax2.axhline(0.3895, color="#AAAAAA", linestyle=":", linewidth=0.75)
    ax2.text(0.42, 0.335, "2017 base rate (0.390)", fontsize=5.8, color="#666666")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Mean predicted P(bust)", fontsize=7.2)
    ax2.set_ylabel("Observed bust frequency", fontsize=7.2)
    ax2.set_title("(b) Calibration (2.43M events)", fontsize=8.0, fontweight="bold", color=HEX_NAVY, pad=5)
    ax2.tick_params(labelsize=6.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white", edgecolor="#CCCCCC")

    # Panel (c): Multi-year pool scaling on unseen 2019 & CNN vs XGBoost challenger
    ax3 = axes[2]
    cats = ["1-yr\n(2017)", "2-yr\n(17–18)", "3-yr\n(16–18)", "CNN\n(2018)", "XGBoost\n(2018)"]
    scores = [0.8231, 0.8418, 0.8487, 0.7454, 0.8711]
    bar_cols = ["#5B9BD5", "#2E74B5", HEX_NAVY, HEX_ORANGE, HEX_GREEN]
    x_pos = np.array([0.0, 0.95, 1.90, 3.05, 4.18])
    bars3 = ax3.bar(x_pos, scores, width=0.55, color=bar_cols, edgecolor="#333333", linewidth=0.45)
    ax3.axvline(2.48, color="#888888", linestyle=":", linewidth=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cats, fontsize=6.0)
    ax3.set_xlim(-0.52, 4.72)
    ax3.set_ylim(0.65, 0.915)
    ax3.set_ylabel("Held-out ROC-AUC", fontsize=7.2)
    ax3.set_title("(c) Scaling & CNN Challenger", fontsize=8.0, fontweight="bold", color=HEX_NAVY, pad=5)
    ax3.tick_params(axis="y", labelsize=6.6)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for b, s in zip(bars3, scores):
        ax3.text(b.get_x() + b.get_width() / 2, s + 0.006, f"{s:.3f}", ha="center", va="bottom", fontsize=6.1, fontweight="bold")
    ax3.text(0.95, 0.892, "Unseen 2019", ha="center", fontsize=6.0, color=HEX_BLUE_DARK, style="italic")
    ax3.text(3.56, 0.892, "Shared 2018", ha="center", fontsize=6.0, color="#444444", style="italic")

    fig.tight_layout(pad=0.7, w_pad=1.05)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_fig3_shap_and_regressors(out_path: Path, sample_eval: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.38), dpi=300, gridspec_kw={"width_ratios": [1.22, 1.0]})

    # Panel (a): Top 10 classifier TreeSHAP features
    ax1 = axes[0]
    top10 = sample_eval["top_clf_shap"].head(10).iloc[::-1]
    pretty_names = {
        "historical_bust_frequency_region_season": "District-season bust freq. (prior)",
        "state_id": "State categorical (state_id)",
        "month": "Calendar month",
        "laf_pool_mean_rainfall_mm": "Time-lagged pool mean: rainfall",
        "conf_humidity_pct": "Stage-1 OOF confidence: humidity",
        "pred_err_humidity_pct": "Stage-1 OOF pred. error: humidity",
        "laf_pool_mean_pressure_hpa": "Time-lagged pool mean: pressure",
        "laf_pool_std_atmospheric_moisture_kgm2": "Time-lagged pool SD: atmos. moist.",
        "laf_pool_mean_humidity_pct": "Time-lagged pool mean: humidity",
        "pred_err_temperature_c": "Stage-1 OOF pred. error: temp.",
        "conf_wind_speed_ms": "Stage-1 OOF confidence: wind spd.",
    }
    ylabels = [pretty_names.get(f, f) for f in top10["feature"]]
    vals = top10["mean_abs_shap"].to_numpy()
    cols = []
    for f in top10["feature"]:
        if f.startswith(("pred_err_", "conf_")):
            cols.append(HEX_ORANGE)
        elif f.startswith("laf_"):
            cols.append(HEX_GREEN)
        else:
            cols.append(HEX_BLUE)

    y = np.arange(len(ylabels))
    bars1 = ax1.barh(y, vals, height=0.66, color=cols, edgecolor="#333333", linewidth=0.45)
    ax1.set_yticks(y)
    ax1.set_yticklabels(ylabels, fontsize=6.5)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.set_xlabel("Mean |TreeSHAP| contribution (log-odds)", fontsize=7.2)
    ax1.set_title("(a) Stage-2 Classifier Global TreeSHAP (Top 10)", fontsize=8.0, fontweight="bold", color=HEX_NAVY, pad=5)
    ax1.tick_params(axis="x", labelsize=6.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for b, v in zip(bars1, vals):
        ax1.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=6.2)

    legend_handles = [
        mpatches.Patch(facecolor=HEX_BLUE, edgecolor="#333333", linewidth=0.4, label="Geography / calendar prior"),
        mpatches.Patch(facecolor=HEX_ORANGE, edgecolor="#333333", linewidth=0.4, label="Stage-1 OOF error / conf."),
        mpatches.Patch(facecolor=HEX_GREEN, edgecolor="#333333", linewidth=0.4, label="Time-lagged ensemble (LAF)"),
    ]
    ax1.legend(handles=legend_handles, loc="lower right", fontsize=5.8, frameon=True, facecolor="white", edgecolor="#CCCCCC")

    # Panel (b): Stage-1 per-variable MAE skill vs baseline mean-error predictor
    ax2 = axes[1]
    var_labels = {
        "rainfall_mm": "Rainfall (mm/d)",
        "wind_speed_ms": "Wind speed (m/s)",
        "atmospheric_moisture_kgm2": "Atmos. moisture (kg/m²)",
        "humidity_pct": "Rel. humidity (%)",
        "soil_moisture_pct": "Soil moisture (%)",
        "temperature_c": "Temperature (°C)",
        "pressure_hpa": "Surface pressure (hPa)",
    }
    rows_b = []
    for var, vlabel in var_labels.items():
        m = sample_eval["regressor_metrics"][var]
        mae_te = m["test"]["mae"]
        mae_base = m["test"]["baseline_mae_predict_mean"]
        pct_change = 100.0 * (mae_base - mae_te) / mae_base  # positive = MAE reduction vs baseline
        rows_b.append((vlabel, pct_change, m["test"]["r2"]))
    rows_b.sort(key=lambda r: r[1])

    yb = np.arange(len(rows_b))
    pct_vals = [r[1] for r in rows_b]
    bcols = [HEX_NAVY if v >= 0 else HEX_LIGHT_GRAY for v in pct_vals]
    bars2 = ax2.barh(yb, pct_vals, height=0.62, color=bcols, edgecolor="#333333", linewidth=0.45)
    ax2.axvline(0.0, color="#444444", linewidth=0.8)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([r[0] for r in rows_b], fontsize=6.6)
    ax2.set_xlim(-22, 52)
    ax2.set_xlabel("Held-out MAE reduction vs baseline (%)", fontsize=7.2)
    ax2.set_title("(b) Stage-1 Error Regressor Skill (2019 Sample)", fontsize=8.0, fontweight="bold", color=HEX_NAVY, pad=5)
    ax2.tick_params(axis="x", labelsize=6.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for b, ( _, pv, r2v) in zip(bars2, rows_b):
        if pv >= 0:
            ax2.text(pv + 1.0, b.get_y() + b.get_height() / 2, f"+{pv:.1f}% (R²={r2v:.2f})", va="center", fontsize=6.0)
        else:
            ax2.text(1.0, b.get_y() + b.get_height() / 2, f"{pv:.1f}% (12-cyc noise)", va="center", fontsize=5.9, color="#555555")

    fig.tight_layout(pad=0.7, w_pad=1.1)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ReportLab styles & table helpers (exact match to Reference PDF)
# ---------------------------------------------------------------------------
def make_styles(fonts: tuple[str, str, str, str]) -> dict[str, ParagraphStyle]:
    f_reg, f_bold, f_ital, _ = fonts
    return {
        "title": ParagraphStyle(
            "Title",
            fontName=f_bold,
            fontSize=17.0,
            leading=21.0,
            alignment=TA_CENTER,
            textColor=C_TITLE,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName=f_bold,
            fontSize=13.0,
            leading=16.5,
            alignment=TA_CENTER,
            textColor=C_SUBTITLE,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "Meta",
            fontName=f_reg,
            fontSize=10.0,
            leading=13.2,
            alignment=TA_CENTER,
            textColor=C_TEXT,
            spaceAfter=2,
        ),
        "meta_last": ParagraphStyle(
            "MetaLast",
            fontName=f_reg,
            fontSize=10.0,
            leading=13.2,
            alignment=TA_CENTER,
            textColor=C_TEXT,
            spaceAfter=11,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName=f_bold,
            fontSize=14.0,
            leading=17.2,
            alignment=TA_LEFT,
            textColor=C_HEADING,
            spaceBefore=9.0,
            spaceAfter=4.0,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName=f_bold,
            fontSize=12.0,
            leading=14.8,
            alignment=TA_LEFT,
            textColor=C_HEADING,
            spaceBefore=7.5,
            spaceAfter=3.2,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=f_reg,
            fontSize=10.8,
            leading=14.4,
            alignment=TA_JUSTIFY,
            textColor=C_TEXT,
            spaceAfter=4.4,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=f_reg,
            fontSize=10.6,
            leading=13.9,
            alignment=TA_JUSTIFY,
            leftIndent=24,
            bulletIndent=10,
            textColor=C_TEXT,
            spaceAfter=2.8,
        ),
        "eq": ParagraphStyle(
            "Equation",
            fontName=f_ital,
            fontSize=10.8,
            leading=14.2,
            alignment=TA_CENTER,
            textColor=C_TEXT,
            spaceBefore=3.5,
            spaceAfter=5.5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=f_ital,
            fontSize=9.6,
            leading=12.4,
            alignment=TA_CENTER,
            textColor=C_TEXT,
            spaceBefore=3.5,
            spaceAfter=7.5,
        ),
        "th_l": ParagraphStyle(
            "THLeft",
            fontName=f_bold,
            fontSize=9.2,
            leading=11.4,
            alignment=TA_LEFT,
            textColor=C_TEXT,
        ),
        "th_c": ParagraphStyle(
            "THCenter",
            fontName=f_bold,
            fontSize=9.2,
            leading=11.4,
            alignment=TA_CENTER,
            textColor=C_TEXT,
        ),
        "td_l": ParagraphStyle(
            "TDLeft",
            fontName=f_reg,
            fontSize=9.1,
            leading=11.4,
            alignment=TA_LEFT,
            textColor=C_TEXT,
        ),
        "td_c": ParagraphStyle(
            "TDCenter",
            fontName=f_reg,
            fontSize=9.1,
            leading=11.4,
            alignment=TA_CENTER,
            textColor=C_TEXT,
        ),
        "td_lb": ParagraphStyle(
            "TDLeftBold",
            fontName=f_bold,
            fontSize=9.1,
            leading=11.4,
            alignment=TA_LEFT,
            textColor=C_TEXT,
        ),
        "td_cb": ParagraphStyle(
            "TDCenterBold",
            fontName=f_bold,
            fontSize=9.1,
            leading=11.4,
            alignment=TA_CENTER,
            textColor=C_TEXT,
        ),
        "ref": ParagraphStyle(
            "Ref",
            fontName=f_reg,
            fontSize=9.5,
            leading=11.9,
            alignment=TA_LEFT,
            leftIndent=20,
            firstLineIndent=-20,
            textColor=C_TEXT,
            spaceAfter=1.8,
        ),
    }


def make_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    st: dict[str, ParagraphStyle],
    caption: str | None = None,
    left_cols: set[int] | None = None,
    bold_rows: set[int] | None = None,
) -> KeepTogether:
    if left_cols is None:
        left_cols = {0}
    if bold_rows is None:
        bold_rows = set()

    data = [[Paragraph(h, st["th_l"] if j in left_cols else st["th_c"]) for j, h in enumerate(headers)]]
    for i, r in enumerate(rows):
        is_b = i in bold_rows
        row_cells = []
        for j, cell in enumerate(r):
            style_key = ("td_lb" if is_b else "td_l") if j in left_cols else ("td_cb" if is_b else "td_c")
            row_cells.append(Paragraph(cell, st[style_key]))
        data.append(row_cells)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), C_TBL_HDR),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.48, C_TBL_GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.1),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.2),
            ]
        )
    )
    flow: list = [Spacer(1, 2), tbl]
    if caption:
        flow.append(Paragraph(caption, st["caption"]))
    else:
        flow.append(Spacer(1, 5))
    return KeepTogether(flow)


def make_figure(img_path: Path, width_pt: float, caption: str, st: dict[str, ParagraphStyle]) -> KeepTogether:
    from PIL import Image as PILImage

    with PILImage.open(img_path) as im:
        w_px, h_px = im.size
    height_pt = width_pt * (h_px / w_px)
    return KeepTogether([Spacer(1, 2), Image(str(img_path), width=width_pt, height=height_pt), Paragraph(caption, st["caption"])])


def draw_footer(canvas, doc, font_reg: str) -> None:
    canvas.saveState()
    canvas.setFont(font_reg, 9.1)
    canvas.setFillColor(C_TEXT)
    canvas.drawCentredString(PAGE_W / 2.0, 38.0, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Document narrative & build
# ---------------------------------------------------------------------------
def build_pdf(sample_eval: dict) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)

    fonts = register_fonts()
    st = make_styles(fonts)

    fig1 = ASSETS / "fig1_methodology_pipeline.png"
    fig2 = ASSETS / "fig2_performance_comparison.png"
    fig3 = ASSETS / "fig3_shap_and_regressors.png"
    build_fig1_pipeline(fig1)
    build_fig2_performance(fig2)
    build_fig3_shap_and_regressors(fig3, sample_eval)

    def B(txt: str) -> Paragraph:
        return Paragraph(txt, st["body"])

    def H1(txt: str) -> Paragraph:
        return Paragraph(txt, st["h1"])

    def H2(txt: str) -> Paragraph:
        return Paragraph(txt, st["h2"])

    def UL(items: list[str]) -> list[Paragraph]:
        return [Paragraph(it, st["bullet"], bulletText="•") for it in items]

    story: list = []

    # ---- Header Block (exact Reference format) ----
    story.append(Paragraph("METHODOLOGY AND MODEL PERFORMANCE<br/>REPORT", st["title"]))
    story.append(
        Paragraph(
            "Explainable AI for District-Level Prediction of Medium-Range<br/>Weather Forecast Busts over India",
            st["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "DN Harith, Dhanush Thota | Dept. of AI &amp; ML, Dayananda Sagar College of Engineering",
            st["meta"],
        )
    )
    story.append(Paragraph("Faculty Guide: Prof. Kusumika Krori Dutta | November 18, 2026", st["meta_last"]))

    # ---- Summary ----
    story.append(H1("Summary"))
    story.append(
        B(
            "This report documents how the Sanket medium-range forecast-bust early-warning model was constructed, "
            "audited for post-issue leakage, and evaluated across India's 666 GADM 4.1 districts over Day 1–10 lead "
            "times. Rather than replacing numerical weather prediction (NWP), Sanket treats forecast failure as a "
            "supervised reliability problem: seven per-variable XGBoost error regressors (Stage 1, out-of-fold over "
            "30-day cycle blocks) feed a Platt-calibrated XGBoost bust classifier (Stage 2) paired with exact "
            "TreeSHAP attributions. Trained on 17 years of NOAA GEFSv12 five-member reforecasts verified against "
            "ERA5 reanalysis (2000–2016; 6,193 cycles) with a 10-day boundary embargo and evaluated strictly as of "
            "00 UTC issue time on all 365 unseen cycles of 2017 (2,430,900 district-day-lead events; base rate "
            "38.95%), the promoted Label-v2 model (<i>run_20261006T200800Z</i>) achieves ROC-AUC <b>0.7615</b> "
            "(+0.1043 over the pre-audit deployment's honest issue-time 0.6572), PR-AUC <b>0.6784</b>, Brier score "
            "<b>0.1898</b> (Brier Skill Score <b>+0.2019</b> vs climatology), and F1 <b>0.5819</b>. It outperforms "
            "every rung of an eight-baseline verification ladder (climatology 0.5000, ensemble spread 0.5996, EMOS "
            "0.6256, district-season frequency 0.6715) and beats a 43,969-parameter dilated convolutional neural "
            "network (<i>BustCNN</i>) by 0.126–0.163 ROC-AUC across two independent years. Important caveats remain: "
            "17-year Label-v2 evaluation covers a single primary test year (with 2018–2019 cross-checked in multi-year "
            "experiments), archive spread comes from 5 of 31 GEFS members, and ERA5-versus-MERRA-2 reanalysis "
            "disagreement spans 23.5–43.5% of the bust threshold itself."
        )
    )

    # ---- 1. Methodology ----
    story.append(H1("1. Methodology"))
    story.append(H2("1.1 Problem and design"))
    story.append(
        B(
            "Medium-range NWP ensembles occasionally suffer synoptic-scale failures ('forecast busts') where realised "
            "weather departs sharply from ensemble mean guidance even when member spread appears modest [2]. Over India, "
            "where monsoon-trough shifts, Western Ghats orography, and Himalayan steep-terrain gradients govern district "
            "impacts, operational forecasters need an <i>a priori</i> trust signal at the moment a 00 UTC cycle is "
            "issued. For district <i>d</i>, surface variable <i>v</i>, initialization cycle <i>t</i><sub>0</sub>, and "
            "lead day <i>l</i> in {1, …, 10} (valid date <i>t</i> = <i>t</i><sub>0</sub> + <i>l</i> − 1), let "
            "<i>e<sub>d,v,l</sub></i> = |μ<sub><i>d,v,l</i></sub> − <i>y<sub>d,v</sub></i> − <i>b<sub>d,v,l,m</sub></i>| "
            "denote the bias-corrected ensemble-mean absolute error, where μ<sub><i>d,v,l</i></sub> is the 5-member "
            "mean and <i>b<sub>d,v,l,m</sub></i> is the standing monthly bias estimated strictly on training cycles. "
            "A variable-level bust occurs when "
            "<i>e<sub>d,v,l</sub></i> &gt; τ<sub><i>v</i></sub> (the 90th percentile of training errors for variable "
            "<i>v</i>), and an event-level bust occurs if any of the seven modelled surface variables busts on that "
            "(<i>d</i>, <i>t</i><sub>0</sub>, <i>l</i>) triple. The pipeline estimates calibrated <i>P</i>(bust) and "
            "decomposes each district-lead score into additive feature contributions via TreeSHAP [1, 10]."
        )
    )
    story.append(
        make_figure(
            fig1,
            BODY_W,
            "Figure 1. Two-stage leakage-free bust prediction, Platt calibration, and TreeSHAP attribution pipeline.",
            st,
        )
    )

    story.append(H2("1.2 Data"))
    story.append(
        make_table(
            ["Item", "Description"],
            [
                [
                    "Sources",
                    "NOAA GEFSv12 00 UTC reforecast (2000–2019, 0.25°, 5 members: <i>gec00</i>, <i>gep01–04</i>) [3] "
                    "paired with Copernicus ERA5 hourly reanalysis [4] aggregated to UTC daily district means; "
                    "static geography from GADM 4.1 (666 districts) and CGIAR-CSI SRTM v4 elevation.",
                ],
                [
                    "Archive scale",
                    "7,288 of 7,305 daily cycles ingested (242,690,400 district-member-lead rows; 1.30B variable-value "
                    "cells). Promoted v2 run spans 2000–2017 (6,558 cycles). Reproducible repository sample: 17 cycles "
                    "of 2019 across 36 capitals (30,600 forecast + 13,680 ERA5 rows; 5,950 scored events).",
                ],
                [
                    "Predictors",
                    "32 issue-time features per Stage-1 regressor (forecast value, 5-member spread, member count, "
                    "time-lagged ensemble pool mean/SD/size/ratio, cycle-to-cycle jumpiness, pressure/moisture rates, "
                    "5 GADM/SRTM descriptors, concurrent variables, state, season); 66 features in Stage 2.",
                ],
                [
                    "Target (Label v2)",
                    "Binary indicator <i>y</i><sub>bust</sub> in {0, 1} that ≥1 of 7 bias-corrected surface variables "
                    "exceeds its training-only 90th-percentile error τ<sub><i>v</i></sub>. Held-out 2017 base rate: "
                    "38.95% (946,820 / 2,430,900 events); 2019 sample test base rate: 39.81% (418 / 1,050).",
                ],
                [
                    "Regime contrast",
                    "Monsoon (JJAS) aggregate bust rate reaches 48.6% versus 28.5% in post-monsoon/winter; rainfall "
                    "error p90 jumps from 4.0 mm/day outside JJAS to 15.9 mm/day during JJAS.",
                ],
            ],
            [98, BODY_W - 98],
            st,
            left_cols={0, 1},
        )
    )

    story.append(H2("1.3 Data cleaning and audit cautions on the input store"))
    story.extend(
        UL(
            [
                "<b>Structural horizon caps vs corrupt values:</b> Observation parquets contain zero nulls. On the "
                "forecast side, 10 m wind components terminate at 120 h (Day 5) and topsoil moisture at 72 h (Day 3) "
                "in AWS GEFSv12; absence at longer leads is structural and never imputed. Two corrupt member-days in "
                "20 years (e.g., 2004-03-05 <i>gep02</i> encoding ~10<sup>5</sup> K) are stripped at ingestion by "
                "physical range bounds (<i>PLAUSIBLE_RANGE</i>), and any cycle short of 33,300 rows is quarantined.",
                "<b>Coastal sea-value soil contamination:</b> Coarse GEFS land-sea masks report 98–100% volumetric "
                "water over ocean cells while ERA5 masks them to 0%, fabricating 100% bust rates in island and coastal "
                "districts (Lakshadweep, Nicobar, Mahe, Yanam). Building <i>soil_land_mask.parquet</i> masks soil "
                "moisture on both sides for all 18 districts with &gt;5% GEFS sea weight.",
                "<b>Standing district bias and circular wind:</b> Variance decomposition (2026-09-25 audit) showed "
                "that 64% of temperature/humidity squared error and 90% of soil-moisture squared error is fixed "
                "district bias (GEFS runs ~16 %RH drier than ERA5 over peninsular districts). Label v2 subtracts "
                "training-only (<i>district</i>, <i>variable</i>, <i>lead</i>, <i>month</i>) bias before computing "
                "τ<sub><i>v</i></sub>, cutting November 2017 humidity p90 from 27.6 to 10.4 %RH, and excludes wind "
                "direction from the label after circular wrap-around reduction (191.0° → 115.1° p90).",
                "<b>Post-issue leakage and dedupe hygiene:</b> Auditing <i>run_20260922T043925Z</i> revealed that "
                "<i>forecast_error_lag</i> read realised lead <i>k</i>−1 error unavailable for Days 2–10 at 00 UTC "
                "(inflating test ROC-AUC from 0.6572 live to 0.8435 offline), while late-ingested IMD rainfall batches "
                "had overwritten ERA5 on 2016–2019 via timestamp dedupe despite a 1-day IST/UTC offset (<i>r</i> = "
                "0.753 shifted vs 0.442 unshifted). Both paths are now hard-blocked by <i>contracts.OBSERVATION_DERIVED</i> "
                "and batch quarantine; stale 106-day-lag OMI MJO indices (0.0% SHAP weight) were also removed.",
            ]
        )
    )

    story.append(H2("1.4 Predictand variables, lead horizons and bust thresholds"))
    story.append(
        B(
            "Table 1 lists the eight district-level surface variables and three grid-only convective/synoptic channels "
            "retained in the CNN grids. Under Label v2, seven surface variables define the aggregate bust target; wind "
            "direction remains an input feature because light-wind bearing flips reflect calm boundary layers rather "
            "than damaging synoptic busts."
        )
    )
    story.append(
        make_table(
            ["Variable", "Unit", "GEFS Lead", "Pipeline Role", "Raw p90 (v1)", "Bias-Corr. p90 (v2 / 2019)"],
            [
                ["Rainfall (<i>apcp_sfc</i>)", "mm/day", "Day 1–10", "Label + Stage 1/2 input", "9.33", "9.33 / 15.01"],
                ["2 m Temperature (<i>tmp_2m</i>)", "°C", "Day 1–10", "Label + Stage 1/2 input", "2.86", "1.85 / 2.42"],
                ["2 m Rel. Humidity (<i>spfh</i>/<i>d2m</i>)", "%", "Day 1–10", "Label + Stage 1/2 input", "27.65", "10.40 / 13.94"],
                ["Surface Pressure (<i>pres_sfc</i>)", "hPa", "Day 1–10", "Label + Stage 1/2 input", "2.30", "1.62 / 2.70"],
                ["Atmos. Moisture (<i>pwat_eatm</i>)", "kg/m²", "Day 1–10", "Label + Stage 1/2 input", "7.52", "6.10 / 6.33"],
                ["10 m Wind Speed (<i>u10</i>, <i>v10</i>)", "m/s", "Day 1–5", "Label + Stage 1/2 input", "1.72", "1.48 / 1.73"],
                ["Topsoil Moisture (<i>soilw</i>)", "%", "Day 1–3", "Label + Stage 1/2 input (masked)", "16.69", "6.85 / 7.72"],
                ["10 m Wind Direction", "deg", "Day 1–5", "Predictor input only (circular)", "137.44", "Excluded from v2 label"],
                ["Z500, CAPE, CIN", "m, J/kg", "Day 1–10", "0.25° BustCNN grid channels only", "—", "Spatial challenger only"],
            ],
            [112, 46, 50, 111, 54, BODY_W - 373],
            st,
            caption="Table 1. Meteorological variables, archive lead coverage, pipeline roles, and training-only 90th-percentile thresholds.",
            left_cols={0, 3},
        )
    )

    story.append(H2("1.5 Preprocessing and leakage-free splitting"))
    story.append(
        B(
            "Both forecast and reanalysis grids pass through the exact same 10,373-pair polygon-cell area-overlap "
            "weight table (<i>district_grid_weights.parquet</i>), preventing coast/orography interpolation mismatch. "
            "Under ERA5 estimator v2, derived quantities (Bolton relative humidity from <i>T</i> and <i>T<sub>d</sub></i>, "
            "vector wind speed/direction from <i>u</i><sub>10</sub> and <i>v</i><sub>10</sub>) are computed <i>after</i> "
            "daily and district averaging, reducing humidity discrepancies by 1.38 %RH via Jensen's inequality. "
            "Tree models consume native IEEE-754 <i>NaN</i> values without imputation, and natural class priors "
            "(~39% bust) are preserved without synthetic resampling or <i>scale_pos_weight</i> distortion."
        )
    )
    story.append(
        B(
            "All splits operate strictly on forecast initialization timestamps (<i>init_date</i>) with a hard "
            "<b>10-day boundary embargo</b> (<i>SPLIT_EMBARGO_DAYS = 10</i>) so a Day-10 forecast initialized before a "
            "cutoff never verifies on an observation date inside the next split. In the 17-year promoted run, "
            "2000–2015 forms the training pool (5,828 cycles; Stage-1 regressors fit on a fixed-seed uniform 2,000-cycle "
            "sample to fit an 83 GB commit ceiling), 2016 (365 cycles) serves as validation for early stopping and Platt "
            "calibration, and all 365 cycles of 2017 (2,430,900 events) form the held-out test set. Inside training, "
            "Stage-1 out-of-fold predictions are generated across <i>K</i> = 5 contiguous 30-day cycle blocks with "
            "10-day embargoes on both edges of every held-out block."
        )
    )

    story.append(H2("1.6 Two-stage XGBoost, Platt calibration, CNN challenger and TreeSHAP"))
    story.append(
        B(
            "In Stage 1, an independent histogram-binned XGBoost regressor (<i>reg:squarederror</i>) per variable "
            "<i>v</i> predicts expected bias-corrected absolute error <i>ê<sub>d,v,l</sub></i> from 32 issue-time "
            "predictors, yielding both <i>pred_err_v</i> and normalized confidence "
            "<i>conf_v</i> = clip(1 − <i>ê<sub>d,v,l</sub></i> / τ<sub><i>v</i></sub>, 0, 1). Because the historical "
            "archive holds only 5 ensemble members, 5-member spread has a 50% coefficient of variation (1/√(2(<i>M</i>−1))); "
            "Sanket augments spread with lagged-average forecast pooling (<i>laf_*</i>) stacking valid-date-matched "
            "forecasts across the three prior cycles (up to 20 member trajectories) and cycle-to-cycle revision "
            "jumpiness (<i>jump_*</i>). Stage 2 combines all 14 Stage-1 outputs with 52 spread, LAF, geographic, and "
            "seasonal-prior features in a shallow regularised XGBoost classifier (depth 3, <i>min_child_weight</i> = 15, "
            "L1 α = 1.0, L2 λ = 3.0). Validation logits undergo monotone Platt scaling:"
        )
    )
    story.append(Paragraph("p' = σ(a · logit(p) + b),    logit(p) = ln(p / (1 − p)),    a &gt; 0", st["eq"]))
    story.append(
        B(
            "where <i>a</i> ≤ 0 falls back to identity calibration. Local and global feature attributions are computed "
            "in log-odds space via exact <i>shap.TreeExplainer</i> [10] (<i>SHAP_ROWS_PER_GROUP</i> = 25 stratified "
            "rows per district-lead for Stage 1; full validation year for Stage 2). To test whether raw 2D spatial "
            "gradients outperform engineered tabular summaries, we also train <i>BustCNN</i>: a 4-block dilated "
            "convolutional encoder (dilations 1, 2, 4, 8 → 31-cell / ~850 km receptive field; GroupNorm + GELU; "
            "43,969 parameters) over 145 × 141 × 48 gridded tensors (24 data + 24 NaN-validity mask channels), "
            "projected onto the 666 districts via a parameter-free sparse <i>DistrictPooling</i> matrix built from "
            "the same weight table."
        )
    )
    story.append(
        make_table(
            ["Component", "Specification"],
            [
                [
                    "Stage-1 Regressors (×7)",
                    "<i>XGBRegressor</i> (<i>tree_method='hist'</i>, <i>enable_categorical=True</i>): up to 3,000 trees, "
                    "<i>max_depth</i> = 4, η = 0.03, subsample = 0.8, colsample = 0.8, <i>min_child_weight</i> = 10, "
                    "L1 α = 0.5, L2 λ = 2.0, early stop = 100 rounds; 5-fold 30-day block OOF + 10-day embargo.",
                ],
                [
                    "Stage-2 Classifier",
                    "<i>XGBClassifier</i> (<i>binary:logistic</i>, eval <i>logloss</i>): up to 3,000 trees, "
                    "<i>max_depth</i> = 3, η = 0.03, subsample = 0.8, colsample = 0.8, <i>min_child_weight</i> = 15, "
                    "γ = 0.5, L1 α = 1.0, L2 λ = 3.0, early stop = 100 rounds on validation year.",
                ],
                [
                    "Calibration & Risk Bands",
                    "Validation-set Platt scaling on clipped logits (ε = 10<sup>−6</sup>; sample fit <i>a</i> = 0.882, "
                    "<i>b</i> = −0.291). Operational risk cuts at 70th/90th validation percentiles (45% Watch, 81% Bust "
                    "on promoted 17-year run).",
                ],
                [
                    "Spatial Challenger (BustCNN)",
                    "PyTorch 4-layer dilated Conv2d (48 → 32 → 32 → 32 → 32, dilations 1, 2, 4, 8) + sparse district "
                    "pooling + 2-layer MLP head (43,969 params); 3-seed ensemble (42, 43, 44); ONNX serving export.",
                ],
                [
                    "Promotion Gate & Serving",
                    "Candidate runs must clear held-out ROC-AUC ≥ 0.55 and regress by ≤ 0.05 on shared test events "
                    "scored <i>as of issue time</i>. Precomputed cycles + standing SHAP keep Render RAM at 176 / 512 MB.",
                ],
            ],
            [115, BODY_W - 115],
            st,
            caption="Table 2. Hyperparameters, calibration rules, challenger architecture, and deployment constraints.",
            left_cols={0, 1},
        )
    )

    story.append(H2("1.7 Models and baselines compared"))
    story.extend(
        UL(
            [
                "<b>Two-stage XGBoost + Platt (promoted primary, <i>run_20261006T200800Z</i>):</b> leakage-free "
                "Label-v2 / Feature-v2 pipeline scored strictly at 00 UTC issue time across all 666 districts.",
                "<b>Pre-audit Two-stage XGBoost (<i>run_20260922T043925Z</i>):</b> Label-v1 uncalibrated deployment "
                "evaluated both as originally logged (with <i>forecast_error_lag</i> populated) and honestly as of "
                "issue time (<i>forecast_error_lag</i> blanked for leads ≥ 2) on identical 2017 events.",
                "<b>Candidate tabular baselines (from the Model Selection stage):</b> Single-stage XGBoost (ablating "
                "Stage-1 OOF inputs), bagged Random Forest (300 trees, depth 8, leaf ≥ 10), and L2 Logistic Regression "
                "(<i>C</i> = 1.0, median-imputed + standardised + one-hot encoded) on identical chronological splits.",
                "<b>Spatial deep-learning challenger (<i>BustCNN</i>):</b> 1-seed and 3-seed dilated CNN ensembles "
                "evaluated head-to-head against tabular XGBoost on identical 2017 and 2018 district cycles.",
                "<b>Eight-rung statistical post-processing ladder:</b> (1) constant Climatology; (2) zero-forecast "
                "District-Season Historical Frequency; (3) Lead Day logistic; (4) Ensemble Spread logistic; "
                "(5) Lead + Spread + Season interaction logistic; (6) Non-homogeneous Gaussian Regression / EMOS "
                "(σ<sub><i>v</i></sub> = <i>a<sub>v</sub></i> + <i>b<sub>v</sub>s<sub>v</sub></i>, analytic tail "
                "exceedance combined via fitted dependence exponent; half-normal for rainfall) [5]; (7) parameter-free "
                "Isotonic Distributional Regression (IDR); and (8) <i>k</i> = 25 Nearest-Neighbour Analog.",
            ]
        )
    )

    story.append(H2("1.8 Evaluation protocol"))
    story.append(
        B(
            "Discrimination is evaluated via trapezoidal ROC-AUC, NCMRWF binormal Z-AUC [7], and PR-AUC. Probabilistic "
            "quality is measured by the Brier score (BS), Brier Skill Score against training climatology "
            "(BSS = 1 − BS / BS<sub>clim</sub>), and the CORP pool-adjacent-violators decomposition "
            "BS = MCB − DSC + UNC into miscalibration (MCB ↓), discrimination (DSC ↑), and base-rate uncertainty [6]. "
            "Thresholded metrics at 0.50 report accuracy, balanced accuracy, sensitivity (recall), specificity, "
            "precision, F1, and the Symmetric Extremal Dependence Index (SEDI). Uncertainty is quantified via "
            "1,000-resample 30-day circular block bootstraps over whole forecast cycles (never rows, which share "
            "synoptic state) and split conformal prediction sets calibrated on validation at α = 0.10 (90% target)."
        )
    )

    # ---- 2. Model Performance ----
    story.append(H1("2. Model Performance"))
    story.append(H2("2.1 Stage-1 per-variable error regressors and candidate tabular models"))
    story.append(
        B(
            "Table 3 reports Stage-1 per-variable regression accuracy alongside the four candidate tabular classifiers "
            "on the reproducible 2019 sample split (12 train / 2 validation / 3 test cycles; 4,200 / 700 / 1,050 "
            "events). Stage 1 delivers substantial held-out MAE reductions over the training-mean error predictor on "
            "convective and advective variables—rainfall (−41.9%, <i>R</i><sup>2</sup> = 0.395), 10 m wind speed "
            "(−25.4%, <i>R</i><sup>2</sup> = 0.398), column moisture (−16.4%, <i>R</i><sup>2</sup> = 0.290), and "
            "relative humidity (−11.6%, <i>R</i><sup>2</sup> = 0.185)—while temperature and surface pressure exhibit "
            "near-zero skill on only 12 training cycles once monthly bias is already removed."
        )
    )

    reg_rows = []
    reg_display = [
        ("rainfall_mm", "Rainfall (mm/day)"),
        ("wind_speed_ms", "10 m Wind speed (m/s)"),
        ("atmospheric_moisture_kgm2", "Atmos. moisture (kg/m²)"),
        ("humidity_pct", "2 m Rel. humidity (%)"),
        ("soil_moisture_pct", "Soil moisture (%)"),
        ("temperature_c", "2 m Temperature (°C)"),
        ("pressure_hpa", "Surface pressure (hPa)"),
    ]
    for vkey, vname in reg_display:
        rm = sample_eval["regressor_metrics"][vkey]
        tau = sample_eval["thresholds"][vkey]
        drv_name, drv_val = sample_eval["reg_top_driver"][vkey]
        reg_rows.append(
            [
                vname,
                f"{tau:.2f}",
                f"{rm['val']['mae']:.2f}",
                f"{rm['test']['mae']:.2f}",
                f"{rm['test']['baseline_mae_predict_mean']:.2f}",
                f"{rm['test']['r2']:.3f}",
                f"<i>{escape(drv_name)}</i> ({drv_val:.2f})",
            ]
        )
    story.append(
        make_table(
            ["Stage-1 Variable", "τ_v (p90)", "Val MAE", "Test MAE", "Base MAE", "Test R²", "Top TreeSHAP Driver (|SHAP|)"],
            reg_rows,
            [106, 44, 44, 46, 46, 44, BODY_W - 330],
            st,
            caption="Table 3a. Stage-1 per-variable XGBoost error regressor performance on the chronological 2019 sample split.",
            left_cols={0, 6},
        )
    )

    cand_order = [
        "Two-Stage XGBoost + Platt",
        "Single-Stage XGBoost (no OOF)",
        "Random Forest (300 trees)",
        "L2 Logistic Regression",
    ]
    cand_rows = []
    for cname in cand_order:
        cm = sample_eval["candidates"][cname]
        cand_rows.append(
            [
                cname,
                f"{cm['acc']:.3f}",
                f"{cm['bal_acc']:.3f}",
                f"{cm['sens']:.3f}",
                f"{cm['spec']:.3f}",
                f"{cm['f1']:.3f}",
                f"{cm['roc_auc']:.4f}",
                f"{cm['pr_auc']:.4f}",
                f"{cm['brier']:.4f}",
            ]
        )
    story.append(
        make_table(
            ["Candidate Model (2019 Sample)", "Acc.", "Bal. Acc.", "Sens.", "Spec.", "F1", "ROC-AUC", "PR-AUC", "Brier ↓"],
            cand_rows,
            [120, 34, 43, 34, 34, 34, 56, 49, BODY_W - 404],
            st,
            caption="Table 3b. Candidate classifier ablation and comparison on the identical 2019 sample test split (n = 1,050).",
            left_cols={0},
            bold_rows={0},
        )
    )
    story.append(
        B(
            "Ablating the 14 Stage-1 OOF features drops XGBoost test ROC-AUC from <b>0.6447</b> to <b>0.6056</b> "
            "(−0.0391) and worsens Brier score from 0.2291 to 0.2351, confirming that out-of-fold per-variable error "
            "estimates carry genuine signal beyond raw spread and geography. On this 12-cycle miniature sample, "
            "bagged Random Forest (0.6798 AUC, 0.2154 Brier) slightly edges boosted trees due to lower variance at "
            "small cycle counts (within the ±0.042 cycle-bootstrap CI), whereas L2 Logistic Regression suffers "
            "threshold collapse (sensitivity 0.034, F1 0.065, Brier 0.3144). At multi-year archive scale across 666 "
            "districts—where native missingness in <i>laf_*</i>, Day 6–10 winds, and masked coastal soil cannot be "
            "handled without lossy imputation in scikit-learn—histogram XGBoost becomes the decisive production choice."
        )
    )

    story.append(H2("2.2 Promoted 17-year model and the eight-baseline ladder on held-out 2017"))
    story.append(
        B(
            "Table 4 benchmarks the promoted 17-year model (<i>run_20261006T200800Z</i>) against the pre-overhaul "
            "deployment (<i>run_20260922T043925Z</i>) and all eight verification baselines on the complete 2017 test "
            "year (365 cycles, 2,430,900 district-day-lead events across 666 districts), alongside CORP and SEDI "
            "diagnostics from the reproducible 2019 sample ladder."
        )
    )
    story.append(
        make_table(
            [
                "Model / Baseline Rung",
                "2017 AUC (n=2.43M)",
                "2017 Brier ↓",
                "2017 BSS ↑",
                "2017 F1",
                "Sample AUC",
                "Sample MCB ↓ / DSC ↑",
                "Sample SEDI",
            ],
            [
                ["Climatology (training base rate)", "0.5000", "0.2378", "0.0000", "0.0000", "0.5000", "0.0000 / 0.0000", "0.0000"],
                ["Lead day only (logistic)", "0.5095", "—", "—", "—", "0.5187", "0.0030 / 0.0028", "0.0000"],
                ["Isotonic Distributional Reg. (IDR)", "0.5987", "—", "—", "—", "0.5522", "0.0092 / 0.0036", "0.1298"],
                ["Ensemble spread only (logistic)", "0.5996", "—", "—", "—", "0.5574", "0.0050 / 0.0074", "0.1562"],
                ["Lead + spread + season (logistic)", "0.6079", "—", "—", "—", "0.5363", "0.0076 / 0.0071", "0.1288"],
                ["<i>k</i>-NN analog (<i>k</i> = 25)", "0.6114", "—", "—", "—", "0.5053", "0.0161 / 0.0017", "0.0350"],
                ["EMOS / Gaussian NGR [5]", "0.6256", "—", "—", "—", "0.5822", "0.0194 / 0.0113", "0.1851"],
                ["District-season hist. frequency", "0.6715", "—", "—", "—", "0.4967", "0.0034 / 0.0000", "0.0000"],
                ["Pre-audit v1 (honest issue-time)", "0.6572", "0.2342", "+0.0151", "0.3712", "—", "—", "—"],
                ["Pre-audit v1 (lag-leaked offline ref.)*", "0.8435*", "0.1630*", "+0.3145*", "0.7460*", "—", "—", "—"],
                ["Two-stage XGBoost v2 + Platt", "0.7615", "0.1898", "+0.2019", "0.5819", "0.6447", "0.0142 / 0.0246", "0.3533"],
            ],
            [130, 54, 46, 46, 43, 42, 45, BODY_W - 406],
            st,
            caption="Table 4. Full-scale 2017 verification (2,430,900 events; fit on 13.32M training events) and 2019 sample CORP/SEDI diagnostics. *Offline leaked score shown strictly for audit transparency; invalid at forecast issue time.",
            left_cols={0},
            bold_rows={10},
        )
    )
    story.append(
        B(
            "Three structural results stand out in Table 4 and Figure 2. First, <i>district_season_frequency</i> "
            "reaches <b>0.6715</b> ROC-AUC on 2017 without reading a single forecast number, outperforming every "
            "ensemble-spread and EMOS post-processor (≤ 0.6256) and demonstrating how strongly geography and monsoon "
            "seasonality govern raw forecast failure over India; meanwhile, lead-day correlation with percentile-normalised "
            "busts is weak (+0.076 train, +0.032 test), confirming the classifier is not merely rediscovering lead "
            "degradation. Second, scoring <i>run_20260922T043925Z</i> honestly at 00 UTC issue time revealed that its "
            "live ROC-AUC was <b>0.6572</b>—below the zero-forecast district-season table—whereas retraining cleanly "
            "under Label v2 with Platt scaling lifts honest issue-time ROC-AUC by <b>+0.1043</b> to <b>0.7615</b> "
            "(PR-AUC 0.6784, BSS +0.2019), beating <i>district_season_frequency</i> by +0.0900 and EMOS by +0.1359. "
            "Third, accounting for synoptic persistence via 30-cycle circular block bootstrapping widens the 95% "
            "ROC-AUC confidence interval 4.5× compared with naive single-cycle resampling (width <b>0.0438</b> vs "
            "0.0098; 10-cycle width 0.0334), establishing ±0.022 as the minimum meaningful multi-week skill gap."
        )
    )
    story.append(
        make_figure(
            fig2,
            BODY_W,
            "Figure 2. (a) Held-out 2017 ROC-AUC across the 8-baseline ladder and audited models; (b) 5-bin Platt reliability on 2.43M events; (c) multi-year pool scaling on unseen 2019 and CNN vs XGBoost head-to-head.",
            st,
        )
    )

    story.append(H2("2.3 Spatial challenger comparison: Dilated CNN versus tabular XGBoost"))
    story.append(
        B(
            "Table 5 compares <i>BustCNN</i> directly against Two-Stage XGBoost on identical chronological cycle splits "
            "across two independent years (2017 single-seed and 2018 three-seed ensemble; all runs evaluated prior to "
            "the Label-v2 bias subtraction on identical targets so the architectural comparison is strictly controlled). "
            "Tabular XGBoost outperforms the dilated CNN by <b>+0.1257</b> ROC-AUC on 2018 (0.8711 vs 0.7454) and "
            "<b>+0.1626</b> on 2017 (0.8466 vs 0.6840). Three-seed ensembling reduces CNN variance (+0.0079 over the "
            "best individual seed, 0.7375) without closing the structural gap: ~2,550 grid-day training frames offer "
            "limited sample efficiency for 43,969 convolutional weights compared with millions of district-day rows "
            "and domain-engineered lagged/tendency features in gradient-boosted trees, while adding +51–57 MB RSS and "
            "65–490 ms latency per cycle. <i>BustCNN</i> is therefore retained as an offline research benchmark rather "
            "than promoted to the 512 MB serving container."
        )
    )
    story.append(
        make_table(
            ["Architecture", "Eval Year / Seeds", "Train Scale", "Train AUC", "Val AUC", "Test AUC", "Serving Footprint"],
            [
                ["BustCNN (4-layer dilated)", "2017 (seed 42)", "2,550 grid-days", "0.7858", "0.7356", "0.6840", "+51–57 MB ONNX, 65–490 ms"],
                ["Two-stage XGBoost (tabular)", "2017 (same cycles)", "1.68M dist.-days", "—", "—", "0.8466", "Included in 176 MB base"],
                ["BustCNN (3-seed ensemble)", "2018 (seeds 42–44)", "2,550 grid-days", "0.8065", "0.7681", "0.7454", "3× encoder pass (CPU)"],
                ["Two-stage XGBoost (tabular)", "2018 (same cycles)", "1.68M dist.-days", "—", "—", "0.8711", "Included in 176 MB base"],
            ],
            [115, 76, 68, 44, 42, 44, BODY_W - 389],
            st,
            caption="Table 5. Controlled head-to-head comparison of the 43,969-parameter Dilated CNN and Tabular Two-Stage XGBoost.",
            left_cols={0, 1, 6},
            bold_rows={1, 3},
        )
    )

    story.append(H2("2.4 Multi-year scaling and seasonal generalisation"))
    story.append(
        B(
            "Single-year chronological splits always place November–December—India's calmest post-monsoon regime—in "
            "the test window, overstating annual skill by ~0.04 ROC-AUC relative to full-year scoring. Table 6 "
            "isolates cross-year generalisation and training-pool scaling on all 2,400,930 unseen events of 2019 "
            "(fixing 2018 as validation for early stopping across multi-year runs). Expanding the training archive "
            "from one year to three consecutive years monotonically lifts unseen-2019 ROC-AUC from <b>0.8231–0.8327</b> "
            "to <b>0.8487</b> (+0.0160 over the strongest single year), while monthly breakdown confirms the "
            "meteorological difficulty gradient: convective southwest-monsoon August dips to <b>0.741</b> ROC-AUC "
            "versus <b>0.870</b> in synoptic post-monsoon November."
        )
    )
    story.append(
        make_table(
            ["Experiment / Training Window", "Val Window", "Unseen Test Window", "Test Events", "Held-out ROC-AUC"],
            [
                ["Cross-year 1-yr: Train 2017 (Jan–Aug)", "Sep–Oct 2017", "Full year 2018", "2,424,240", "0.8348"],
                ["Single-year pool: Train 2016", "Full year 2018", "Full year 2019", "2,400,930", "0.8302"],
                ["Single-year pool: Train 2017", "Full year 2018", "Full year 2019", "2,400,930", "0.8231"],
                ["Single-year pool: Train 2018", "Sep–Oct 2018", "Full year 2019", "2,400,930", "0.8327"],
                ["Two-year pool: Train 2017 + 2018", "Late 2018", "Full year 2019", "2,400,930", "0.8418"],
                ["Three-year pool: Train 2016 + 2017 + 2018", "Full year 2018*", "Full year 2019", "2,400,930", "0.8487"],
                ["Seasonal slice (2018→2019): Peak monsoon (Aug) vs Post-monsoon (Nov)", "Sep–Oct 2018", "Aug / Nov 2019", "~206k / mo", "0.7410 / 0.8700"],
            ],
            [180, 68, 76, 62, BODY_W - 386],
            st,
            caption="Table 6. Cross-year transfer, multi-year training scaling on unseen 2019, and seasonal monsoon skill contrast (*2016–17 fit, 2018 val).",
            left_cols={0},
            bold_rows={5},
        )
    )

    story.append(H2("2.5 Error profile, reliability and historical bust replay"))
    story.append(
        B(
            "Table 7 details the confusion matrix at the standard 0.50 probability threshold and the five-bin Platt "
            "reliability profile on all 2,430,900 held-out 2017 events (946,820 busts; 1,484,080 non-busts). Compared "
            "with the uncalibrated v1 model scored at issue time, the promoted Label-v2 model nearly doubles true "
            "positives from 258,114 to <b>494,619</b>—lifting bust sensitivity from 0.2726 to <b>0.5224</b> and "
            "precision from 0.5814 to <b>0.6568</b>—while maintaining <b>0.8259</b> specificity (accuracy 0.7077, "
            "balanced accuracy 0.6741). Whereas uncalibrated v1 suffered severe top-bin over-confidence (mean predicted "
            "0.888 vs 0.619 observed in [0.8, 1.0]), Platt calibration aligns predicted and observed frequencies "
            "within ≤ 0.022 across all five probability bins (0.8810 predicted vs 0.9027 observed across 100,143 "
            "high-risk alerts). On the 2019 sample, split conformal sets calibrated at α = 0.10 achieve 0.833 empirical "
            "coverage (mean set size 1.461 across 3 held-out cycles)."
        )
    )
    story.append(
        make_table(
            ["Model / Calibration Bin (Held-out 2017)", "Events (n)", "TN / FP or Pred.", "FN / TP or Obs.", "Key Rate / Deviation"],
            [
                [
                    "Pre-audit v1 (<i>run_20260922T043925Z</i>, issue-time)",
                    "2,430,900",
                    "TN 1,298,215 / FP 185,865",
                    "FN 688,706 / TP 258,114",
                    "Sens. 0.273 | Spec. 0.875 | Acc. 0.640",
                ],
                [
                    "Promoted v2 (<i>run_20261006T200800Z</i>, Platt)",
                    "2,430,900",
                    "TN 1,225,647 / FP 258,433",
                    "FN 452,201 / TP 494,619",
                    "Sens. 0.522 | Spec. 0.826 | Acc. 0.708",
                ],
                ["Platt Bin 1: <i>P</i>(bust) in [0.0, 0.2)", "503,268", "Mean pred: 0.1308", "Obs. rate: 0.1162", "Gap: +0.0146 (Low risk)"],
                ["Platt Bin 2: <i>P</i>(bust) in [0.2, 0.4)", "808,292", "Mean pred: 0.2986", "Obs. rate: 0.2870", "Gap: +0.0116 (Low risk)"],
                ["Platt Bin 3: <i>P</i>(bust) in [0.4, 0.6)", "687,203", "Mean pred: 0.4955", "Obs. rate: 0.4907", "Gap: +0.0048 (Watch transition)"],
                ["Platt Bin 4: <i>P</i>(bust) in [0.6, 0.8)", "331,994", "Mean pred: 0.6800", "Obs. rate: 0.6890", "Gap: −0.0090 (Watch band)"],
                ["Platt Bin 5: <i>P</i>(bust) in [0.8, 1.0]", "100,143", "Mean pred: 0.8810", "Obs. rate: 0.9027", "Gap: −0.0217 (Bust alert ≥ 0.81)"],
            ],
            [152, 52, 92, 86, BODY_W - 382],
            st,
            caption="Table 7. Exact confusion counts (@0.50) and 5-bin Platt reliability across all 2,430,900 held-out 2017 district-day-lead events.",
            left_cols={0, 2, 3, 4},
            bold_rows={1},
        )
    )
    story.append(
        B(
            "Regime-stratified inspection confirms physical coherence: dry rainfall forecasts (&lt;2 mm/day) bust only "
            "0.5–0.7% of the time, whereas events where Stage-1 predicted rain error exceeds 9 mm verify as actual "
            "rainfall busts 49–58% of the time. In out-of-sample historical replay of the <b>August 2018 Kerala floods</b> "
            "(2018 outside the 2000–2017 training/test span), scoring the 2018-08-13 00 UTC cycle at Day-3 lead "
            "(valid 2018-08-15) over <b>Idukki</b>—where GEFSv12 forecast 35.3 mm against 145.0 mm ERA5 rainfall "
            "(109.7 mm error)—assigns a <b>0.71</b> issue-time bust probability, placing the Western Ghats escarpment "
            "firmly in the operational Watch/Bust regime three days ahead."
        )
    )

    story.append(H2("2.6 Explainability and feature attribution behaviour"))
    story.append(
        B(
            "Figure 3a shows global TreeSHAP attributions (mean |SHAP| in log-odds) for the Stage-2 bust classifier. "
            "Rather than relying on any single heuristic, the classifier integrates three distinct physical information "
            "streams: (i) geographical and seasonal vulnerability priors (<i>historical_bust_frequency_region_season</i> "
            "0.320, <i>state_id</i> 0.254, <i>month</i> 0.241); (ii) time-lagged ensemble trajectory consistency across "
            "successive 00 UTC cycles (<i>laf_pool_mean_rainfall_mm</i> 0.146, <i>laf_pool_mean_pressure_hpa</i> 0.088, "
            "<i>laf_pool_std_atmospheric_moisture_kgm2</i> 0.069); and (iii) Stage-1 out-of-fold error and confidence "
            "estimates (<i>conf_humidity_pct</i> 0.107, <i>pred_err_humidity_pct</i> 0.098, <i>pred_err_temperature_c</i> "
            "0.057, <i>conf_wind_speed_ms</i> 0.050, <i>pred_err_rainfall_mm</i> 0.043). Inside Stage 1, TreeSHAP "
            "confirms physically sensible drivers: <i>forecast_value</i> and <i>laf_pool_mean</i> dominate rainfall "
            "(3.22 and 1.33 mm |SHAP|), whereas <i>ensemble_spread</i> leads column moisture (0.85 kg/m²)."
        )
    )
    story.append(
        B(
            "TreeSHAP also proved operationally diagnostic during model auditing: inspecting per-lead Shapley values "
            "revealed a −0.36 log-odds step-down on Days 6–10 traced to <i>pred_err_wind_speed_ms</i> turning <i>NaN</i> "
            "once GEFS wind terminates at Day 5 (reflecting the mechanical drop from 7 to 5 active label variables), "
            "and exposed that stale 106-day-lag OMI MJO features carried 0.0% of classifier |SHAP| before removal. "
            "Crucially, however, our post-mortem established that <b>classifier-level SHAP cannot detect a Stage-1 "
            "input leak</b>: when <i>forecast_error_lag</i> leaked into Stage 1, Stage-2 SHAP simply credited "
            "<i>conf_humidity_pct</i> as an unusually accurate feature, motivating code-level input contracts "
            "(<i>contracts.OBSERVATION_DERIVED</i>) alongside XAI."
        )
    )
    story.append(
        make_figure(
            fig3,
            BODY_W,
            "Figure 3. (a) Global TreeSHAP feature importance of the Stage-2 bust classifier; (b) Stage-1 per-variable MAE skill on the 2019 sample.",
            st,
        )
    )

    story.append(H2("2.7 Sensitivity: leakage removal, truth product and sea masking"))
    story.append(
        make_table(
            ["Controlled Sensitivity Test", "Evaluation Slice", "Reference / Unmasked", "Treated / Audited", "Interpretation"],
            [
                [
                    "Issue-time vs lag-filled scoring",
                    "Sep 2017 (199,800 events)",
                    "0.797 AUC (Day 1: 0.840)",
                    "0.699 AUC (Day 1: 0.840)",
                    "Identical at Day 1; −0.098 artificial gain on Days 2–10 removed.",
                ],
                [
                    "IMD 0.25° gauge vs ERA5 rain truth",
                    "Train 2018 → Test 2019",
                    "0.8327 AUC (ERA5 rain)",
                    "0.8328 AUC (IMD rain)",
                    "ΔAUC = +0.0001: self-referential p90 absorbs product scale.",
                ],
                [
                    "Coastal sea-soil district exclusion",
                    "2018 test (643 land vs 18 sea)",
                    "0.8411 AUC | Brier 0.1656",
                    "0.8550 AUC | Brier 0.1582",
                    "Sea-only slice scored 0.7209; soil mask removes false busts.",
                ],
            ],
            [115, 82, 86, 86, BODY_W - 369],
            st,
            caption="Table 8. Sensitivity of bust discrimination to issue-time lag blanking, verification rainfall product, and sea-soil masking.",
            left_cols={0, 4},
        )
    )

    # ---- 3. Discussion and Limitations ----
    story.append(H1("3. Discussion and Limitations"))
    story.append(
        B(
            "Across full-year, multi-year, and archived-sample evaluations, Sanket demonstrates that medium-range "
            "forecast busts over Indian districts are meaningfully predictable at 00 UTC issue time when standing "
            "district bias, post-issue observation leaks, and coastal land-sea mismatches are removed. At the same "
            "time, several methodological and data constraints bound what these results can claim:"
        )
    )
    story.extend(
        UL(
            [
                "<b>Single Label-v2 test year on the 17-year run:</b> The promoted 17-year model is evaluated on all "
                "365 cycles of 2017 (2.43M events). While cross-year generalisation across 2018 and 2019 (2.40M events) "
                "was verified separately under Label v1 (0.833–0.849 AUC), full Label-v2 + ERA5-estimator-v2 rescoring "
                "on 2018–2019 remains pending.",
                "<b>Reanalysis ground-truth floor:</b> Direct comparison of ERA5 against NASA MERRA-2 across 21,492 "
                "Indian city-days shows inter-reanalysis MAE equals <b>23.5–43.5%</b> of each variable's bust threshold "
                "(e.g., 4.06 mm vs 9.33 mm for rainfall; 1.13 °C vs 2.86 °C for temperature), and ERA5 smooths extreme "
                "coastal cloudbursts (recording 57 mm vs ~490 mm gauge rainfall during the 1 Dec 2015 Chennai flood).",
                "<b>Five-member reforecast spread and staggered lead horizons:</b> Archive GEFSv12 provides 5 of 31 "
                "operational members (50% spread sampling noise), and wind/soil fields stop at Days 5 and 3, creating "
                "mechanical step-downs in the any-variable bust rate from Days 1–3 (7 variables) to Days 6–10 (5 variables).",
                "<b>National pooled thresholds and sparse pre-2017 jumpiness:</b> Although Label v2 removes standing "
                "bias, τ<sub><i>v</i></sub> remains pooled nationally; high-altitude Leh still exhibits 5.32 hPa "
                "pressure error versus the 0.555 hPa national median. Pre-2017 sparse backfills also leave cycle-to-cycle "
                "<i>jump_*</i> features mostly <i>NaN</i> during training.",
                "<b>Regressor cycle sub-sampling and validation reuse:</b> Stage-1 regressors fit on 2,000 of 5,828 "
                "training cycles to stay within CI memory/time limits, and 2016 serves jointly for early stopping, Platt "
                "scaling, and conformal calibration.",
                "<b>Operational TreeSHAP approximations:</b> To respect the 512 MB serving ceiling (live scoring peaks "
                "at 1,406 MB), district detail panels serve pre-aggregated validation-year TreeSHAP profiles rather than "
                "on-the-fly per-cycle Shapley tensors, and all SHAP attributions remain associative rather than causal.",
            ]
        )
    )
    story.append(
        B(
            "<b>Next steps:</b> (1) train per-variable bust probability heads—starting with heavy monsoon rainfall—so "
            "staggered Day-3/Day-5 variable drop-offs no longer perturb aggregate risk; (2) rescore 2018–2019 end-to-end "
            "under Label v2 with ERA5 estimator v2 and UTC-aligned IMD gauge rainfall; (3) replace pooled pressure "
            "thresholds with district-stratified quantiles; and (4) ingest real-time ROMI MJO indices available at "
            "00 UTC issue time."
        )
    )

    # ---- 4. Conclusion ----
    story.append(H1("4. Conclusion"))
    story.append(
        B(
            "Sanket establishes a rigorous, end-to-end explainable post-processing framework for predicting Day 1–10 "
            "medium-range weather forecast busts across all 666 Indian districts. By pairing a single polygon-overlap "
            "spatial weight table, 10-day embargoed chronological splits, and training-only monthly bias correction "
            "with out-of-fold per-variable XGBoost error regressors, a Platt-calibrated XGBoost bust classifier, and "
            "exact TreeSHAP attributions, the promoted 17-year model achieves an honest issue-time ROC-AUC of "
            "<b>0.7615</b> (PR-AUC 0.6784, Brier 0.1898, BSS +0.2019) across 2,430,900 held-out 2017 events. It "
            "improves by +0.1043 ROC-AUC over the pre-audit deployment at issue time, surpasses an eight-rung "
            "meteorological baseline ladder (including EMOS and district-season climatology) as well as a dilated CNN "
            "challenger, flags major failures such as the August 2018 Kerala floods three days in advance, and serves "
            "calibrated probabilities and transparent feature attributions within a zero-cost 512 MB operational footprint."
        )
    )

    # ---- References ----
    story.append(H1("References"))
    refs = [
        "1. Potvin, C. K., et al. (2024). Explainable machine learning for predicting medium-range weather forecast busts. <i>Weather and Forecasting</i> (AMS).",
        "2. Rodwell, M. J., et al. (2013). Characteristics of occasional poor medium-range weather forecasts for Europe. <i>Quarterly Journal of the Royal Meteorological Society</i>, 139(677), 2025–2039.",
        "3. Guan, H., et al. (2022). GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological applications. <i>Monthly Weather Review</i>, 150(3), 647–672.",
        "4. Hersbach, H., et al. (2020). The ERA5 global reanalysis. <i>Quarterly Journal of the Royal Meteorological Society</i>, 146(730), 1999–2049.",
        "5. Gneiting, T., Raftery, A. E., Westveld, A. H., &amp; Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. <i>Monthly Weather Review</i>, 133(5), 1098–1118.",
        "6. Dimitriadis, T., Gneiting, T., &amp; Jordan, A. I. (2021). Stable reliability diagrams for probabilistic classifiers. <i>Proceedings of the National Academy of Sciences</i>, 118(8), e2016191118.",
        "7. Shanker, R., Sarkar, A., &amp; Mamgain, A. (2024). Probabilistic verification of medium-range ensemble rainfall forecasts over India (NCMRWF). <i>Quarterly Journal of the Royal Meteorological Society</i>, doi:10.1002/qj.4674.",
        "8. Rasp, S., &amp; Lerch, S. (2018). Neural networks for postprocessing ensemble weather forecasts. <i>Monthly Weather Review</i>, 146(11), 3885–3900.",
        "9. Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i> (pp. 785–794).",
        "10. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. <i>Nature Machine Intelligence</i>, 2(1), 56–67.",
        "11. Silva, M. E., et al. (2022). Explainable artificial intelligence for weather and climate prediction errors. <i>Environmental Data Science</i>, 1, e15.",
        "12. Dube, A., Ashrit, R., Ashish, A., &amp; Mitra, A. K. (2022). Verification of NCMRWF medium-range ensemble forecasts during the Indian summer monsoon. <i>Journal of Earth System Science</i>, 131, 98.",
    ]
    for r in refs:
        story.append(Paragraph(r, st["ref"]))

    doc = BaseDocTemplate(
        str(PDF_OUT),
        pagesize=A4,
        leftMargin=MARGIN_LR,
        rightMargin=MARGIN_LR,
        topMargin=MARGIN_TB,
        bottomMargin=MARGIN_TB,
        title="Methodology and Model Performance Report — Sanket",
        author="DN Harith, Dhanush Thota",
        subject="Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India",
    )
    frame = Frame(MARGIN_LR, MARGIN_TB, BODY_W, PAGE_H - 2 * MARGIN_TB, id="normal", topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=lambda c, d: draw_footer(c, d, fonts[0]))])
    doc.build(story)
    print(f"Built PDF: {PDF_OUT} ({PDF_OUT.stat().st_size:,} bytes)")


def main() -> None:
    print("Running reproducible archived-sample evaluation in isolated temporary store...")
    sample_eval = run_archived_sample_eval()
    # Clean up any generated docs/results.md just in case
    gen_res = REPO / "docs" / "results.md"
    if gen_res.exists():
        gen_res.unlink()
    build_pdf(sample_eval)


if __name__ == "__main__":
    main()
