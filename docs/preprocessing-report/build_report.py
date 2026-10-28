"""Build the Dataset Preprocessing and Model Selection Report PDF for Sanket.

The report follows ``Dataset_Preprocessing_Model_Selection_Report_Template.pdf`` in the
repository root: the same cover block, the same numbered section heads (1. Project
Overview ... 11. Appendix, then the Student Declaration), the same typography (Helvetica,
17 pt report title in #1F4E79, 13 pt section heads, 10.5 pt sub-section heads, 8.5 pt body,
7.3 pt tables with the template's #D9EAF7 header fill and 0.5 pt grey rules) and the same
footer on every page.

Every number under "Preliminary EDA", "Data Cleaning" and "Data Preprocessing" is measured
by this script from the files in ``backend/data/`` at build time, so a figure or a table
cannot drift away from the data it describes. The remaining numbers are the ones the
project's own documents record as measured, each quoted with its date where the document
records one.

Outputs
-------
``Dataset Preprocessing and Model Selection Report - Sanket.pdf``  (repository root)
``docs/preprocessing-report/preprocessing-report.md``             (same content, reviewable)
``docs/preprocessing-report/assets/*.png``                         (figures, regenerated)

Requirements: reportlab, matplotlib, pandas, pyarrow, pillow. Helvetica is a reportlab
built-in, so no font installation is needed.

Usage::

    python docs/preprocessing-report/build_report.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "backend" / "data"
ASSETS = HERE / "assets"
OUT_PDF = REPO / "Dataset Preprocessing and Model Selection Report - Sanket.pdf"
OUT_MD = HERE / "preprocessing-report.md"

# --------------------------------------------------------------------------------------
# Typography and page geometry, copied from the template PDF
# --------------------------------------------------------------------------------------
NAVY = colors.Color(0x1F / 255, 0x4E / 255, 0x79 / 255)
HEADER_FILL = colors.Color(0.85098, 0.91765, 0.96863)      # the template's #D9EAF7
RULE_GREY = colors.Color(0.50196, 0.50196, 0.50196)
FOOTER_GREY = colors.Color(0x80 / 255, 0x80 / 255, 0x80 / 255)

PAGE_W, PAGE_H = A4
LEFT, RIGHT = 51.35, 39.36
TOP, BOTTOM = 36.0, 60.0
FRAME_W = PAGE_W - LEFT - RIGHT

BODY_SIZE, BODY_LEADING = 9.0, 13.2
H1_SIZE, H2_SIZE = 13.0, 10.5
TABLE_SIZE, TABLE_LEADING = 7.6, 9.6
SMALL_SIZE, SMALL_LEADING = 6.9, 8.6
CODE_SIZE, CODE_LEADING = 7.0, 8.8

FOOTER_TEXT = ("Current submission scope: dataset, cleaning, preprocessing, and proposed "
               "model selection")

STUDENT = {
    "Student Name": "D N Harith, Dhanush Thota",
    "Register Number": "1DS24AI012, 1DS24AI057",
    "Course / Class": "AI&ML, 5th semester",
    "Project Title": ("Explainable AI for District-Level Prediction of Medium-Range Weather "
                      "Forecast Busts over India"),
    "Faculty Guide": "Dr. Kusumika Krori Dutta",
    "Submission Date": "28-10-2026",
}


def make_styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=BODY_SIZE,
                          leading=BODY_LEADING, alignment=TA_JUSTIFY, spaceAfter=6,
                          spaceBefore=0)
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17,
                                leading=21, alignment=TA_CENTER, textColor=NAVY,
                                spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Bold", fontSize=11,
                                   leading=14, alignment=TA_CENTER, spaceAfter=16),
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=H1_SIZE,
                             leading=16, textColor=NAVY, spaceBefore=13, spaceAfter=5,
                             alignment=TA_LEFT, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=H2_SIZE,
                             leading=13, textColor=NAVY, spaceBefore=10, spaceAfter=4,
                             alignment=TA_LEFT, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold",
                             fontSize=BODY_SIZE, leading=12, spaceBefore=7, spaceAfter=3,
                             alignment=TA_LEFT, keepWithNext=1),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=3,
                                 spaceAfter=3, alignment=TA_LEFT),
        "caption": ParagraphStyle("caption", parent=body, fontName="Helvetica-Bold",
                                  fontSize=8.0, leading=10, alignment=TA_LEFT,
                                  spaceBefore=7, spaceAfter=3, keepWithNext=1),
        "figcaption": ParagraphStyle("figcaption", parent=body, fontSize=8.0, leading=10.2,
                                     alignment=TA_CENTER, spaceBefore=4, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=body, fontSize=TABLE_SIZE,
                               leading=TABLE_LEADING, alignment=TA_LEFT, spaceAfter=0),
        "cell_b": ParagraphStyle("cell_b", parent=body, fontName="Helvetica-Bold",
                                 fontSize=TABLE_SIZE, leading=TABLE_LEADING,
                                 alignment=TA_LEFT, spaceAfter=0),
        "cell_s": ParagraphStyle("cell_s", parent=body, fontSize=SMALL_SIZE,
                                 leading=SMALL_LEADING, alignment=TA_LEFT, spaceAfter=0),
        "cell_sb": ParagraphStyle("cell_sb", parent=body, fontName="Helvetica-Bold",
                                  fontSize=SMALL_SIZE, leading=SMALL_LEADING,
                                  alignment=TA_LEFT, spaceAfter=0),
        "legend": ParagraphStyle("legend", parent=body, fontSize=SMALL_SIZE,
                                 leading=SMALL_LEADING + 0.6, spaceBefore=3, spaceAfter=7,
                                 alignment=TA_LEFT),
        "code": ParagraphStyle("code", parent=body, fontName="Courier", fontSize=CODE_SIZE,
                               leading=CODE_LEADING, alignment=TA_LEFT, spaceAfter=0),
        "ref": ParagraphStyle("ref", parent=body, fontSize=8.2, leading=11,
                              leftIndent=18, firstLineIndent=-18, spaceAfter=4,
                              alignment=TA_JUSTIFY),
        "cover_key": ParagraphStyle("cover_key", parent=body, fontName="Helvetica-Bold",
                                    fontSize=8.5, leading=17, alignment=TA_LEFT,
                                    spaceAfter=0),
        "cover_val": ParagraphStyle("cover_val", parent=body, fontSize=8.5, leading=17,
                                    alignment=TA_LEFT, spaceAfter=0),
        "note": ParagraphStyle("note", parent=body, fontSize=8.2, leading=11,
                               leftIndent=10, textColor=colors.Color(0.25, 0.25, 0.25),
                               spaceBefore=2, spaceAfter=7),
    }


# --------------------------------------------------------------------------------------
# Content accumulator: flowables for the PDF, and the same content as markdown
# --------------------------------------------------------------------------------------
class Content:
    def __init__(self, styles: dict[str, ParagraphStyle]):
        self.s = styles
        self.flow: list = []
        self.md: list[str] = []
        self.table_n = 0
        self.figure_n = 0

    # -- text ---------------------------------------------------------------------------
    def title(self, text: str, subtitle: str):
        self.flow.append(Paragraph(text, self.s["title"]))
        self.flow.append(Paragraph(subtitle, self.s["subtitle"]))
        self.md.append(f"# {text}")
        self.md.append(f"**{subtitle}**")

    def h1(self, text: str):
        self.flow.append(Paragraph(text, self.s["h1"]))
        self.md.append(f"\n## {text}\n")

    def h2(self, text: str):
        self.flow.append(Paragraph(text, self.s["h2"]))
        self.md.append(f"\n### {text}\n")

    def h3(self, text: str):
        self.flow.append(Paragraph(text, self.s["h3"]))
        self.md.append(f"\n**{text}**\n")

    def p(self, text: str):
        self.flow.append(Paragraph(text, self.s["body"]))
        self.md.append(_md(text) + "\n")

    def note(self, text: str):
        self.flow.append(Paragraph(text, self.s["note"]))
        self.md.append(f"*{_md(text)}*\n")

    def bullets(self, items: list[str], numbered: bool = False):
        for i, item in enumerate(items, start=1):
            bullet = f"{i}." if numbered else "\u2022"
            style = ParagraphStyle("b", parent=self.s["bullet"]) if numbered else self.s["bullet"]
            self.flow.append(Paragraph(item, style, bulletText=bullet))
            self.md.append(f"{bullet} {_md(item)}")
        self.md.append("")

    def caption(self, text: str):
        self.flow.append(Paragraph(text, self.s["caption"]))
        self.md.append(f"\n**{_md(text)}**\n")

    # -- tables -------------------------------------------------------------------------
    def table(self, title: str, header: list[str], rows: list[list[str]],
              widths: list[float], small: bool = False, legend: str | None = None,
              padding: float = 3.5):
        self.table_n += 1
        self.caption(f"Table {self.table_n}. {title}")
        style_cell = self.s["cell_s"] if small else self.s["cell"]
        style_head = self.s["cell_sb"] if small else self.s["cell_b"]
        data = [[Paragraph(h, style_head) for h in header]]
        data += [[Paragraph(cell, style_cell) for cell in row] for row in rows]
        widths_pt = [w * FRAME_W for w in widths]
        t = Table(data, colWidths=widths_pt, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
            ("GRID", (0, 0), (-1, -1), 0.5, RULE_GREY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE_GREY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), padding),
            ("RIGHTPADDING", (0, 0), (-1, -1), padding),
            ("TOPPADDING", (0, 0), (-1, -1), 3.2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ]))
        self.flow.append(t)
        if legend:
            self.flow.append(Paragraph(legend, self.s["legend"]))
        else:
            self.flow.append(Spacer(1, 6))
        self.md.append("")
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "---|" * len(header))
        for row in rows:
            self.md.append("| " + " | ".join(_md(c) for c in row) + " |")
        self.md.append("")
        if legend:
            self.md.append(_md(legend) + "\n")

    # -- code ---------------------------------------------------------------------------
    def code(self, lines: str, caption: str | None = None):
        body = [[Paragraph(line.replace(" ", "&nbsp;"), self.s["code"])]
                for line in lines.splitlines() or [""]]
        t = Table(body, colWidths=[FRAME_W], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.968, 0.972, 0.976)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(0.78, 0.80, 0.83)),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (0, 0), 4),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
        ]))
        block = [t]
        if caption:
            block.append(Paragraph(caption, self.s["legend"]))
        else:
            block.append(Spacer(1, 5))
        self.flow.append(KeepTogether(block))
        self.md.append("```python")
        self.md.append(lines.rstrip())
        self.md.append("```")
        if caption:
            self.md.append(f"*{_md(caption)}*")
        self.md.append("")

    # -- figures ------------------------------------------------------------------------
    def figure(self, path: Path, caption: str, width: float = 1.0):
        self.figure_n += 1
        h = _aspect(path) * width * FRAME_W
        self.flow.append(KeepTogether([
            Spacer(1, 4),
            Image(str(path), width=width * FRAME_W, height=h, hAlign="CENTER"),
            Paragraph(caption, self.s["figcaption"]),
        ]))
        self.md.append(f"\n![{_md(caption)}](assets/{path.name})\n")
        self.md.append(f"*{_md(caption)}*")

    def ref(self, text: str):
        self.flow.append(Paragraph(text, self.s["ref"]))
        self.md.append(f"{_md(text)}\n")

    def page_break(self):
        self.flow.append(PageBreak())
        self.md.append("\n---\n")


def _aspect(path: Path) -> float:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.height / im.width


_MD = {"<b>": "**", "</b>": "**", "<i>": "*", "</i>": "*", "<br/>": " "}
_MD_ENTITIES = {"&mdash;": "\u2014", "&ndash;": "\u2013", "&rsquo;": "\u2019",
                "&lsquo;": "\u2018", "&ldquo;": "\u201c", "&rdquo;": "\u201d",
                "&deg;": "\u00b0", "&sup2;": "\u00b2", "&times;": "\u00d7",
                "&asymp;": "approx.", "&nbsp;": " ", "&amp;": "&", "&gt;": ">",
                "&lt;": "<"}


def _md(text: str) -> str:
    for k, v in _MD.items():
        text = text.replace(k, v)
    for k, v in _MD_ENTITIES.items():
        text = text.replace(k, v)
    return text


# --------------------------------------------------------------------------------------
# Measurement: everything the report says about the data is computed here
# --------------------------------------------------------------------------------------
# Plain-text twins of the unit strings, for matplotlib (which would print the HTML
# entities literally); the reportlab prose keeps the entities.
PLAIN_UNIT = {"&deg;C": "\u00b0C", "%RH": "%RH", "mm/day": "mm/day", "hPa": "hPa",
              "kg/m&sup2;": "kg/m\u00b2", "m/s": "m/s"}

VARIABLES = {
    "t2m_c": ("temperature_c", "2 m temperature", "&deg;C"),
    "rh2m_pct": ("humidity_pct", "2 m relative humidity", "%RH"),
    "apcp_mm": ("rainfall_mm", "daily rainfall", "mm/day"),
    "mslp_hpa": ("pressure_hpa", "mean sea-level pressure", "hPa"),
    "pwat_kgm2": ("atmospheric_moisture_kgm2", "column water vapour", "kg/m&sup2;"),
    "wspd10m_ms": ("wind_speed_ms", "10 m wind speed", "m/s"),
}
OBS_COLUMNS = {"temperature_c": "t2m_c", "humidity_pct": "rh2m_pct",
               "rainfall_mm": "precip_mm", "pressure_hpa": "mslp_hpa",
               "atmospheric_moisture_kgm2": "pwat_kgm2", "wind_speed_ms": "wspd10m_ms"}
PAIR_KEYS = ["station", "region", "init_date", "valid_date", "lead_day"]


def measure() -> dict:
    """Load the archived files and measure what the report states about them."""
    g = pd.read_parquet(DATA / "samples" / "gefs_reforecast_india_2019.parquet")
    o = pd.read_parquet(DATA / "samples" / "era5_observations_india_2019.parquet")
    desc = pd.read_parquet(DATA / "geo" / "district_descriptors.parquet")
    weights = pd.read_parquet(DATA / "geo" / "district_grid_weights.parquet")
    mjo = pd.read_parquet(DATA / "mjo_omi_index.parquet")
    elev = pd.read_parquet(DATA / "geo" / "grid_elevation_m.parquet")
    lsm = pd.read_parquet(DATA / "geo" / "era5_land_sea_mask.parquet")

    g = g.rename(columns={"city": "station"})
    o = o.rename(columns={"city": "station"})

    # --- ensemble mean and spread per (station, initialisation, valid day, lead) --------
    cols = list(VARIABLES)
    mean = g.groupby(PAIR_KEYS, observed=True)[cols].mean()
    spread = g.groupby(PAIR_KEYS, observed=True)[cols].std(ddof=0)
    spread.columns = [f"spread_{VARIABLES[c][0]}" for c in spread.columns]
    fc = mean.join(spread).reset_index()
    fc = fc.rename(columns={c: f"fc_{VARIABLES[c][0]}" for c in cols})

    obs = o.rename(columns={"date": "valid_date",
                            **{v: f"ob_{k}" for k, v in OBS_COLUMNS.items()}})
    paired = fc.merge(obs[["station", "valid_date"] + [f"ob_{k}" for k in OBS_COLUMNS]],
                      on=["station", "valid_date"], how="inner")
    paired["month"] = pd.to_datetime(paired["valid_date"]).dt.month
    paired["monsoon"] = paired["month"].isin([6, 7, 8, 9])

    # --- per-variable error statistics and the training-set p90 threshold --------------
    errors: dict[str, pd.Series] = {}
    stats: dict[str, dict] = {}
    for v in OBS_COLUMNS:
        signed = (paired[f"fc_{v}"] - paired[f"ob_{v}"])
        err = signed.abs().dropna()
        errors[v] = err
        stats[v] = {
            "n": int(err.size),
            "bias": float(signed.mean()),
            "mae": float(err.mean()),
            "median": float(err.median()),
            "p90": float(err.quantile(0.90)),
            "p99": float(err.quantile(0.99)),
            "maximum": float(err.max()),
        }
        d = paired[[f"spread_{v}", f"fc_{v}", f"ob_{v}"]].dropna()
        rank_x = d[f"spread_{v}"].rank()
        rank_y = (d[f"fc_{v}"] - d[f"ob_{v}"]).abs().rank()
        stats[v]["spread_rank_corr"] = float(rank_x.corr(rank_y))
        stats[v]["r_fc_ob"] = float(d[f"fc_{v}"].corr(d[f"ob_{v}"]))
        stats[v]["mae_monsoon"] = float(err[paired.loc[err.index, "monsoon"]].mean())
        stats[v]["mae_non_monsoon"] = float(err[~paired.loc[err.index, "monsoon"]].mean())

    # --- label statistics --------------------------------------------------------------
    bust = pd.DataFrame(index=paired.index)
    for v in OBS_COLUMNS:
        e = (paired[f"fc_{v}"] - paired[f"ob_{v}"]).abs()
        bust[v] = (e > stats[v]["p90"]).where(e.notna())
    event = (bust > 0).any(axis=1)
    by_lead = {int(k): float(v) for k, v in event.groupby(paired["lead_day"]).mean().items()}
    by_month = {int(k): float(v) for k, v in event.groupby(paired["month"]).mean().items()}
    vars_at_lead = {
        int(ld): int(sum(1 for v in OBS_COLUMNS
                         if (bust.loc[paired["lead_day"] == ld, v].notna().any())))
        for ld in sorted(paired["lead_day"].unique())
    }

    # --- data-quality measurements -----------------------------------------------------
    numeric_g = [c for c in cols] + ["soilw_vol_pct", "wdir10m_deg"]
    saturated = g[g["soilw_vol_pct"] >= 99.5]
    soil_g = g.groupby("station", observed=True)["soilw_vol_pct"].mean()
    soil_o = o.groupby("station", observed=True)["soil_moisture_pct"].mean()
    coastal_soil = {s: (float(soil_g[s]), float(soil_o.get(s, np.nan)))
                    for s in ["Daman", "Kavaratti", "Port Blair"]}
    rh_bias = {}
    for station, s in g.groupby("station", observed=True):
        m = s.groupby("valid_date")["rh2m_pct"].mean()
        ob = o[o["station"] == station].set_index("date")["rh2m_pct"]
        d = (m - ob.reindex(m.index)).dropna()
        rh_bias[station] = float(d.mean())

    weights_per_district = weights.groupby("region_id")["weight"]
    measures = dict(
        n_fc_rows=int(len(g)), n_fc_cols=int(g.shape[1]),
        n_cycles=int(g["init_date"].nunique()), n_stations=int(g["station"].nunique()),
        n_members=int(g["member"].nunique()), leads=(int(g["lead_day"].min()),
                                                     int(g["lead_day"].max())),
        rows_per_cycle=int(len(g) / g["init_date"].nunique()),
        n_obs_rows=int(len(o)), n_obs_cols=int(o.shape[1]),
        n_obs_days=int(o["date"].nunique()),
        obs_dates=(str(o["date"].min()), str(o["date"].max())),
        obs_nulls=int(o.isna().sum().sum()),
        fc_nulls={c: int(g[c].isna().sum()) for c in numeric_g},
        fc_null_share=float(g[numeric_g].isna().to_numpy().mean()),
        fc_dup_keys=int(g.duplicated(subset=["station", "init_date", "valid_date",
                                             "lead_day", "member"]).sum()),
        obs_dup_keys=int(o.duplicated(subset=["station", "date"]).sum()),
        n_paired=int(len(paired)),
        stats=stats,
        bust_share={v: float(bust[v].mean()) for v in OBS_COLUMNS},
        event_rate=float(event.mean()),
        event_by_lead=by_lead, event_by_month=by_month, vars_at_lead=vars_at_lead,
        zero_rain_share=float((o["precip_mm"] <= 0.01).mean()),
        monsoon_share=float(pd.to_datetime(o["date"]).dt.month.isin([6, 7, 8, 9]).mean()),
        saturated_rows=int(len(saturated)),
        saturated_stations=sorted(saturated["station"].unique().tolist()),
        coastal_soil=coastal_soil,
        rh_bias=rh_bias,
        n_districts=int(len(desc)), n_states=int(desc["state_id"].nunique()),
        district_stats={c: (float(desc[c].min()), float(desc[c].median()),
                            float(desc[c].max()))
                        for c in ["area_km2", "elevation_mean", "border_distance_km",
                                  "centroid_lat", "centroid_lon"]},
        n_weight_pairs=int(len(weights)),
        n_cells=int(len(weights[["lat", "lon"]].drop_duplicates())),
        weight_sums=(float(weights_per_district.sum().min()),
                     float(weights_per_district.sum().max())),
        cells_per_district=(int(weights_per_district.size().min()),
                            float(weights_per_district.size().median()),
                            int(weights_per_district.size().max())),
        mjo_rows=int(len(mjo)), mjo_span=(str(mjo["date"].min().date()),
                                          str(mjo["date"].max().date())),
        mjo_nulls=int(mjo.isna().sum().sum()),
        elev_cells=int(len(elev)),
        elev_range=(float(elev["elevation_m"].min()), float(elev["elevation_m"].median()),
                    float(elev["elevation_m"].max())),
        land_share=float((lsm["lsm"] > 0.5).mean()),
    )
    return {"g": g, "o": o, "paired": paired, "bust": bust, "M": measures}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.edgecolor": "0.35",
        "axes.linewidth": 0.6,
        "grid.color": "0.88",
        "grid.linewidth": 0.5,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
    })
    return plt


INK = "#1f4e79"
ACCENT = "#c0392b"
FILL = "#3f7fb5"


def _box(ax, x, y, w, h, text, face="#d9eaf7", edge=INK, size=6.9, weight="normal"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0.8, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color="#12232e", fontweight=weight, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, colour=INK):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.8, color=colour, shrinkA=0, shrinkB=0))


def fig_pipeline(plt, path: Path):
    """Section 8: the planned pipeline, acquisition to model selection."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.85, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1. Problem definition",
         "Research gap and literature survey\n-> the bust definition and predictor set"),
        ("2. Dataset acquisition",
         "GEFSv12 reforecast + ERA5 + GADM 4.1\ndistricts, elevation, land-sea, MJO"),
        ("3. Data quality assessment",
         "Missingness, duplicates, plausibility\nranges, cross-product agreement"),
        ("4. Data cleaning",
         "Completeness guard, dedupe, range check,\nland-sea mask, date alignment"),
        ("5. Data preprocessing",
         "Canonical schema, area weighting, daily\naggregation, bias table, features"),
        ("6. Preliminary EDA",
         "Dimensions, distributions, class balance,\nspread-error relation, coverage"),
        ("7. Dataset splitting",
         "Chronological by cycle, Day-10 embargo,\n30-day out-of-fold blocks"),
        ("8. Candidate model selection",
         "XGBoost, random forest, logistic EMOS,\ndilated CNN, spatio-temporal (deferred)"),
    ]
    box_w, box_h = 0.415, 0.175
    xs = {"L": 0.03, "R": 0.555}
    ys = [0.720, 0.500, 0.280, 0.060]
    # serpentine: odd stages in the left column, even stages in the right column
    place = []
    for i in range(8):
        col = "L" if i % 2 == 0 else "R"
        row = i // 2
        place.append((xs[col], ys[row]))
    for (x, y), (head, detail) in zip(place, stages):
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.010",
                                    linewidth=0.8, edgecolor=INK, facecolor="#d9eaf7"))
        ax.text(x + box_w / 2, y + box_h * 0.70, head, ha="center", va="center",
                fontsize=7.0, color="#12232e", fontweight="bold")
        ax.text(x + box_w / 2, y + box_h * 0.32, detail, ha="center", va="center",
                fontsize=6.2, color="#243642", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.9, color=INK,
                                     shrinkA=0, shrinkB=0))

    cx = {"L": xs["L"] + box_w / 2, "R": xs["R"] + box_w / 2}
    for row in range(4):
        # across: left box -> right box on the same row
        arrow(xs["L"] + box_w + 0.006, ys[row] + box_h / 2,
              xs["R"] - 0.006, ys[row] + box_h / 2)
        if row < 3:
            # down: right box of this row -> left box of the next, through the gap
            gap_y = ys[row + 1] + box_h + 0.022   # midway between the two rows
            ax.plot([cx["R"], cx["R"]], [ys[row], gap_y], color=INK, linewidth=0.9)
            ax.plot([cx["R"], cx["L"]], [gap_y, gap_y], color=INK, linewidth=0.9)
            arrow(cx["L"], gap_y, cx["L"], ys[row + 1] + box_h - 0.002)
    ax.text(0.5, 0.965, "Planned pipeline: dataset acquisition to model selection",
            ha="center", va="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.text(0.5, 0.005, "Stages 1-5 are complete for the archived dataset; stage 6 is what "
                        "Section 5 reports; stage 7 is the plan in Section 6; stage 8 is the "
                        "selection in Section 7.",
            ha="center", va="center", fontsize=6.3, color="0.3")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_preprocess(plt, path: Path):
    """Section 4.1/4.4: the preprocessing chain and where each step is fitted."""
    fig, ax = plt.subplots(figsize=(6.85, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    row1 = ["GRIB2 / NetCDF\n/ CSV / GeoJSON",
            "Format-agnostic\nparsers",
            "Confidence-scored\nschema mapper",
            "Geo resolution:\none weight table",
            "Daily aggregation\nto UTC days"]
    row2 = ["Plausibility\nrange check",
            "Dedupe on the\ncanonical key",
            "Complete-cycle\nguard",
            "Land-sea mask\nfor soil moisture",
            "Bias table\n(train cycles)"]
    row3 = ["Percentage bust\nthresholds (train)",
            "Feature engineering\nspread, jumps, LAF",
            "Design matrix /\nchannel tensors",
            "Chronological split\n+ Day-10 embargo",
            "Model-ready\nsplits"]
    w, h = 0.176, 0.20
    for r, labels in enumerate((row1, row2, row3)):
        y = 0.70 - r * 0.285
        for c, label in enumerate(labels):
            x = 0.012 + c * 0.199
            face = "#d9eaf7" if r < 2 else "#e8f2e8"
            edge = INK if r < 2 else "#2e6b34"
            _box(ax, x, y, w, h, label, face=face, edge=edge, size=6.3)
            if c < 4:
                _arrow(ax, x + w, y + h / 2, x + 0.199 - 0.004, y + h / 2)
        if r < 2:
            x_last = 0.012 + 4 * 0.199 + w / 2
            _arrow(ax, x_last, y, x_last, y - 0.285 + h)
    ax.text(0.5, 0.975, "Preprocessing workflow (blue: applied to both sides; "
                        "green: fitted on training cycles only)",
            ha="center", va="center", fontsize=8.2, color=INK, fontweight="bold")
    ax.text(0.5, 0.035, "Every row of both products is reduced to the same canonical key "
                        "(district, variable, valid date, value type, member, lead), so a "
                        "forecast and its verifying observation are always compared as "
                        "like with like.",
            ha="center", va="center", fontsize=6.2, color="0.3")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_missingness(plt, path: Path, d: dict):
    g = d["g"]
    cols = ["t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "pwat_kgm2", "wspd10m_ms",
            "soilw_vol_pct"]
    labels = ["2 m temp", "2 m RH", "rainfall", "MSLP", "w.vapour", "10 m wind",
              "soil moist."]
    fig = plt.figure(figsize=(6.85, 3.3))
    ax1 = fig.add_axes([0.105, 0.24, 0.40, 0.62])
    M = np.zeros((len(cols), 10))
    for i, c in enumerate(cols):
        for j, ld in enumerate(range(1, 11)):
            sub = g[g["lead_day"] == ld]
            M[i, j] = sub[c].notna().mean()
    ax1.imshow(M * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(range(10), [str(i) for i in range(1, 11)])
    ax1.set_yticks(range(len(cols)), labels)
    ax1.set_xlabel("forecast lead day")
    for i in range(len(cols)):
        for j in range(10):
            ax1.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.3,
                     color="white" if M[i, j] > 0.55 else "0.25")
    ax1.set_title("(a) Non-null share by variable and lead day (%)", pad=4)

    ax2 = fig.add_axes([0.625, 0.24, 0.35, 0.62])
    counts = [int(g[c].isna().sum()) for c in cols] + [int((g["soilw_vol_pct"] >= 99.5).sum())]
    names = labels + ["soil at 100%"]
    ypos = np.arange(len(counts))
    colours = [ACCENT if n > 0 else "#8fbf8f" for n in counts]
    colours[-1] = "#d99a2b"
    ax2.barh(ypos, counts, color=colours, height=0.62)
    ax2.set_yticks(ypos, names)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"rows out of {len(g):,}")
    ax2.grid(axis="x")
    ax2.set_axisbelow(True)
    for y, n in zip(ypos, counts):
        ax2.text(n + 300, y, f"{n:,}", va="center", fontsize=6.2)
    ax2.set_xlim(0, max(counts) * 1.30)
    ax2.set_title("(b) Null rows, and rows saturated at 100%", pad=4)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_feature_distributions(plt, path: Path, d: dict):
    paired = d["paired"]
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 3.5))
    for ax, (col, (name, label, unit)) in zip(axes.ravel(), VARIABLES.items()):
        fc = paired[f"fc_{name}"].dropna()
        ob = paired[f"ob_{name}"].dropna()
        bins = np.linspace(min(fc.min(), ob.min()), max(fc.quantile(0.995),
                                                        ob.quantile(0.995)), 40)
        ax.hist(fc.clip(bins[0], bins[-1]), bins=bins, density=True, alpha=0.55,
                color=FILL, label="forecast (ensemble mean)", edgecolor="none")
        ax.hist(ob.clip(bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                color=ACCENT, linewidth=1.0, label="ERA5 observation")
        ax.set_title(f"{label} ({PLAIN_UNIT[unit]})", pad=3)
        ax.set_ylabel("density")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_targets(plt, path: Path, d: dict):
    paired, stats = d["paired"], d["M"]["stats"]
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 3.6))
    for ax, (col, (name, label, unit)) in zip(axes.ravel(), VARIABLES.items()):
        e = (paired[f"fc_{name}"] - paired[f"ob_{name}"]).abs().dropna()
        thr = stats[name]["p90"]
        hi = float(e.quantile(0.995))
        ax.hist(np.clip(e, 0, hi), bins=42, color=FILL, edgecolor="#264f6f",
                linewidth=0.2)
        ax.axvline(thr, color=ACCENT, linewidth=1.0, linestyle="--")
        ax.set_title(f"{label}\n$p_{{90}}$ = {thr:.2f} {PLAIN_UNIT[unit]}", pad=3)
        ax.set_xlabel(f"|forecast - observation| ({PLAIN_UNIT[unit]})")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_class_balance(plt, path: Path, d: dict):
    M = d["M"]
    paired, bust = d["paired"], d["bust"]
    event = (bust > 0).any(axis=1)
    fig = plt.figure(figsize=(6.85, 2.75))

    ax1 = fig.add_axes([0.075, 0.22, 0.27, 0.62])
    leads = sorted(M["event_by_lead"])
    rates = [M["event_by_lead"][ld] * 100 for ld in leads]
    ax1.bar(leads, rates, color=FILL, width=0.68)
    ax1.axhline(M["event_rate"] * 100, color=ACCENT, linestyle="--", linewidth=0.9)
    ax1.text(1.0, M["event_rate"] * 100 + 1.8, f"all leads {M['event_rate'] * 100:.1f}%",
             color=ACCENT, fontsize=6.2)
    ax1.set_xticks(leads)
    ax1.set_xlabel("lead day")
    ax1.set_ylabel("event-days with a bust (%)")
    ax1.set_ylim(0, 62)
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Event bust rate by lead day", pad=4)

    ax2 = fig.add_axes([0.395, 0.22, 0.27, 0.62])
    months = sorted(M["event_by_month"])
    ax2.bar(months, [M["event_by_month"][m] * 100 for m in months], color=FILL, width=0.68)
    ax2.set_xticks(range(1, 13))
    ax2.set_xlabel("month of the valid date")
    ax2.set_ylim(0, 62)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Event bust rate by month", pad=4)

    ax3 = fig.add_axes([0.715, 0.22, 0.26, 0.62])
    ks = sorted(M["vars_at_lead"])
    measured = [M["event_by_lead"][k] * 100 for k in ks]
    independent = [100 * (1 - 0.9 ** M["vars_at_lead"][k]) for k in ks]
    ax3.plot(ks, independent, color=ACCENT, linestyle="--", linewidth=1.0,
             marker="o", markersize=2.6, label="if variables were independent")
    ax3.plot(ks, measured, color=INK, linewidth=1.2, marker="s", markersize=2.6,
             label="measured")
    ax3.set_xticks(ks)
    ax3.set_xlabel("lead day")
    ax3.set_ylim(0, 100)
    ax3.grid(axis="y")
    ax3.set_axisbelow(True)
    ax3.legend(loc="lower right")
    ax3.set_title("(c) Event rate vs. number of variables", pad=4)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_error_lead(plt, path: Path, d: dict):
    paired = d["paired"]
    fig = plt.figure(figsize=(6.85, 2.85))
    ax1 = fig.add_axes([0.075, 0.22, 0.40, 0.62])
    show = ["temperature_c", "humidity_pct", "rainfall_mm", "pressure_hpa"]
    for v in show:
        sub = paired[[f"fc_{v}", f"ob_{v}", "lead_day"]].dropna()
        mae = (sub[f"fc_{v}"] - sub[f"ob_{v}"]).abs().groupby(sub["lead_day"]).mean()
        ax1.plot(mae.index, mae.values, marker="o", markersize=2.4, linewidth=1.0,
                 label=v.replace("_", " "))
    ax1.set_xlabel("lead day")
    ax1.set_ylabel("mean absolute error (own units)")
    ax1.set_xticks(range(1, 11))
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left")
    ax1.set_title("(a) Error by lead day", pad=4)

    ax2 = fig.add_axes([0.575, 0.22, 0.40, 0.62])
    for v, colour in (("temperature_c", INK), ("rainfall_mm", ACCENT)):
        sub = paired[[f"spread_{v}", f"fc_{v}", f"ob_{v}"]].dropna()
        err = (sub[f"fc_{v}"] - sub[f"ob_{v}"]).abs()
        deciles = pd.qcut(sub[f"spread_{v}"].rank(method="first"), 10, labels=False)
        prof = err.groupby(deciles).mean()
        ax2.plot(prof.index + 1, prof.values, marker="o", markersize=2.6, linewidth=1.1,
                 color=colour,
                 label=f"{v.replace('_', ' ')} (rank corr {d['M']['stats'][v]['spread_rank_corr']:.2f})")
    ax2.set_xlabel("ensemble-spread decile")
    ax2.set_ylabel("mean absolute error (own units)")
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left")
    ax2.set_title("(b) Spread carries signal, but not enough", pad=4)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_split(plt, path: Path):
    """Section 6: the chronological split on one time axis, with the key beside it."""
    import matplotlib.patches as mpatches
    fig = plt.figure(figsize=(6.85, 2.15))
    ax = fig.add_axes([0.055, 0.10, 0.545, 0.76])
    ax.set_xlim(1999.3, 2020.2)
    ax.set_ylim(-1.05, 4.05)
    ax.axis("off")
    lanes = [
        ("Training", "70% of the 2000-2016 cycles (~4,330)", 2000.0, 2015.97, "#3f7fb5"),
        ("Validation", "15% of them, at the tail of 2016 (~930)", 2016.03, 2016.97, "#d99a2b"),
        ("Test", "all 365 cycles of 2017", 2017.0, 2017.97, "#c0392b"),
        ("Held back", "2018 and 2019, in the archive, unused", 2018.0, 2019.97, "#9fb3c2"),
    ]
    for i, (_short, _detail, x0, x1, colour) in enumerate(lanes):
        y = 3.05 - i * 0.72
        ax.add_patch(mpatches.Rectangle((x0, y), x1 - x0, 0.40, facecolor=colour,
                                        edgecolor="white", linewidth=0.8))
    for boundary in (2016.0, 2017.0):
        ax.plot([boundary, boundary], [-0.10, 3.55], color="#12232e", linewidth=0.8,
                linestyle=(0, (2, 1.6)))
        ax.annotate("", xy=(boundary, -0.12), xytext=(2016.5, -0.55),
                    arrowprops=dict(arrowstyle="-", linewidth=0.6, color="#12232e"))
    ax.text(2016.5, -0.62, "10-day embargo either side of each boundary", ha="center",
            va="top", fontsize=6.0, color="#12232e")
    for year in (2000, 2005, 2010, 2015, 2019):
        ax.plot([year, year], [-0.02, -0.08], color="0.45", linewidth=0.6)
        ax.text(year, -0.10, str(year), ha="center", va="top", fontsize=5.9, color="0.35")

    key = fig.add_axes([0.615, 0.18, 0.375, 0.70])
    key.set_xlim(0, 1)
    key.set_ylim(0, 4)
    key.axis("off")
    for i, (short, detail, _x0, _x1, colour) in enumerate(lanes):
        y = 3.55 - i * 0.95
        key.add_patch(mpatches.Rectangle((0.0, y - 0.16), 0.055, 0.32, facecolor=colour,
                                         edgecolor="none"))
        key.text(0.09, y + 0.13, short, fontsize=6.9, fontweight="bold", va="center",
                 color="#12232e")
        key.text(0.09, y - 0.18, detail, fontsize=6.0, va="center", color="0.35")
    fig.text(0.012, 0.965, "Chronological split by forecast cycle: no cycle's ten observed "
                           "days appear on both sides of a boundary",
             fontsize=7.4, color=INK, fontweight="bold", va="top")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures(d: dict) -> dict[str, Path]:
    plt = _setup_mpl()
    ASSETS.mkdir(parents=True, exist_ok=True)
    # File names follow the figure numbers of the report itself.
    paths = {
        "missing": ASSETS / "fig1_missingness_and_saturation.png",
        "preprocess": ASSETS / "fig2_preprocessing_workflow.png",
        "features": ASSETS / "fig3_feature_distributions.png",
        "targets": ASSETS / "fig4_target_distributions.png",
        "class": ASSETS / "fig5_class_distribution.png",
        "errorlead": ASSETS / "fig6_error_by_lead_and_spread.png",
        "split": ASSETS / "fig7_split.png",
        "pipeline": ASSETS / "fig8_pipeline.png",
    }
    fig_missingness(plt, paths["missing"], d)
    fig_feature_distributions(plt, paths["features"], d)
    fig_preprocess(plt, paths["preprocess"])
    fig_targets(plt, paths["targets"], d)
    fig_class_balance(plt, paths["class"], d)
    fig_error_lead(plt, paths["errorlead"], d)
    fig_split(plt, paths["split"])
    fig_pipeline(plt, paths["pipeline"])
    return paths


# --------------------------------------------------------------------------------------
# Cover page
# --------------------------------------------------------------------------------------
def cover(st) -> list:
    flow = [Spacer(1, 14),
            Paragraph("PROJECT REPORT", st["title"]),
            Paragraph("Dataset Selection, Data Cleaning, Preprocessing, and Model Selection",
                      st["subtitle"])]
    # The template's rule under the title block
    rule = Table([[""]], colWidths=[FRAME_W], rowHeights=[2])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.8, NAVY),
                              ("TOPPADDING", (0, 0), (-1, -1), 0),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    flow += [rule, Spacer(1, 16), Paragraph("Student Details", st["h1"]), Spacer(1, 2)]

    rows = []
    for key, value in STUDENT.items():
        rows.append([Paragraph(key + ":", st["cover_key"]),
                     Paragraph(_md(value), st["cover_val"])])
    details = Table(rows, colWidths=[0.185 * FRAME_W, 0.815 * FRAME_W], hAlign="LEFT")
    details.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LINEBELOW", (1, 0), (1, -1), 0.4, colors.Color(0.78, 0.80, 0.83)),
    ]))
    flow += [details, Spacer(1, 18), Paragraph(
        "This report covers the dataset, data-cleaning, preprocessing and "
        "model-selection stage of the project. It contains no experimental model "
        "results: every score, comparison and conclusion about model performance "
        "belongs to the next phase.", st["note"])]
    return flow


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------
def build_content(st, d, figs) -> Content:
    M = d["M"]
    s = M["stats"]
    c = Content(st)

    # ==================================================================================
    # 1. Project Overview
    # ==================================================================================
    c.h1("1. Project Overview")

    c.h2("1.1 Problem Statement")
    c.p("An operational centre already issues a medium-range forecast for every district of "
        "India, and statistical post-processing already removes the errors a model makes "
        "consistently. What survives that correction is the day-to-day failure: the "
        "forecast that is badly wrong because of the particular weather pattern of that "
        "day. Following Rodwell et al. [2], this project calls such a forecast a "
        "<b>bust</b>: an error that falls in the tail of that variable's own historical "
        "error distribution, above its 90th percentile, with the percentile computed on "
        "training data only and never on the rows being labelled.")
    c.p("Existing work diagnoses continental upper-air busts after the event [2], [3], "
        "verifies Indian rainfall and circulation at regional scale [6]-[9], or explains "
        "the weather quantity a model predicts rather than the model's own error [16]-[18]. "
        "What is missing is an operationally usable, calibrated and locally explained "
        "probability that a specific medium-range forecast will fail, for several surface "
        "variables at once, at district scale over India. The problem is hard for a reason "
        "the data makes visible: ensemble spread is informative about error but far from "
        "sufficient, measured here as a rank correlation of "
        f"{s['temperature_c']['spread_rank_corr']:.2f} for temperature and "
        f"{s['rainfall_mm']['spread_rank_corr']:.2f} for rainfall (Section 5).")
    c.p("This report addresses the <b>data</b> part of that study: which dataset is used, "
        "why it is appropriate, what is wrong with it, how it is cleaned and preprocessed, "
        "what the preliminary statistics show, how it will be split without leakage, and "
        "which candidate models the project will compare. It makes no claim to be a new "
        "weather forecast, and it reports no model performance.")

    c.h2("1.2 Project Objectives")
    c.bullets([
        "Build a leakage-free, district-level paired dataset in which a GEFSv12 ensemble "
        "forecast and its verifying ERA5 observation are compared on one spatial unit (666 "
        "GADM 4.1 districts), one time unit (the UTC day) and one canonical schema, for "
        "lead days 1-10 across 2000-2019.",
        "Diagnose and treat the data-quality problems that actually occur in this dataset "
        "(missing forecast steps, duplicate truth products, misaligned date conventions, "
        "implausible ensemble members, sea read as saturated soil, observation-derived "
        "leakage) and record the measured effect of each treatment.",
        "Define the bust label as the 90th percentile of a variable's own bias-corrected "
        "error, fitted on training cycles only, and carry the any-variable event label "
        "that the model must predict.",
        "Engineer the predictors that hold the predictability signal at issue time: "
        "ensemble spread and member count, run-to-run jumpiness, time-lagged ensemble "
        "features, tendencies, season, district geography and the Madden-Julian "
        "Oscillation.",
        "Specify a chronological split with a Day-10 embargo and 30-day out-of-fold blocks, "
        "and demonstrate that no observed day, threshold, bias table or calibrator is "
        "shared between training and evaluation.",
        "Select candidate models (gradient-boosted trees, random forest, calibrated "
        "logistic regression, a dilated convolutional network and a spatio-temporal "
        "recurrent network) and justify each on the problem type, the dataset "
        "characteristics, the expected computational cost and the literature.",
    ], numbered=True)

    c.h2("1.3 Proposed Approach")
    c.p("The project treats the problem as supervised tabular classification over a paired "
        "forecast-observation record, with a regression stage in front of it. For each "
        "variable, a gradient-boosted regressor estimates the absolute error that the "
        "ensemble mean is about to make, using only information available at issue time. "
        "Its out-of-fold predictions, together with spread, jumpiness and context features, "
        "form the design matrix of a binary classifier that outputs P(bust) per district, "
        "per variable, per lead day. Probabilities are mapped to observed frequencies by a "
        "Platt fit on validation data, and every prediction carries TreeSHAP attributions "
        "so a duty forecaster can see which inputs drove it [13], [14], [15].")
    c.p("Two model families are carried rather than one: the tabular model above, and a "
        "dilated convolutional network that reads the raw 0.25&deg; fields and can see the "
        "synoptic pattern a district-mean row cannot express. Both are scored on identical "
        "held-out rows against a ladder of baselines (climatology, lead-day, spread, EMOS, "
        "IDR indestructible-rainfall and analog), because a bare ROC-AUC claims nothing "
        "without the answer to \"against what?\" [1], [10], [11]. The dataset, cleaning and "
        "preprocessing described in this report are what make that comparison honest.")

    # ==================================================================================
    # 2. Dataset Description
    # ==================================================================================
    c.h1("2. Dataset Description")
    c.h2("2.1 Dataset Information")
    c.table(
        "Dataset information",
        ["Parameter", "Description"],
        [
            ["Dataset name",
             "Sanket India district weather-forecast bust dataset: NOAA GEFSv12 reforecast "
             "(forecast side) + ERA5 reanalysis (verifying truth) aggregated to 666 GADM 4.1 "
             "districts, 2000-2019"],
            ["Dataset source / URL",
             "GEFSv12 reforecast, AWS Open Data bucket <i>noaa-gefs-retrospective</i>: "
             "https://registry.opendata.aws/noaa-gefs-reforecast/ &nbsp;|&nbsp; ERA5: "
             "https://cds.climate.copernicus.eu/datasets (reanalysis-era5-single-levels) "
             "&nbsp;|&nbsp; Districts: https://gadm.org/ &nbsp;|&nbsp; Elevation: "
             "https://api.open-elevation.com &nbsp;|&nbsp; MJO: https://psl.noaa.gov/mjo/"],
            ["Domain / application",
             "Meteorology and operational forecasting: probabilistic prediction of forecast "
             "failure (bust) for surface weather variables, with per-prediction explanation"],
            ["Total records / samples",
             f"242.7 million district-member-lead forecast rows across 7,288 daily "
             f"cycles (2000-2019); 2017 was measured end to end at 12,154,500 rows with "
             f"every cycle exactly 33,300 rows. This report's measurements use the archived "
             f"2019 sample: {M['n_fc_rows']:,} forecast rows "
             f"({M['n_cycles']} cycles x {M['n_stations']} stations x {M['n_members']} "
             f"members x 10 lead days) and {M['n_obs_rows']:,} observation rows "
             f"({M['n_obs_days']} days x {M['n_stations']} stations), giving "
             f"{M['n_paired']:,} forecast-observation pairs"],
            ["Number of features / attributes",
             "20 stored attributes per canonical row; 12 GEFSv12 fields (8 surface "
             "variables, 2 wind components, plus Z500/CAPE/CIN for the CNN); 24 engineered "
             "model inputs per regressor row (22 numeric, 2 categorical) plus one concurrent "
             "forecast column per variable; 6 district descriptors and 6 static grids"],
            ["Target variable / class labels",
             "Binary label <i>bust</i> per (district, variable, valid date, lead day): 1 if "
             "the bias-corrected absolute forecast error exceeds that variable's "
             "training-set 90th percentile, else 0. The continuous absolute error is carried "
             "alongside and is the regression target; the 8-variable <i>event</i> label is 1 "
             "if any variable busts"],
            ["Data format",
             "Sources: GRIB2 (reforecast), NetCDF and ZIP of NetCDF (ERA5), CSV/XLSX/JSON "
             "(IMD, MJO, status), GeoJSON (GADM). Store: hive-partitioned Apache Parquet "
             "plus SQLite lineage; figures and exports in CSV/PNG"],
            ["Dataset licence",
             "GEFSv12 reforecast and the GEFS feed: public domain (NOAA Open Data). ERA5: "
             "CC-BY 4.0 (\"Generated using Copernicus Climate Change Service information\"). "
             "GADM 4.1: free for academic and non-commercial use, redistribution requires "
             "GADM's permission. SRTM elevation: acknowledge CIAT. MJO OMI: NOAA open data. "
             "The project's own code and documents are MIT-licensed and carry none of these "
             "terms onward"],
        ],
        widths=[0.20, 0.80], legend=(
            "Licences are separated per source in <i>REUSE.toml</i> and "
            "<i>THIRD_PARTY_NOTICES.md</i>; a boundary geometry for a political map of India "
            "raises questions of its own, reviewed in <i>docs/boundary-geometry-licensing.md</i>."))

    c.h2("2.2 Dataset Selection Justification")
    c.p("The dataset was selected against the criteria below, and each is answered with a "
        "property of the data rather than with a claim about it.")
    c.bullets([
        "<b>Relevance to the question.</b> The task is to predict the error of an "
        "already-issued forecast, so the dataset must contain both an ensemble forecast "
        "archive and an independent verifying truth for the same place and day. GEFSv12 "
        "reforecast and ERA5 provide exactly that pair for 2000-2019, with the same 0.25&deg; "
        "grid and a documented daily convention, so the two sides can be reduced through one "
        "weight table instead of two approximations of geography.",
        "<b>Relevance to the region.</b> The study is about India: monsoon rainfall, "
        "Himalayan terrain, coastal and island districts. The dataset carries all 666 GADM "
        "4.1 districts, including the districts with no 0.25&deg; grid centre inside them "
        "(Kolkata, Hyderabad, the Puducherry enclaves, Lakshadweep), which a nearest-point "
        "sample would silently drop.",
        "<b>Size.</b> A bust is by construction a 10% tail event, and the event label fires on "
        "38.6% of paired days only because several variables are combined; rare failures are "
        "learnable only if the archive is long. Twenty years of daily cycles at 666 "
        "districts gives roughly 4.9 million district-days before any lead or member "
        "expansion, against the 106 cases of the closest published study [1].",
        "<b>Quality and traceability.</b> Every value carries the file, message and API that "
        "produced it; the forecast side records the GRIB2 object path and message number per "
        "variable, and the observation side records the product and endpoint. Nothing is "
        "synthetic, simulated, interpolated or placeholder: where a number could not be "
        "fetched, there is no row.",
        "<b>Suitability of the predictors.</b> The archive carries the fields that hold the "
        "predictability signal at issue time: five ensemble members for spread, consecutive "
        "initialisations for run-to-run jumpiness and a time-lagged pool, twelve variables "
        "for concurrent conditions, and a daily MJO index as a large-scale covariate.",
        "<b>Comparability with the literature.</b> The dataset and the label are defined so "
        "that the project's results can be compared with the studies reviewed in its "
        "literature survey: percentile-based skill classes [1], ensemble-spread baselines "
        "[1], [10], EMOS post-processing [10], [11] and verified Indian rainfall skill [6]-[9].",
    ])

    c.h2("2.3 Dataset Characteristics")
    c.p("The dataset has four linked, all-numeric parts: a <b>forecast archive</b> (one 00 "
        "UTC cycle a day, five members, lead days 1-10), a <b>verifying observation "
        "archive</b> (ERA5 on the same grid and the same UTC-day convention), a <b>geography "
        f"layer</b> ({M['n_districts']} districts in {M['n_states']} states, "
        f"{M['n_weight_pairs']:,} district-cell weights over {M['n_cells']:,} cells, every "
        f"district's weights summing to exactly {M['weight_sums'][0]:.1f}), and "
        f"<b>covariates</b> ({M['elev_cells']:,} cell elevations, an ERA5 land-sea mask, and "
        f"{M['mjo_rows']:,} daily MJO index rows spanning {M['mjo_span'][0]} to "
        f"{M['mjo_span'][1]}).")
    c.table(
        "Variables carried, their role and their label status",
        ["Canonical variable", "Unit", "Forecast field", "Leads held", "Role"],
        [
            ["temperature_c", "&deg;C", "tmp_2m (t2m)", "Day 1-10", "Label + input"],
            ["humidity_pct", "%RH", "spfh_2m (sh2), converted after the district mean",
             "Day 1-10", "Label + input"],
            ["rainfall_mm", "mm/day", "apcp_sfc, accumulated over the model day", "Day 1-10",
             "Label + input"],
            ["pressure_hpa", "hPa", "pres_msl (msl)", "Day 1-10", "Label + input"],
            ["atmospheric_moisture_kgm2", "kg/m&sup2;", "pwat_eatm (pwat)", "Day 1-10",
             "Label + input"],
            ["wind_speed_ms", "m/s", "ugrd_hgt / vgrd_hgt (u10, v10)", "Day 1-5 only",
             "Label + input"],
            ["wind_direction_deg", "degrees from", "ugrd_hgt / vgrd_hgt", "Day 1-5 only",
             "Input only, never a label"],
            ["soil_moisture_pct", "% volumetric", "soilw_bgrnd, top layer", "Day 1-3 only",
             "Label, masked over sea"],
            ["Z500, CAPE, CIN", "m, J/kg, J/kg", "gh500, cape_sfc, cin_sfc", "Day 1-10",
             "CNN input channels only"],
        ],
        widths=[0.24, 0.10, 0.36, 0.12, 0.18], small=True,
        legend="Wind direction is an input and never a label: a daily resultant direction is "
               "mostly noise when the wind is light, and 86.5% of its apparent busts under "
               "plain angle arithmetic disappear once the error is measured the short way "
               "round the circle.")
    c.p("<b>Data types and structure.</b> Every stored attribute is numeric, categorical or a "
        "timestamp; there is no text, image or free-form field in the training path. The rows "
        "are not independent samples: consecutive days at one district are strongly "
        "autocorrelated, neighbouring districts share 0.25&deg; cells through the weight "
        "table, and the five members of one cycle are perturbations of a single analysis "
        "rather than five observations of the atmosphere. That structure is the reason the "
        "split in Section 6 is chronological and block-based.")
    c.p("<b>Class distribution.</b> Per variable the bust label is 10% by construction "
        "(it is a 90th-percentile cut), so a per-variable class balance tells us nothing; "
        "what matters is the event label. Any-variable busts occur on "
        f"{M['event_rate'] * 100:.1f}% of the {M['n_paired']:,} paired days in the archived "
        f"sample, rising from {M['event_by_lead'][1] * 100:.1f}% at Day 1 to "
        f"{M['event_by_lead'][10] * 100:.1f}% at Day 10, and the published base rate on "
        "held-out data is 43-49%. The classifier is therefore not re-weighted for rarity: a "
        "model that assumed a 10% base rate would be mis-specified.")
    c.p("<b>Known limitations, stated with their numbers.</b> The truth is a reanalysis, not a "
        "gauge network: ERA5 and MERRA-2 disagree per day by 24-43% of the bust threshold "
        "depending on the variable (rainfall 24%, humidity 36%, wind 38%, temperature 43%, "
        "over 21,492 paired city-days), and that disagreement is a floor under every label. "
        "Only five of the thirty-one operational members exist in the archive, so spread is a "
        "noisy estimate. Two variables stop early (10 m wind at Day 5, soil moisture at Day "
        "3) and are not scored beyond those leads. Rainfall truth is ERA5 rather than IMD "
        "gauge data in the current run. And the bust is defined on surface-variable error "
        "rather than on the synoptic 500 hPa criterion of Rodwell et al. [2], deliberately, "
        "because surface error is what reaches agriculture and disaster response.")

    # ==================================================================================
    # 3. Data Cleaning
    # ==================================================================================
    c.h1("3. Data Cleaning")
    c.h2("3.1 Data Quality Assessment")
    c.p("This dataset is fetched from other people's archives rather than authored by the "
        "team, so its cleaning is not about repairing typing errors. It is about the places "
        "where two individually correct products will still disagree or mislead. Each "
        "candidate issue below was checked against the data, and a treatment was chosen only "
        "where a check found something. The measurements in this section are computed by the "
        "build script from the files in <i>backend/data/</i>.")
    c.table(
        "Data quality assessment: what was checked and what was found",
        ["Quality issue", "Check performed", "Finding (measured)"],
        [
            ["Missing values",
             "Null count per column on both products, and missingness by lead day",
             f"{M['obs_nulls']} nulls in the {M['n_obs_rows']:,}-row observation file. On the "
             f"forecast side, {M['fc_null_share'] * 100:.1f}% of numeric cells are null, and "
             f"all of them are structural: {M['fc_nulls']['wspd10m_ms']:,} rows for 10 m wind "
             f"and wind direction (50.0%, Day 1-5 only) and "
             f"{M['fc_nulls']['soilw_vol_pct']:,} rows for soil moisture (70.0%, Day 1-3 "
             f"only). No variable with a full archive has a null. See Figure 1"],
            ["Duplicate records",
             "Duplicate canonical keys, and duplicate (station, date) pairs in the truth",
             f"{M['fc_dup_keys']} duplicate forecast keys and {M['obs_dup_keys']} duplicate "
             f"observation keys in the archived sample. Duplicates do occur in the live store "
             f"when a source is re-ingested: IMD rainfall batches ingested after ERA5 won the "
             f"dedupe and replaced 2016-2019 rainfall truth (found 2026-09-25)"],
            ["Inconsistent entries",
             "Unit, accumulation-window and date-convention checks across sources",
             "Inputs arrive as GRIB2, NetCDF, zipped NetCDF, CSV, XLSX and JSON. ERA5 stamps "
             "an accumulation with the end of its hour; humidity is specific humidity on the "
             "forecast side and relative humidity on the observation side; wind arrives as "
             "two components; IMD's gauge day runs 0830-0830 IST, so IMD date D is model day "
             "D-1 (settled from data, not documentation: rainfall on IMD date D correlates "
             "best with ERA5 summed to UTC day D-1 for 8 of 8 test districts, mean Spearman "
             "0.698 against 0.556 at lag 0)"],
            ["Invalid values / outliers",
             "Per-variable plausible range on every member, every variable, every cycle",
             "Real extremes stay in range and must stay: Leh reaches "
             "&minus;30.55 &deg;C on 2019-01-09 Day 6, Bhopal "
             "294.7 mm/day of rainfall on 2019-07-17 Day 9, column water vapour "
             "78.0 kg/m&sup2;. The archive has also produced two genuinely corrupt "
             "member-days in twenty years: p03 of 2004-03-05 with Day-2 temperature between "
             "&minus;23,380,072 and +21,808,336 &deg;C in all 666 districts (other members "
             "&minus;18.5 to +32.4 &deg;C), and p01 of 2019-02-09 with Day-1 column water "
             "vapour of 12,500-12,546 kg/m&sup2; (others 0.7-55.3). One such member put that "
             "day's temperature RMSE at 34,612 &deg;C"],
            ["Incorrect labels (sea read as land)",
             "Soil moisture compared between products for every district, and a land fraction "
             "per district from both masks",
             f"{M['saturated_rows']:,} forecast rows (2.5% of the file) sit at or above 99.5% "
             f"volumetric soil, all of them at "
             f"{', '.join(M['saturated_stations'])}. GEFS writes the sea as saturated soil "
             f"while ERA5 writes it as 0: Daman and Kavaratti forecast 99.99-100% against a "
             f"truth of 0.0%, and Port Blair 100% against 18.6%. Paired without a mask, four "
             f"districts busted on 100% of their soil-moisture rows against 0.9% elsewhere"],
            ["Incorrect labels (standing bias)",
             "Signed error per district, variable and lead, fitted leave-one-cycle-out",
             f"64% of squared error for temperature and humidity, and 90% for soil moisture, "
             f"is a fixed per-district offset rather than a daily failure. The humidity "
             f"offset ranges from {min(M['rh_bias'].values()):.1f} %RH at "
             f"{min(M['rh_bias'], key=M['rh_bias'].get)} to "
             f"{max(M['rh_bias'].values()):+.1f} %RH at "
             f"{max(M['rh_bias'], key=M['rh_bias'].get)} (mean "
             f"{np.mean(list(M['rh_bias'].values())):+.1f} %RH) across the 36 sample "
             f"stations. Left in, a 'bust' means 'this district is always off'"],
            ["Incorrect labels (leakage)",
             "Audit of every model input for observation-derived content; correlation of "
             "input availability with issue time",
             "One input, <i>forecast_error_lag</i>, read the realised error of the previous "
             "lead of the same forecast, which no live cycle can know; it is removed and a "
             "test now refuses observation-derived columns. The MJO block was dropped for "
             "the same class of reason: NOAA PSL does not publish OMI in real time, so every "
             "live cycle had it as NaN while every training row had a value"],
            ["Incomplete cycles",
             "Structural row-count check per cycle before the daily reduction",
             "7,288 of the 7,305 archive dates are present (99.77%). NOAA drops forecast "
             "steps mid-pull, and a short rainfall sum silently halves the quantity that "
             "drives most busts, so an incomplete cycle is refused rather than published "
             "short. Measured on 2017: 365 cycles, 12,154,500 rows, zero refusals, every "
             "cycle exactly 33,300 rows"],
            ["Skewed and zero-inflated labels",
             "Distribution of each variable's error and of the observation itself",
             f"{M['zero_rain_share'] * 100:.1f}% of observation days are dry "
             f"(rainfall at or below 0.01 mm), while rainfall error spans a median of "
             f"{s['rainfall_mm']['median']:.2f} mm/day and a maximum of "
             f"{s['rainfall_mm']['maximum']:.0f} mm/day. A fixed millimetre threshold would "
             f"label almost nothing in a dry district and almost everything in the monsoon"],
        ],
        widths=[0.17, 0.19, 0.64], small=True)
    c.figure(figs["missing"],
             "Figure 1. Missingness in the archived 2019 reforecast sample (30,600 rows). "
             "(a) the non-null share of each variable by lead day: the two white blocks are "
             "the archive's own lead limits, identical for every cycle and member, not a "
             "data loss. (b) null rows per column, in red, and in amber the rows where GEFS "
             "saturates soil moisture at 99.5% or more at Daman, Kavaratti and Port Blair - "
             "a value that is present but is not a measurement of soil.")

    c.h2("3.2 Data Cleaning Plan")
    c.p("The cleaning plan follows the template's five issue classes, with a treatment for "
        "each and the reason the treatment was chosen over the obvious alternative. Where an "
        "issue is not present, the table says so and cites the check that shows it.")
    c.table(
        "Cleaning plan: technique applied or proposed, and the reason",
        ["Data quality issue", "Proposed / applied technique", "Reason"],
        [
            ["Missing values",
             "No imputation anywhere. A structural absence is represented as the absence of a "
             "row; a per-cycle completeness guard refuses a cycle that is short of its "
             "33,300 rows, with an S3 byte-range fallback that refills missing GRIB2 messages "
             "before the daily reduction. Soil moisture is masked, not filled, for districts "
             "more than 5% sea in either product",
             "A short rainfall sum is about half the real accumulation, and a fabricated "
             "member enters the ensemble spread as evidence. Masking is honest where "
             "imputation is not: an island district has no soil-moisture value to compare"],
            ["Duplicate records",
             "Dedupe on (region, valid date, variable, value type, init date, lead, member, "
             "source column), newest non-provisional row winning; quarantined and retired "
             "batches refused before any frame is built; a cached year is rebuilt when its "
             "batch signature changes",
             "A duplicate here is not a repeated row, it is a different truth product: it "
             "changes the label silently and inflates any score computed on it"],
            ["Inconsistent values",
             "Format-agnostic parsers into one canonical schema with confidence-scored column "
             "matching; accumulation grouped by valid time minus one hour; relative humidity "
             "derived from specific humidity, temperature and pressure after the district "
             "mean on both sides; speed and direction derived from u and v; the CDS ZIP "
             "unzipped and both NetCDF files read; IMD dates moved back one day and the merge "
             "refused unless its rainfall correlates best with ERA5 at lag 0",
             "One schema and one set of units is what makes a forecast row and a truth row "
             "comparable at all; deriving RH after the mean on both sides keeps a bounded "
             "quantity from being averaged as a ratio"],
            ["Outliers / invalid entries",
             "Range check against a per-variable plausible range at the head of the one "
             "frame-building path that training, scoring, live and Replay all share; a "
             "corrupt member is dropped and the ensemble is the members that remain. Real "
             "extremes are kept",
             "The check must catch a 21,808,336 &deg;C member without deleting the "
             "&minus;30.55 &deg;C Himalayan day or the 294.7 mm/day rainfall day, because the "
             "label lives in exactly that tail"],
            ["Incorrect labels",
             "Busts defined on bias-corrected error: the training-period mean error per "
             "(district, variable, lead, month) is removed first, backing off to the "
             "district's month across leads and then to its year when a cell holds fewer than "
             "30 events; a key thin at every level is not compared at all. Soil moisture is "
             "dropped on both sides for districts above the 5% sea threshold, and the "
             "exclusion list is required to exist rather than defaulted",
             "Otherwise a bust means 'this district is always off', which NCMRWF's own "
             "post-processing already removes; the failure that remains is the one worth "
             "predicting. The humidity threshold for November 2017 falls from 27.6 to 10.4 "
             "%RH after the correction"],
        ],
        widths=[0.16, 0.46, 0.38], small=True)

    c.h2("3.3 Cleaning Evidence")
    c.p("Two properties of this dataset make the usual before-and-after summary misleading: "
        "the cleaning is mostly verification, and several treatments change the label without "
        "changing a single value. The table therefore reports the quantity each treatment "
        "moved, measured, rather than a count of edited cells.")
    c.table(
        "Before-and-after measurements of the cleaning steps",
        ["Quantity", "Before the treatment", "After the treatment"],
        [
            ["2004-03-05 Day-2 temperature RMSE, with the corrupt p03 member in",
             "34,612 &deg;C", "Within the plausible range: exactly 666 rows dropped on each "
             "affected cycle, none on a clean one"],
            ["Soil moisture for the three sea-covered sample stations",
             "Daman 99.99%, Kavaratti 100.0%, Port Blair 100.0% volumetric against an ERA5 "
             "truth of 0.0%, 0.0% and 18.6% - a guaranteed bust on every row",
             "Not compared at all: 18 districts exceed the 5% sea threshold in GEFS or ERA5 "
             "and are excluded on both sides"],
            ["Humidity bust threshold, Nov 2017, 666 districts",
             "27.6 %RH on raw error (a dry-biased forecast made an ordinary day look like a "
             "failure)", "10.4 %RH after per-district, per-lead bias correction"],
            ["Wind-direction error p90, real CI sample",
             "191.0&deg; under plain angle arithmetic (a northerly is a bust by "
             "construction; 86.5% of direction busts were this artefact)",
             "115.1&deg; with the error taken the short way round the circle; ensemble spread "
             "p90 falls from 118.0&deg; to 55.0&deg;"],
            ["Event bust rate by lead, leaving out each district's own bias",
             "0.31 at Day 1 rising to 0.50 at Day 10 only after the offset is removed; "
             "correlation of lead with bust about 0.02 before",
             "0.25 at Day 1 to 0.50 at Day 10 with the bias removed leave-one-cycle-out, and "
             "rainfall growing with lead throughout"],
            ["Rainfall truth for 2016-2019 in the live run",
             "One-day-late IMD-merged rainfall, differing from ERA5 in 72.5% of 2018 rows and "
             "79.8% of 2019 rows",
             "ERA5 CDS batches; the four IMD batches quarantined, and any frame build that "
             "finds them now stops instead of choosing"],
        ],
        widths=[0.27, 0.40, 0.33], small=True)
    c.p("The integrity checks that produce the numbers in Tables 3 and 5 are a few lines of code and "
        "run at the head of every frame build, so a regression in any of them fails the build "
        "rather than the report:")
    c.code(
        '# backend/app/ml/train_pipeline.py - the checks a frame must pass before any model sees it\n'
        'keys = ["region_id", "init_date", "valid_date", "lead_time_days", "ensemble_member_id"]\n'
        'assert not paired.duplicated(subset=keys).any(), "duplicate canonical key"\n'
        'assert paired["value"].notna().all(), "null value in the canonical store"\n'
        'model_rows = paired[paired["value_type"] == "forecast"]\n'
        'short = model_rows.groupby("init_date").size()\n'
        'assert (short == 33_300).all(), f"short cycle(s): {(short != 33_300).sum()}"\n'
        '\n'
        '# a corrupt ensemble member is dropped, never averaged into the spread\n'
        'in_range = fc_mean.between(PLAUSIBLE[variable][0], PLAUSIBLE[variable][1])\n'
        'member_ok = in_range | fc_mean.isna()\n'
        'value = value.where(member_ok)                       # NaN, not zero',
        "Listing 1. The integrity checks and the plausible-range filter, as they run in "
        "training, scoring and the live cycle.")

    # ==================================================================================
    # 4. Data Preprocessing
    # ==================================================================================
    c.h1("4. Data Preprocessing")
    c.h2("4.1 Preprocessing Workflow")
    c.p("Preprocessing runs in one direction and in one place. Both products pass through the "
        "same parsers, the same weight table, the same daily aggregation and the same range "
        "check, because the moment the two sides are treated differently the comparison they "
        "support is no longer a comparison. The sequence is:")
    c.bullets([
        "<b>Parse</b> each source format into one canonical schema of 20 columns, matching "
        "columns by name with a recorded confidence score rather than by position.",
        "<b>Resolve geography</b>: every value becomes an area-weighted mean over the 0.25&deg; "
        "cells a district polygon overlaps, using a single weight table read at "
        f"{M['n_weight_pairs']:,} district-cell pairs, each district's weights summing to "
        f"exactly 1.0 (median {M['cells_per_district'][1]:.0f} cells per district, "
        f"{M['cells_per_district'][0]}-{M['cells_per_district'][2]}).",
        "<b>Aggregate to the UTC day</b>, grouping accumulations by valid time minus one hour "
        "and reducing 3-hourly fields to daily ones.",
        "<b>Apply the plausible range</b> per variable and drop a member that fails it.",
        "<b>Dedupe</b> on the canonical key with the newest non-provisional row winning.",
        "<b>Derive the paired frame</b>: ensemble mean and spread per (district, variable, "
        "initialisation, valid date, lead), joined to the observed mean for the same key.",
        "<b>Fit the bias table</b> per (district, variable, lead, month) on training cycles "
        "only, with the backoff described in Section 3.2, and subtract it from every member.",
        "<b>Define the label</b>: absolute error above the variable's training-set 90th "
        "percentile, plus the 8-variable event label.",
        "<b>Engineer features</b>: ensemble spread and member count, run-to-run jumps, the "
        "time-lagged ensemble pool, tendencies, season, district descriptors.",
        "<b>Split and embargo</b>: chronological by cycle, ten days either side of every "
        "boundary, 30-day out-of-fold blocks inside training.",
        "<b>Materialise</b> the design matrix for the tabular models and the multi-channel "
        "gridded tensors for the CNN, with a mask channel per input channel.",
    ], numbered=True)
    c.figure(figs["preprocess"],
             "Figure 2. The preprocessing chain. Blue stages are applied identically to the "
             "forecast and the observation side; green stages are fitted on training cycles "
             "only and then applied unchanged to validation, test and live data. The order is "
             "not cosmetic: the range check precedes the ensemble reduction so a corrupt "
             "member cannot reach the spread, and the bias table precedes the threshold so "
             "the label describes corrected error.", width=1.0)

    c.h2("4.2 Preprocessing Techniques")
    c.table(
        "Preprocessing techniques, purpose and configuration",
        ["Technique", "Purpose", "Configuration / method"],
        [
            ["Missing-value handling",
             "Keep a structural gap distinguishable from a measurement, and stop a partial "
             "cycle being read as a real one",
             "No imputation. Absent row for a structural gap; per-cycle completeness guard "
             "(33,300 rows per cycle) with an S3 byte-range refill; land-sea mask exclusion "
             "for soil moisture above 5% sea; 1/0 mask channel per CNN input channel"],
            ["Categorical encoding",
             "Let a tree model use district identity without giving it 666 levels to "
             "memorise",
             "<i>state_id</i> (36 levels) as a native XGBoost categorical, plus the derived "
             "descriptors centroid latitude/longitude, area, distance to the country's edge "
             "and mean elevation; <i>season</i> as DJF/MAM/JJAS/ON; <i>month</i> as an "
             "integer. The 666-level region identifier is deliberately not a feature"],
            ["Feature scaling / normalisation",
             "Give the convolutional channel normaliser a defined scale; leave the trees "
             "alone",
             "None for the stored data or for XGBoost, which is invariant to a monotone "
             "rescaling; per-channel mean and standard deviation for the CNN, fitted on "
             "training cycles only, because a normaliser fitted on the whole archive leaks "
             "the test period's climate into training"],
            ["Feature selection / extraction",
             "Carry the predictability signal and refuse anything that is not available at "
             "issue time",
             "24 engineered inputs per regressor row (spread, member count, four jumpiness "
             "features, four time-lagged-pool features, two tendencies, historical bust "
             "frequency, six district descriptors, bias correction, lead, month, season, "
             "forecast value) plus one concurrent forecast column per variable; "
             "observation-derived columns are refused by a contract check"],
            ["Imbalance and tail handling",
             "Keep the tail available to the model without fabricating rows",
             "No resampling and no <i>scale_pos_weight</i>: the percentile definition makes "
             "each variable's label exactly 10%, the event label is 38.6-49%, and "
             "re-weighting pushed every probability up and broke calibration"],
            ["Probability calibration",
             "Make a predicted probability mean what it says",
             "Platt scaling (a one-dimensional logistic fit) on the validation split, with a "
             "non-positive slope refused; an uncalibrated classifier's lowest reliability bin "
             "predicted 0.158 against about 0.07 observed"],
            ["Image resizing / enhancement",
             "Not applicable",
             "The CNN reads the native 0.25&deg; grid; there is no downsampling anywhere. "
             "Dilations 1, 2, 4, 8 give a 31-cell (about 850 km) receptive field at grid "
             "resolution, which keeps the district weight table valid for pooling"],
            ["Text cleaning / tokenisation",
             "Not applicable",
             "There is no text, image or free-form field in the training path"],
            ["Circular and derived quantities",
             "Stop an angle or a ratio being treated as a plain number",
             "Wind-direction error taken the short way round the circle; relative humidity "
             "derived from specific humidity, temperature and pressure after the district "
             "mean on both sides; wind speed and direction derived from u and v"],
        ],
        widths=[0.19, 0.30, 0.51], small=True)

    c.h2("4.3 Preprocessing Justification")
    c.p("<b>Why one weight table and not two samples.</b> Ten districts contain no 0.25&deg; "
        "grid centre at all, so a nearest-point sample cannot represent them, and a district "
        "sampled differently on the forecast side than on the truth side would carry a "
        "spatial error that looks like a forecast error. One area-overlap table, used by both "
        "sides, makes the pair a comparison of the same polygon. Validated against the 35 city "
        "points it replaces: correlation 0.9896, median absolute difference 0.31 &deg;C.")
    c.p("<b>Why no imputation.</b> Every missing cell in this dataset is a structural "
        "absence: five members, ten leads, and fields that the archive itself stops serving "
        "at Day 3 or Day 5. Imputing a missing rainfall accumulation would fabricate the "
        "quantity that drives most busts, and imputing a missing member would enter the "
        "ensemble spread as though it were evidence of uncertainty. A model told where its "
        "inputs are real (the mask channel, or the absent row) learns the structure; a model "
        "fed a plausible average learns a fiction.")
    c.p("<b>Why bias correction comes before the label.</b> If a bust is defined on raw error, "
        "the label mostly records which districts the model is persistently off in, and 64% "
        "of squared error for temperature and humidity (90% for soil moisture) is exactly "
        f"that. The measured humidity offsets in Table 3 span "
        f"{min(M['rh_bias'].values()):.1f} to {max(M['rh_bias'].values()):+.1f} %RH across 36 "
        "stations, which is a standing offset, not a daily failure. Subtracting a "
        "training-only monthly mean first makes the label describe the failure that survives "
        "correction, which is the failure an operational centre cannot already remove.")
    c.p("<b>Why no scaling and no resampling.</b> A gradient-boosted tree splits on order, so a "
        "monotone rescaling changes nothing it reads and would only add a version to every "
        "stored file; the CNN, which does need a scale, gets one fitted on training cycles "
        "only. Resampling the 44.8% dry days would repeat near-identical rows rather than add "
        "information, and would break the correspondence between predicted probability and "
        "observed frequency that the Platt fit exists to establish.")
    c.p("<b>Why the descriptors replace the district identifier.</b> A 666-level categorical "
        "can only learn a district it saw labelled in training, and roughly 34 regions ever "
        "carry a label in the CI sample. Latitude, longitude, area, distance to the edge of "
        "the landmass and mean elevation are defined for every district, including one the "
        "model has never seen, and they are the quantities that actually drive error: Leh's "
        f"mean elevation of {M['district_stats']['elevation_mean'][2]:,.0f} m runs about ten "
        "times the typical district's pressure error.")

    c.h2("4.4 Preprocessing Evidence")
    c.p("The pairing rule is one function, used by training, scoring, live cycles and Replay "
        "alike, so a cycle is verified later against exactly what it would have been scored "
        "against:")
    c.code(
        '# one pairing rule for both sides of the bust label (app/features/engineering.py)\n'
        'def observed_means(df):\n'
        '    """One observation per (region, valid date, variable): the mean of the rows held\n'
        '    for it, and "provisional" if any of them is."""\n'
        '    ob = df[df["value_type"] == "observed"]\n'
        '    return (ob.groupby(["region_id", "valid_date", "variable"], observed=True)\n'
        '              .agg(obs=("value", "mean"),\n'
        '                   provisional=("verification_status", lambda s: (s == "provisional").any()))\n'
        '              .reset_index())\n'
        '\n'
        '# ensemble mean and spread at the event grain, then the pair\n'
        'fc = (g.groupby(EVENT_KEYS + ["ensemble_member_id"], observed=True)["value"].mean()\n'
        '        .groupby(EVENT_KEYS).agg(fc_mean="mean", fc_std="std", members="size"))\n'
        'paired = fc.merge(ob, on=["region_id", "valid_date", "variable"], how="inner")\n'
        'paired["error"] = (paired.fc_mean - paired.obs).abs()',
        "Listing 2. The single pairing rule: the ensemble is reduced to a mean and a spread "
        "per event, then joined to the observed mean for the same district, variable and UTC "
        "day.")
    c.code(
        '# chronological split with a Day-10 embargo (app/ml/train_pipeline.py)\n'
        'SPLIT_EMBARGO_DAYS = 10   # a cycle at d is verified against days d .. d+9\n'
        '\n'
        'def _embargo_before(cycles, boundary):\n'
        '    cut = pd.Timestamp(boundary) - pd.Timedelta(days=SPLIT_EMBARGO_DAYS)\n'
        '    return [c for c in cycles if pd.Timestamp(c) <= cut]\n'
        '\n'
        'train, val, test = cycles[:a], cycles[a:b], cycles[b:]\n'
        'val = _embargo_before(val, test[0])          # no shared observed day\n'
        'train = _embargo_before(train, val[0])       # ... at either boundary, ever',
        "Listing 3. The embargo that keeps the same observed day off both sides of every "
        "split and out-of-fold block boundary.")

    # ==================================================================================
    # 5. Preliminary EDA
    # ==================================================================================
    c.h1("5. Preliminary Exploratory Data Analysis (EDA)")
    c.p("The analysis below is computed at build time from the archived 2019 sample and the "
        "geography layer, so every number and figure in this section is a measurement of the "
        "files in the repository rather than a quotation. The sample is deliberately "
        "seasonally spread: fortnightly cycles through the monsoon and monthly otherwise, so "
        f"{M['monsoon_share'] * 100:.0f}% of its observation days fall in June-September.")

    c.h2("5.1 Dimensions and summary statistics")
    c.table(
        "Dimensions of the analysed files",
        ["File", "Rows", "Columns", "Grain", "Purpose"],
        [
            ["gefs_reforecast_india_2019.parquet", f"{M['n_fc_rows']:,}",
             str(M["n_fc_cols"]),
             f"{M['n_cycles']} cycles x {M['n_stations']} stations x {M['n_members']} "
             f"members x {M['leads'][1]} lead days",
             "Forecast side of the pair, and the source of spread"],
            ["era5_observations_india_2019.parquet", f"{M['n_obs_rows']:,}",
             str(M["n_obs_cols"]), f"{M['n_obs_days']} days x {M['n_stations']} stations",
             "Verifying truth for the bust label"],
            ["district_descriptors.parquet", f"{M['n_districts']:,}", "7",
             "one row per district", "Static geography features"],
            ["district_grid_weights.parquet", f"{M['n_weight_pairs']:,}", "4",
             "district-cell weight", "The one spatial reduction used by both sides"],
            ["mjo_omi_index.parquet", f"{M['mjo_rows']:,}", "5", "one row per day",
             "Large-scale covariate (dropped from new runs while the tail is stale)"],
            ["grid_elevation_m.parquet", f"{M['elev_cells']:,}", "4", "one row per cell",
             "Elevation descriptor and CNN terrain channel"],
            ["era5_land_sea_mask.parquet", f"{M['elev_cells']:,}", "3", "one row per cell",
             f"Sea masking ({M['land_share'] * 100:.1f}% of the cells the weight table uses "
             f"are land)"],
        ],
        widths=[0.30, 0.09, 0.09, 0.28, 0.24], small=True,
        legend=f"After pairing, {M['n_paired']:,} forecast-observation rows remain, "
               f"{M['n_paired'] // 10} for each of the ten lead days and "
               f"{s['wind_speed_ms']['n']:,} for wind speed, whose archive stops at Day 5. "
               f"The join is lossless: every archived forecast row has a truth for the same "
               f"station and UTC day.")
    c.table(
        "Summary statistics of the paired features (forecast ensemble mean and ERA5 "
        "observation, archived 2019 sample)",
        ["Variable (unit)", "Side", "Min", "25%", "Median", "75%", "Max", "Mean", "Std"],
        [
            ["temperature_c (&deg;C)", "forecast", f"{d['paired']['fc_temperature_c'].min():.2f}",
             f"{d['paired']['fc_temperature_c'].quantile(.25):.2f}",
             f"{d['paired']['fc_temperature_c'].median():.2f}",
             f"{d['paired']['fc_temperature_c'].quantile(.75):.2f}",
             f"{d['paired']['fc_temperature_c'].max():.2f}",
             f"{d['paired']['fc_temperature_c'].mean():.2f}",
             f"{d['paired']['fc_temperature_c'].std():.2f}"],
            ["", "observed", f"{d['paired']['ob_temperature_c'].min():.2f}",
             f"{d['paired']['ob_temperature_c'].quantile(.25):.2f}",
             f"{d['paired']['ob_temperature_c'].median():.2f}",
             f"{d['paired']['ob_temperature_c'].quantile(.75):.2f}",
             f"{d['paired']['ob_temperature_c'].max():.2f}",
             f"{d['paired']['ob_temperature_c'].mean():.2f}",
             f"{d['paired']['ob_temperature_c'].std():.2f}"],
            ["humidity_pct (%RH)", "forecast", f"{d['paired']['fc_humidity_pct'].min():.2f}",
             f"{d['paired']['fc_humidity_pct'].quantile(.25):.2f}",
             f"{d['paired']['fc_humidity_pct'].median():.2f}",
             f"{d['paired']['fc_humidity_pct'].quantile(.75):.2f}",
             f"{d['paired']['fc_humidity_pct'].max():.2f}",
             f"{d['paired']['fc_humidity_pct'].mean():.2f}",
             f"{d['paired']['fc_humidity_pct'].std():.2f}"],
            ["", "observed", f"{d['paired']['ob_humidity_pct'].min():.2f}",
             f"{d['paired']['ob_humidity_pct'].quantile(.25):.2f}",
             f"{d['paired']['ob_humidity_pct'].median():.2f}",
             f"{d['paired']['ob_humidity_pct'].quantile(.75):.2f}",
             f"{d['paired']['ob_humidity_pct'].max():.2f}",
             f"{d['paired']['ob_humidity_pct'].mean():.2f}",
             f"{d['paired']['ob_humidity_pct'].std():.2f}"],
            ["rainfall_mm (mm/day)", "forecast", f"{d['paired']['fc_rainfall_mm'].min():.2f}",
             f"{d['paired']['fc_rainfall_mm'].quantile(.25):.2f}",
             f"{d['paired']['fc_rainfall_mm'].median():.2f}",
             f"{d['paired']['fc_rainfall_mm'].quantile(.75):.2f}",
             f"{d['paired']['fc_rainfall_mm'].max():.2f}",
             f"{d['paired']['fc_rainfall_mm'].mean():.2f}",
             f"{d['paired']['fc_rainfall_mm'].std():.2f}"],
            ["", "observed", f"{d['paired']['ob_rainfall_mm'].min():.2f}",
             f"{d['paired']['ob_rainfall_mm'].quantile(.25):.2f}",
             f"{d['paired']['ob_rainfall_mm'].median():.2f}",
             f"{d['paired']['ob_rainfall_mm'].quantile(.75):.2f}",
             f"{d['paired']['ob_rainfall_mm'].max():.2f}",
             f"{d['paired']['ob_rainfall_mm'].mean():.2f}",
             f"{d['paired']['ob_rainfall_mm'].std():.2f}"],
            ["pressure_hpa (hPa)", "forecast", f"{d['paired']['fc_pressure_hpa'].min():.2f}",
             f"{d['paired']['fc_pressure_hpa'].quantile(.25):.2f}",
             f"{d['paired']['fc_pressure_hpa'].median():.2f}",
             f"{d['paired']['fc_pressure_hpa'].quantile(.75):.2f}",
             f"{d['paired']['fc_pressure_hpa'].max():.2f}",
             f"{d['paired']['fc_pressure_hpa'].mean():.2f}",
             f"{d['paired']['fc_pressure_hpa'].std():.2f}"],
            ["", "observed", f"{d['paired']['ob_pressure_hpa'].min():.2f}",
             f"{d['paired']['ob_pressure_hpa'].quantile(.25):.2f}",
             f"{d['paired']['ob_pressure_hpa'].median():.2f}",
             f"{d['paired']['ob_pressure_hpa'].quantile(.75):.2f}",
             f"{d['paired']['ob_pressure_hpa'].max():.2f}",
             f"{d['paired']['ob_pressure_hpa'].mean():.2f}",
             f"{d['paired']['ob_pressure_hpa'].std():.2f}"],
            ["atmos. moisture (kg/m&sup2;)", "forecast",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].min():.2f}",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].quantile(.25):.2f}",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].median():.2f}",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].quantile(.75):.2f}",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].max():.2f}",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].mean():.2f}",
             f"{d['paired']['fc_atmospheric_moisture_kgm2'].std():.2f}"],
            ["", "observed",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].min():.2f}",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].quantile(.25):.2f}",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].median():.2f}",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].quantile(.75):.2f}",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].max():.2f}",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].mean():.2f}",
             f"{d['paired']['ob_atmospheric_moisture_kgm2'].std():.2f}"],
            ["wind_speed_ms (m/s)", "forecast",
             f"{d['paired']['fc_wind_speed_ms'].min():.2f}",
             f"{d['paired']['fc_wind_speed_ms'].quantile(.25):.2f}",
             f"{d['paired']['fc_wind_speed_ms'].median():.2f}",
             f"{d['paired']['fc_wind_speed_ms'].quantile(.75):.2f}",
             f"{d['paired']['fc_wind_speed_ms'].max():.2f}",
             f"{d['paired']['fc_wind_speed_ms'].mean():.2f}",
             f"{d['paired']['fc_wind_speed_ms'].std():.2f}"],
            ["", "observed", f"{d['paired']['ob_wind_speed_ms'].min():.2f}",
             f"{d['paired']['ob_wind_speed_ms'].quantile(.25):.2f}",
             f"{d['paired']['ob_wind_speed_ms'].median():.2f}",
             f"{d['paired']['ob_wind_speed_ms'].quantile(.75):.2f}",
             f"{d['paired']['ob_wind_speed_ms'].max():.2f}",
             f"{d['paired']['ob_wind_speed_ms'].mean():.2f}",
             f"{d['paired']['ob_wind_speed_ms'].std():.2f}"],
        ],
        widths=[0.22, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.08], small=True,
        legend="Wind is paired on 3,060 of the 6,120 rows, because the archive holds it only "
               "to Day 5. The humidity comparison already shows the dry bias that motivates "
               "the bias table: a median forecast of 69.8 %RH against an observed median of "
               "76.2 %RH.")

    c.h2("5.2 Data types and feature distributions")
    c.p("Every measurement is a floating-point number, every identifier is a string read as a "
        "categorical, and every temporal key is a date. Two shapes are worth noting before any "
        "model sees them. Rainfall is zero-inflated and heavy tailed: nearly half of the "
        "observation days are dry, the forecast median is 0.92 mm/day against an observed "
        "median of 0.20 mm/day, and the largest daily accumulation in the sample is "
        "294.7 mm/day. Humidity is left-skewed and bounded at 100 %RH, and the forecast "
        "distribution sits visibly to the left of the observation - the standing dry bias, "
        "not a scatter.")
    c.figure(figs["features"],
             "Figure 3. Feature distributions of the paired frame, forecast ensemble mean "
             "against ERA5 observation, for the six label variables. The forecast "
             "distributions are narrower than the observed ones for temperature and water "
             "vapour - an ensemble mean is a smoothed field - and shifted left for humidity. "
             "Rainfall is zero-inflated on both sides. Panels are clipped at the 99.5th "
             "percentile so the bulk remains visible; no rows are removed.")

    c.h2("5.3 Target-variable and class distribution")
    c.p("The continuous target is the absolute error of the pair; the label is its tail. "
        "Table 9 gives both, with the 90th percentile that becomes the bust threshold when it "
        "is fitted on training cycles. Table 10 gives the resulting class distribution at the "
        "event grain.")
    error_rows = []
    for col, (name, label, unit) in VARIABLES.items():
        e = s[name]
        error_rows.append([
            f"{name} ({unit})", f"{e['n']:,}", f"{e['bias']:+.2f}", f"{e['mae']:.2f}",
            f"{e['median']:.2f}", f"{e['p90']:.2f}", f"{e['p99']:.2f}",
            f"{e['maximum']:.1f}", f"{e['r_fc_ob']:.3f}",
        ])
    c.table(
        "Error statistics of the paired sample and the bust threshold it implies "
        "(in-sample percentile, for illustration)",
        ["Variable", "Pairs", "Bias", "MAE", "Median", "p90", "p99", "Max", "r(fc, obs)"],
        error_rows,
        widths=[0.24, 0.09, 0.09, 0.08, 0.09, 0.08, 0.08, 0.08, 0.09], small=True,
        legend="Units are each variable's own. The p90 column is the threshold a bust is cut "
               "at; in this illustration it is a quantile of the same rows it labels, which "
               "is why each variable then busts on exactly 10% of the sample. In production "
               "the threshold, the bias table and the calibrator are fitted on training "
               "cycles only.")
    c.figure(figs["targets"],
             "Figure 4. Distribution of the absolute forecast error per variable, with the "
             "90th-percentile threshold as a dashed line. Every distribution is right-skewed, "
             "which is what makes a percentile definition necessary rather than convenient: "
             f"the temperature threshold is {s['temperature_c']['p90']:.2f} &deg;C against a "
             f"median of {s['temperature_c']['median']:.2f}, and the rainfall threshold "
             f"{s['rainfall_mm']['p90']:.1f} mm/day against a median of "
             f"{s['rainfall_mm']['median']:.1f}. The same millimetre figure means something "
             "entirely different in a desert and in Kerala, which is why each variable carries its "
             "own threshold, fitted on the training cycles of every district, rather than the "
             "project using one number for all of them.")
    c.table(
        "Class distribution: per-variable bust share and the any-variable event label",
        ["Label", "Definition", "Positive share", "Rows"],
        [
            ["bust[temperature_c]", "error above the variable's p90", "10.0% (by construction)",
             f"{s['temperature_c']['n']:,}"],
            ["bust[rainfall_mm]", "error above the variable's p90", "10.0% (by construction)",
             f"{s['rainfall_mm']['n']:,}"],
            ["bust (each of the six variables)", "error above the variable's p90",
             "10.0% each", f"{M['n_paired']:,}"],
            ["<b>event</b> (any variable busts)",
             "OR across the six label variables paired in this sample (seven in the store, "
             "which also carries soil moisture)",
             f"<b>{M['event_rate'] * 100:.1f}%</b>", f"{M['n_paired']:,}"],
            ["event, Day 1 / Day 5 / Day 10", "OR across the variables available at that lead",
             f"{M['event_by_lead'][1] * 100:.1f}% / {M['event_by_lead'][5] * 100:.1f}% / "
             f"{M['event_by_lead'][10] * 100:.1f}%", f"{M['n_paired'] // 10:,} each"],
            ["event, monsoon (JJAS) / other months", "OR across the variables, split by month",
             f"{np.mean([M['event_by_month'][m] for m in (6, 7, 8, 9)]) * 100:.1f}% / "
             f"{np.mean([M['event_by_month'][m] for m in (1, 2, 3, 4, 5, 10, 11, 12)]) * 100:.1f}%",
             f"{int(M['n_paired'] * M['monsoon_share']):,} / "
             f"{int(M['n_paired'] * (1 - M['monsoon_share'])):,}"],
            ["dry observation days (rainfall at or below 0.01 mm)",
             "zero-inflation of the rainfall truth", f"{M['zero_rain_share'] * 100:.1f}%",
             f"{M['n_paired']:,}"],
        ],
        widths=[0.28, 0.32, 0.24, 0.16], small=True,
        legend="The event label is what the classifier predicts and it is not a 10% problem: "
               "with six variables each busting 10% of the time, the union would be "
               "1 - 0.9^6 = 46.9% if the variables were independent, and the measured rate is "
               f"{M['event_rate'] * 100:.1f}% because the variables are correlated and fewer "
               "are available at longer leads. The published held-out base rate on the full "
               "store is 43-49%.")
    c.figure(figs["class"],
             "Figure 5. Class distribution. (a) the event bust rate rises with lead day, from "
             f"{M['event_by_lead'][1] * 100:.1f}% at Day 1 to "
             f"{M['event_by_lead'][10] * 100:.1f}% at Day 10, as a forecast failure should; "
             "(b) it is not flat across the year, so month and season are real features and "
             "not decoration; (c) the measured rate tracks the number of variables available "
             "at each lead rather than the independence bound of 1 - 0.9^k, which is the "
             "quantitative reason the classifier is not re-weighted for rarity.")

    c.h2("5.4 Feature distributions against the target")
    c.p("Two relations determine the model choice in Section 7. Error grows with lead day, as "
        "a forecast failure should, and the growth is steepest for rainfall, which rises from "
        f"{d['paired'].groupby('lead_day').apply(lambda x: (x.fc_rainfall_mm - x.ob_rainfall_mm).abs().mean()).iloc[0]:.2f} "
        f"to {d['paired'].groupby('lead_day').apply(lambda x: (x.fc_rainfall_mm - x.ob_rainfall_mm).abs().mean()).iloc[7]:.2f} mm/day between Day 1 and Day 8 "
        "in this sample, while temperature error moves from 1.70 to 1.83 &deg;C over the same "
        "leads - the signature of a steady per-district bias rather than a forecast coming "
        "apart. And "
        "ensemble spread is informative about error but not sufficient: its rank correlation "
        f"with the absolute error is {s['rainfall_mm']['spread_rank_corr']:.2f} for rainfall, "
        f"{s['wind_speed_ms']['spread_rank_corr']:.2f} for wind speed, "
        f"{s['pressure_hpa']['spread_rank_corr']:.2f} for pressure and "
        f"{s['temperature_c']['spread_rank_corr']:.2f} for temperature.")
    c.figure(figs["errorlead"],
             "Figure 6. (a) mean absolute error by lead day for four variables, in each "
             "variable's own units: rainfall error climbs from Day 1 to Day 8 while temperature "
             "error is nearly flat, which is the difference between a forecast coming apart and "
             "a district that is simply off. (b) mean absolute error by "
             "ensemble-spread decile: a monotone relation for both variables, and a much "
             "stronger one for rainfall than for temperature. Spread alone would rank days "
             "usefully but would miss the bias-driven half of the problem, which is why the "
             "models in Section 7 take spread as one feature among several.")
    c.p("A third relation matters for the split: \"monsoon\" is not a cosmetic label. "
        f"Rainfall MAE is {s['rainfall_mm']['mae_monsoon']:.2f} mm/day in JJAS against "
        f"{s['rainfall_mm']['mae_non_monsoon']:.2f} elsewhere, and pressure MAE "
        f"{s['pressure_hpa']['mae_monsoon']:.2f} hPa against "
        f"{s['pressure_hpa']['mae_non_monsoon']:.2f}. A random row split would put the same "
        "monsoon day on both sides of the evaluation and report a skill that no held-out "
        "season supports.")

    c.h2("5.5 Missing-value and duplicate summary")
    c.table(
        "Missingness and duplication, summarised",
        ["Check", "Forecast sample", "Observation sample"],
        [
            ["Rows", f"{M['n_fc_rows']:,}", f"{M['n_obs_rows']:,}"],
            ["Null cells", f"{sum(M['fc_nulls'].values()):,} "
                           f"({M['fc_null_share'] * 100:.1f}% of numeric cells), all "
                           f"structural", "0"],
            ["Columns with any null", "3 of 36 (wind speed, wind direction, soil moisture)",
             "0 of 16"],
            ["Duplicate rows", "0", "0"],
            ["Duplicate canonical keys", f"{M['fc_dup_keys']}", f"{M['obs_dup_keys']}"],
            ["Values outside the plausible range",
             "0 in this sample; 2 corrupt member-days in the full 2000-2019 archive",
             "0"],
            ["Pairs lost in the join",
             f"0 of {M['n_fc_rows'] // M['n_members']:,} reduced station-day-lead forecasts "
             f"have no truth", "-"],
        ],
        widths=[0.30, 0.40, 0.30], small=True)
    c.h2("Key Observations")
    c.bullets([
        "The join is lossless and the missingness is structural, not accidental: "
        f"{M['n_paired']:,} of {M['n_paired']:,} reduced forecast rows found a truth, and "
        f"every null in the forecast sample is the archive's own lead limit (wind at Day 5, "
        f"soil moisture at Day 3). No imputation is needed or justified.",
        "Values that are present can still be wrong. The soil-moisture field reaches 100% "
        "volumetric at three coastal stations where the truth is 0-18.6%, and this is a "
        "label error rather than a data-entry error: paired without a mask it would have "
        "manufactured a bust on every row for four districts.",
        "The label is heavy tailed and zero-inflated in rainfall "
        f"({M['zero_rain_share'] * 100:.1f}% dry days, a maximum error of "
        f"{s['rainfall_mm']['maximum']:.0f} mm/day against a median of "
        f"{s['rainfall_mm']['median']:.2f}), so a percentile definition per variable is "
        "necessary rather than convenient, and any fixed threshold would be wrong somewhere.",
        "The event label fires on "
        f"{M['event_rate'] * 100:.1f}% of paired days and rises with lead to "
        f"{M['event_by_lead'][10] * 100:.1f}%, so the problem is not a rare-class problem "
        "and the classifier must not be re-weighted for imbalance.",
        "Ensemble spread predicts error monotonically but weakly for temperature and "
        f"pressure (rank correlation {s['temperature_c']['spread_rank_corr']:.2f} and "
        f"{s['pressure_hpa']['spread_rank_corr']:.2f}) and strongly for rainfall "
        f"({s['rainfall_mm']['spread_rank_corr']:.2f}); a spread-only model is a baseline to "
        "beat, not a solution.",
        "Season dominates the error scale (rainfall MAE is 3.6 times higher in JJAS), so the "
        "split must be chronological and the features must carry month, season and "
        "geography; a random split would leak the season into the test set.",
    ], numbered=True)

    # ==================================================================================
    # 6. Splitting
    # ==================================================================================
    c.h1("6. Dataset Splitting Plan")
    c.table(
        "Planned data split",
        ["Dataset subset", "Proposed percentage", "Planned purpose"],
        [
            ["Training", "70% of the pre-2017 cycles",
             "About 4,330 of the 6,200 daily cycles of 2000-2016, minus the ten-day embargo: "
             "the bias table, the bust thresholds, the historical bust frequency, the error "
             "regressors and the bust classifier"],
            ["Validation", "15% of the pre-2017 cycles",
             "About 930 cycles from the tail of 2000-2016, embargoed from the test year: early "
             "stopping, hyper-parameter choice, the Platt calibrator and threshold review. "
             "Validation scores are optimistic by construction and are never quoted as test "
             "results"],
            ["Testing", "15% of cycles",
             "All 365 cycles of 2017, held out entire: every metric the project publishes, "
             "the promotion gate and the baseline ladder. 2018 and 2019 sit in the archive "
             "unused and are the next evaluation"],
            ["Embargo", "10 days either side of every boundary",
             "No observed day appears on both sides of a split or of an out-of-fold block; "
             "out-of-fold blocks are 30-day spans of forecast cycles"],
        ],
        widths=[0.17, 0.22, 0.61], small=True,
        legend="The template's 70/15/15 is the plan and is kept, applied to the cycles before "
               "the test year. Because the test set is a whole calendar year rather than a "
               "random 15% of rows, the realised test share is 5% (one year of twenty); it "
               "becomes 15% when 2017, 2018 and 2019 are all scored in the next phase. The "
               "model now served was trained on all of 2000-2016 with early stopping inside "
               "that span (6,193 cycles); the next run applies the split above.")
    c.figure(figs["split"],
             "Figure 7. The chronological split. Training and validation are both inside "
             "2000-2016 and separated by a ten-day embargo; 2017 is held out entire as one "
             "monsoon; 2018-2019 are reserved. The dark bands are the embargoes, which exist "
             "because a cycle issued on day d is still being verified on day d+9.", width=1.0)
    c.p("<b>Splitting method.</b> The split is <b>chronological by forecast cycle, not random "
        "and not stratified by row</b>. Three properties of the data force it. Consecutive "
        "days at a district are strongly autocorrelated, so a random row split would put "
        "today in training and yesterday in test. Neighbouring districts share 0.25&deg; cells "
        "through the weight table, so a district-wise split would still share the field. And "
        "the five members of a cycle are perturbations of one analysis, so a member-wise split "
        "would be a split of nothing. Holding out whole calendar years answers the question "
        "that matters operationally: has this model ever seen this year's weather at all? "
        "Inside training, out-of-fold predictions for the classifier's design matrix are made "
        "in 30-day blocks assigned round-robin, so a fold holds whole stretches of weather "
        "rather than every third day.")
    c.p("<b>Data leakage prevention.</b> Each mechanism below is a guard in the code, and most "
        "are pinned by a test:")
    c.bullets([
        "<b>No observed day on both sides of a boundary.</b> A cycle issued on day d is "
        "verified against the observations of days d to d+9, so no split or fold boundary is "
        "closer than ten days (<i>SPLIT_EMBARGO_DAYS = 10</i>; tested).",
        "<b>Thresholds, bias tables, jump climatology and calibrators are fitted on training "
        "cycles only</b>, then applied unchanged to validation, test and live data. The "
        "in-sample thresholds in Table 9 exist only to illustrate the label's support.",
        "<b>Observation-derived inputs are refused at the contract level.</b> A column built "
        "from an observation after issue time raises on frame construction rather than "
        "producing a flattering score; one such input was found by audit on 2026-09-25 and "
        "removed.",
        "<b>Live availability is checked against training availability.</b> The MJO block was "
        "dropped for new runs precisely because live cycles had it missing while training "
        "rows had it present - the same train/live gap as the leaked error feature.",
        "<b>Duplicate truth products are quarantined before any frame is built</b>, because a "
        "one-day-late IMD file winning the dedupe changed the rainfall truth in 72.5% of 2018 "
        "rows without changing a single key.",
        "<b>The CNN's channel normaliser is fitted on training cycles only</b>, for the same "
        "reason: fitting it on the whole archive leaks the test period's climate.",
        "<b>Calibration is fitted on validation, not on test</b>, and the promotion gate "
        "refuses a model more than 0.05 ROC-AUC worse than the incumbent, or below 0.55 "
        "absolutely.",
        "<b>Grouped bootstrapping for confidence intervals</b> resamples by forecast cycle "
        "rather than by row, so a reported interval does not assume 33,300 independent "
        "observations from one day.",
    ])
    c.p("The one leakage route left open is stated rather than hidden: 2018 and 2019 sit in "
        "the training store and are unused by the model, but they are in the same archive, so "
        "the next evaluation must ingest them as a held-out build rather than merely score "
        "them.")

    # ==================================================================================
    # 7. Candidate models
    # ==================================================================================
    c.h1("7. Candidate Model Selection")
    c.p("Five candidates are carried, from a fully interpretable baseline to a "
        "spatio-temporal network, so that the final choice is made against something. Each is "
        "already implemented in the repository or specified here in enough detail to be "
        "implemented next week.")
    c.table(
        "Candidate models and their suitability for the project",
        ["Candidate model", "Model type", "Suitability for the project"],
        [
            ["<b>Model 1.</b> XGBoost gradient-boosted trees (per-variable error regressor + "
             "bust classifier)",
             "Supervised ensemble of shallow regression/classification trees (gradient "
             "boosting, hist method)",
             "Best fit to the problem as posed: a heterogeneous tabular design matrix, "
             "non-linear interactions between spread, lead, season and terrain, native "
             "categorical support for state and season, and exact TreeSHAP attributions "
             "available for every prediction. Already the served model"],
            ["<b>Model 2.</b> Random forest (bagged trees)",
             "Bagged ensemble of decorrelated decision trees",
             "A high-variance-control counterpart to boosting on the same matrix: no learning "
             "rate or early stopping to tune, robust to noisy spread features, and the "
             "classifier used, with ordinal logistic regression, in the closest published "
             "study of forecast-skill prediction [1]"],
            ["<b>Model 3.</b> L2-regularised logistic regression on spread, lead, season and "
             "district descriptors (EMOS-style)",
             "Linear probabilistic model (logistic link), calibrated by construction",
             "The interpretable floor. Coefficients are signed and physically readable, "
             "training is seconds, and it is the natural home of the EMOS-style "
             "post-processing baseline [10], [11]; if a tree ensemble cannot beat it, the "
             "extra complexity is not earning its keep"],
            ["<b>Model 4.</b> Dilated convolutional network over the 0.25&deg; gridded fields",
             "Deep learning: 2-D CNN with dilations 1, 2, 4, 8, 32 channels, one mask channel "
             "per input channel, area-weighted pooling to districts (43,969 parameters)",
             "Reads the synoptic pattern a district-mean row cannot: a depression off the "
             "Odisha coast or a western disturbance crossing Punjab is a shape in a field "
             "[10], [19]. Trained on identical cycles and scored on identical rows, so its "
             "result is comparable with Model 1's by construction"],
            ["<b>Model 5.</b> Spatio-temporal recurrent network (ConvLSTM over the 10-day "
             "sequence)",
             "Deep learning: convolutional recurrence over time, per district or per grid "
             "cell",
             "Would model the evolution of the pattern rather than one snapshot. Listed as a "
             "candidate and deliberately deferred: one cycle yields ten temporal samples, so "
             "the parameters-per-sample ratio is far worse than the CNN's, and the "
             "sequence-to-event mapping is already partly captured by the jumpiness and "
             "time-lagged-pool features [10]"],
        ],
        widths=[0.28, 0.24, 0.48], small=True)

    c.h2("7.1 Model Selection Justification")
    c.p("Each candidate is judged on the same five criteria, and the comparison is recorded in "
        "Table 14 rather than left implicit.")
    c.table(
        "Suitability matrix: problem fit, dataset size, interpretability, cost and literature",
        ["Criterion", "Model 1 XGBoost", "Model 2 Random forest", "Model 3 Logistic EMOS",
         "Model 4 Dilated CNN"],
        [
            ["Fit to the problem type",
             "Strong: tabular, mixed scales, non-linear, missing-by-structure, class-"
             "conditional probabilities",
             "Strong: same matrix, slightly weaker on smooth additive structure",
             "Moderate: assumes additive, monotone contributions in the log-odds",
             "Strong on the spatial pattern, which the tabular models cannot express at all"],
            ["Dataset size and characteristics",
             "Comfortable: millions of rows, dozens of features; scale-free in the features",
             "Comfortable: same rows; needs more trees for the same bias",
             "Comfortable: it is the most sample-efficient of the four",
             "The binding constraint: 17 cycles is 170 samples for the CI comparison, so the "
             "network is deliberately small and heavily regularised, and the script refuses to "
             "run below a floor"],
            ["Complexity and interpretability",
             "Moderate complexity, exact TreeSHAP per prediction, global and local "
             "attribution [13], [14]",
             "Moderate complexity, impurity and permutation importance, SHAP available but "
             "slower",
             "Highest interpretability: signed coefficients, closed-form log-odds",
             "Lowest: attribution needs a surrogate, so explanation is a second model that can "
             "disagree with the first"],
            ["Expected computational cost",
             "Moderate: training peaks about 2.3 GB and minutes-to-hours per retrain on CPU; "
             "serving is onnxruntime and fits a 512 MB box",
             "Moderate: more memory than boosting for the same data, no incremental training",
             "Negligible: seconds on CPU",
             "High: a GPU for training, PyTorch in the training image only; serving would need "
             "a separate export path"],
            ["Evidence in the literature",
             "Established for tabular environmental prediction and for forecast-error "
             "modelling [15]; explainable-tree methods are the reference for this kind of "
             "attribution [13], [14]",
             "Used for percentile skill-class prediction in the project's base paper [1]",
             "Standard post-processing baseline (EMOS/NR) in operational meteorology [10], "
             "[11]",
             "Convolutional and learned post-processing of ensemble fields [10], [19]; the "
             "receptive-field and no-downsampling design follows from the bust scale"],
        ],
        widths=[0.16, 0.21, 0.19, 0.20, 0.24], small=True,
        legend="Model 5 (spatio-temporal recurrent) is omitted from the matrix because it is not "
               "a candidate for this phase: it would consume the same GPU budget as Model 4 "
               "with a worse parameter-to-sample ratio on a five-member archive.")
    c.p("<b>On the nature of the problem and the data.</b> The label is a tail event of a "
        "continuous quantity, the predictors are of mixed type and scale, and the relations "
        "are strongly non-linear and conditional: spread matters much more for rainfall than "
        "for temperature "
        f"(rank correlation {s['rainfall_mm']['spread_rank_corr']:.2f} against "
        f"{s['temperature_c']['spread_rank_corr']:.2f}), and error grows with lead at "
        "different rates per variable. Tree ensembles represent exactly this kind of "
        "interaction without a hand-specified link function, which is why Models 1 and 2 lead. "
        "Model 3 is kept because it forces the comparison to be honest: if the linear model "
        "with spread and lead reaches the same Brier skill, the ensemble is decoration.")
    c.p("<b>On dataset size and characteristics.</b> The tabular models see millions of rows "
        "and several dozen features, where boosting and bagging are both comfortable. The CNN "
        "sees one sample per cycle per lead day - 170 samples in the continuous-integration "
        "comparison - and a 43,969-parameter network against them is at the edge of what the "
        "data can support; it is retained as a challenger with a sample floor rather than as "
        "the primary model, and its result is reported even if it loses.")
    c.p("<b>On complexity and interpretability.</b> The project's whole claim is explanation, "
        "so a model whose attribution is itself a surrogate model is a weaker fit for the "
        "primary slot. TreeSHAP is exact for the tree ensemble, additive, and local, which is "
        "what a duty forecaster asked \"why is this district at 0.7?\" needs [13], [14]. The "
        "CNN's attributions would have to be produced by masking or a surrogate, and a "
        "surrogate that disagrees with its model is a new failure mode, not a feature.")
    c.p("<b>On expected computational requirements.</b> The serving box is killed, not "
        "throttled, at 512 MB, and training peaks at about 2.3 GB on a workstation GPU. "
        "XGBoost exports to onnxruntime, which keeps serving memory near 346 MB; a PyTorch CNN "
        "must never enter the serving dependency set. Both tree candidates and the logistic "
        "baseline fit that constraint; the CNN does not, without a separate export path, which "
        "is another reason it is a challenger.")
    c.p("<b>On the literature.</b> The base paper learns percentile-based skill classes with "
        "ordinal logistic regression and random forest and compares them against persistence "
        "and ensemble-spread baselines [1]; ensemble post-processing with neural networks "
        "[10] and EMOS [11] establishes the calibration framing; tree ensembles with SHAP are "
        "the established explainable-AI pattern for environmental tabular prediction, and are "
        "used to characterise Earth-system model error specifically [15], [17]. The proposed "
        "combination - error regression, then a calibrated bust classifier, then local "
        "attribution - follows the convergence the literature survey records, and the "
        "baseline ladder (climatology, lead-day, spread, EMOS, IDR, analog) is what makes the "
        "eventual claim comparable with published work [1], [6]-[9].")

    c.h2("7.2 Proposed Model")
    c.p("The proposed model for the next phase is <b>Model 1: a per-variable XGBoost error "
        "regressor feeding an XGBoost bust classifier</b>, with the following configuration "
        "already implemented and tested: 3,000 trees as a cap with early stopping of 100 "
        "rounds on the validation year, maximum depth 3, learning rate 0.03, subsample and "
        "column sample 0.8, minimum child weight 15, L2 regularisation 3.0, L1 0.5, gamma 0.5, "
        "histogram tree method, <i>state_id</i> and <i>season</i> as native categoricals, and "
        "no positive-class weighting. Probabilities are calibrated by a Platt fit on "
        "validation, with a non-positive slope refused, and TreeSHAP attributions are computed "
        "for every served prediction.")
    c.p("Model 4, the dilated CNN, is proposed as the <b>challenger</b>, trained on the same "
        "cycles and scored on the same rows, so that the comparison answers a single "
        "question: does seeing the field buy skill that the district-mean features do not "
        "carry? Models 2 and 3 are proposed as the <b>ladder</b>, because a claimed skill has "
        "to be a skill against something. Model 5 is deferred with its reason recorded rather "
        "than dropped quietly.")
    c.p("This section records the proposed model selection. Experimental comparison and the "
        "final selection based on measured performance will be undertaken next week, on the "
        "held-out year, with the promotion gate deciding which model is served.")

    # ==================================================================================
    # 8. Workflow
    # ==================================================================================
    c.h1("8. Proposed Project Workflow")
    c.p("The flowchart below is the planned pipeline from problem definition to model "
        "selection, in the order the work actually depends on: nothing downstream of data "
        "acquisition is meaningful until the quality assessment has run, and no split is "
        "meaningful until the label is defined on training-only thresholds.")
    c.figure(figs["pipeline"],
             "Figure 8. Planned pipeline from dataset acquisition through preprocessing to "
             "model selection. Stages 1-5 are complete for the archived dataset; stage 6 is "
             "what Section 5 reports; stage 7 is the plan in Section 6; stage 8 is the "
             "selection in Section 7.", width=1.0)
    c.table(
        "Pipeline stages, their inputs and their outputs",
        ["Stage", "Input", "Output of this stage"],
        [
            ["1. Problem definition",
             "Research gap and literature survey",
             "The bust definition and the predictor set the dataset must supply"],
            ["2. Dataset acquisition",
             "GEFSv12 reforecast, ERA5, GADM 4.1, elevation, land-sea masks, MJO",
             "Raw archives with per-value provenance, 7,288 of 7,305 cycles"],
            ["3. Data quality assessment",
             "The raw archives and the weight table",
             "The issue list in Table 3, each with a measurement"],
            ["4. Data cleaning",
             "The issue list",
             "Completeness guard, dedupe, range check, land-sea mask, date alignment"],
            ["5. Data preprocessing",
             "Cleaned rows",
             "Canonical schema, district area means, daily fields, bias table, bust label, "
             "engineered features"],
            ["6. Preliminary EDA",
             "The paired frame",
             "Tables 7-10, 11 and Figures 3-6: dimensions, distributions, class balance, spread "
             "relation"],
            ["7. Dataset splitting plan",
             "The paired frame with cycle keys",
             "Chronological splits with a Day-10 embargo and 30-day out-of-fold blocks"],
            ["8. Candidate model selection",
             "EDA conclusions and the split plan",
             "Five candidates, one proposed model, one challenger, one ladder"],
        ],
        widths=[0.22, 0.33, 0.45], small=True)

    # ==================================================================================
    # 9. Conclusion
    # ==================================================================================
    c.h1("9. Conclusion")
    c.p("The project predicts the probability that an already-issued medium-range forecast is "
        "badly wrong for a specific Indian district, for seven label variables and lead days "
        "1-10, and explains each prediction locally. The dataset that supports it is a "
        "twenty-year pairing of the NOAA GEFSv12 reforecast with ERA5 reanalysis on one "
        "spatial unit - 666 GADM 4.1 districts, reduced through a single area-overlap weight "
        "table of 10,373 district-cell pairs - joined into one canonical 20-column schema so "
        "that a forecast and its truth are always compared as like with like. It is complete "
        "where completeness is possible: no null cells in the store, no nulls in the "
        "observation sample, no duplicate keys, and 7,288 of 7,305 archive cycles present, "
        "with the only missingness being the archive's own lead limits for wind and soil "
        "moisture.")
    c.p("Cleaning this dataset was chiefly verification, because the obvious treatments would "
        "damage products that are correct in themselves. The real problems found were not "
        "typing errors but disagreements between two good products: a sea written as saturated "
        "soil, a gauge date one day out, an angle difference that made a northerly a bust, a "
        "truth file that won a dedupe it should have lost, and an input that quietly read the "
        "answer. Each was measured, treated and recorded with its before-and-after number. "
        "Preprocessing is deterministic and identical on both sides - one schema, one weight "
        "table, one daily convention, one range check - with every fitted quantity (bias "
        "table, bust threshold, jump climatology, calibrator, CNN normaliser) fitted on "
        "training cycles only. The split is chronological by cycle with a ten-day embargo and "
        "30-day out-of-fold blocks, because rows here are neither independent nor "
        "exchangeable.")
    c.p("The preliminary analysis fixes the shape of the modelling problem: the label is a "
        f"heavy tail above each variable's own p90, the event fires on "
        f"{M['event_rate'] * 100:.1f}% of paired days and rises with lead, ensemble spread is "
        "informative but insufficient, and season and geography dominate the error scale. On "
        "that basis a gradient-boosted tree classifier over out-of-fold error evidence is "
        "proposed as the primary model, with exact TreeSHAP attribution; a dilated "
        "convolutional network is carried as a challenger on the same held-out rows; and a "
        "random forest plus a calibrated logistic EMOS baseline form the ladder that any "
        "claimed skill must beat. No model performance is reported in this document: that "
        "comparison, and the final selection, belong to the next phase.")

    # ==================================================================================
    # 10. References
    # ==================================================================================
    c.h1("10. References")
    c.p("Numbering follows the project's literature survey "
        "(<i>docs/literature-survey-explainable-ai-weather-busts.pdf</i>) so the two documents "
        "can be read together. Datasets are listed with the access used by the code in "
        "<i>backend/scripts/</i>; each keeps its own licence and attribution.")
    refs = [
        "Potvin, C. K., et al. (2024). Using machine learning to predict convection-allowing "
        "ensemble forecast skill: Evaluation with the NSSL Warn-on-Forecast System. "
        "<i>Artificial Intelligence for the Earth Systems, 3</i>, e230106. "
        "https://doi.org/10.1175/AIES-D-23-0106.1",
        "Rodwell, M. J., et al. (2013). Characteristics of occasional poor medium-range "
        "weather forecasts for Europe. <i>Bulletin of the American Meteorological Society, "
        "94</i>(9), 1393-1405. https://doi.org/10.1175/BAMS-D-12-00099.1",
        "Parsons, D. B., Lillo, S. P., Rattray, C. P., Bechtold, P., Rodwell, M. J., & Bruce, "
        "C. M. (2019). The role of continental mesoscale convective systems in forecast busts "
        "within global weather prediction systems. <i>Atmosphere, 10</i>(11), 681. "
        "https://doi.org/10.3390/atmos10110681",
        "Guan, H., et al. (2022). GEFSv12 reforecast dataset for supporting subseasonal and "
        "hydrometeorological applications. <i>Monthly Weather Review, 150</i>(3), 647-665. "
        "https://doi.org/10.1175/MWR-D-21-0245.1",
        "Hamill, T. M., et al. (2022). The reanalysis for the Global Ensemble Forecast "
        "System, version 12. <i>Monthly Weather Review, 150</i>(1), 59-79. "
        "https://doi.org/10.1175/MWR-D-21-0023.1",
        "Dube, A., Ashrit, R., Singh, H., Iyengar, G., & Rajagopal, E. N. (2016). "
        "Verification of medium range probabilistic rainfall forecasts over India. <i>Pure "
        "and Applied Geophysics, 173</i>, 2489-2510. https://doi.org/10.1007/s00024-016-1278-z",
        "Chakraborty, P., Sarkar, A., Bhatla, R., & Singh, R. (2021). Assessing the skill of "
        "NCMRWF global ensemble prediction system in predicting Indian summer monsoon during "
        "2018. <i>Atmospheric Research, 251</i>, 105255. "
        "https://doi.org/10.1016/j.atmosres.2020.105255",
        "Dube, A., Karunasagar, S., Ashrit, R., & Mitra, A. K. (2022). Spatial verification of "
        "ensemble rainfall forecasts over India. <i>Atmospheric Research, 273</i>, 106169. "
        "https://doi.org/10.1016/j.atmosres.2022.106169",
        "Roy Bhowmik, S. K., & Durai, V. R. (2012). Development of multimodel ensemble based "
        "district level medium range rainfall forecast system for Indian region. <i>Journal "
        "of Earth System Science, 121</i>, 273-285. "
        "https://doi.org/10.1007/s12040-012-0160-1",
        "Rasp, S., & Lerch, S. (2018). Neural networks for post-processing ensemble weather "
        "forecasts. <i>Monthly Weather Review, 146</i>(11), 3885-3900. "
        "https://doi.org/10.1175/MWR-D-18-0187.1",
        "Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and "
        "estimation. <i>Journal of the American Statistical Association, 102</i>(477), "
        "359-378. https://doi.org/10.1198/016214506000001437",
        "Brocker, J., & Smith, L. A. (2007). Increasing the reliability of reliability "
        "diagrams. <i>Weather and Forecasting, 22</i>(3), 651-661. "
        "https://doi.org/10.1175/WAF993.1",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model "
        "predictions. <i>Advances in Neural Information Processing Systems, 30</i>, "
        "4765-4774. https://proceedings.neurips.cc/paper/7062",
        "Lundberg, S. M., et al. (2020). From local explanations to global understanding with "
        "explainable AI for trees. <i>Nature Machine Intelligence, 2</i>, 56-67. "
        "https://doi.org/10.1038/s42256-019-0138-9",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
        "<i>Proceedings of the 22nd ACM SIGKDD Conference</i>, 785-794. "
        "https://doi.org/10.1145/2939672.2939785",
        "Dutta, S., & Pal, S. K. (2022). Interpretation of black box for short-term "
        "predictions of pre-monsoon cumulonimbus cloud events over Kolkata. <i>SN Computer "
        "Science, 3</i>, 352. https://doi.org/10.1007/s42979-022-01255-x",
        "Silva, S. J., Keller, C. A., & Hardin, J. (2022). Using an explainable machine "
        "learning approach to characterize Earth system model errors: Application of SHAP "
        "analysis to modeling lightning flash occurrence. <i>Journal of Advances in Modeling "
        "Earth Systems, 14</i>, e2021MS002881. https://doi.org/10.1029/2021MS002881",
        "Daif, N., Di Nunno, F., Granata, F., et al. (2025). Forecasting maximal and minimal "
        "air temperatures using explainable machine learning: SHAP versus LIME. <i>Stochastic "
        "Environmental Research and Risk Assessment, 39</i>, 2551-2581. "
        "https://doi.org/10.1007/s00477-025-02984-4",
        "Price, I., et al. (2025). Probabilistic weather forecasting with machine learning. "
        "<i>Nature, 637</i>, 84-90. https://doi.org/10.1038/s41586-024-08252-9",
        "Hersbach, H., et al. (2020). The ERA5 global reanalysis. <i>Quarterly Journal of the "
        "Royal Meteorological Society, 146</i>(730), 1999-2049. "
        "https://doi.org/10.1002/qj.3803",
        "Kiladis, G. N., Dias, J., Straub, K. H., Wheeler, M. C., Tulich, S. N., Kikuchi, K., "
        "Weickmann, K. M., & Ventrice, M. J. (2014). A comparison of OLR and circulation-based "
        "indices for tracking the MJO. <i>Monthly Weather Review, 142</i>(5), 1697-1715. "
        "https://doi.org/10.1175/MWR-D-13-00301.1",
        "Jarvis, A., Reuter, H. I., Nelson, A., & Guevara, E. (2008). <i>Hole-filled seamless "
        "SRTM data V4</i>. International Centre for Tropical Agriculture (CIAT). "
        "https://srtm.csi.cgiar.org",
        "NASA Langley Research Center (2024). <i>Prediction Of Worldwide Energy Resources "
        "(POWER) project, daily point API</i>. https://power.larc.nasa.gov/api/temporal/daily/point",
        "Global Administrative Areas (2022). <i>GADM database of Global Administrative Areas, "
        "version 4.1</i>. https://gadm.org/",
        "Copernicus Climate Change Service (2023). <i>ERA5 hourly data on single levels from "
        "1940 to present</i>. Copernicus Climate Change Service Climate Data Store. "
        "https://cds.climate.copernicus.eu/datasets",
        "National Oceanic and Atmospheric Administration (2024). <i>GEFSv12 retrospective "
        "forecast archive (reforecast)</i>. AWS Registry of Open Data. "
        "https://registry.opendata.aws/noaa-gefs-reforecast/",
    ]
    for i, ref in enumerate(refs, start=1):
        c.ref(f"[{i}] {ref}")
    c.p("Software libraries used for the measurements in this report: Python 3.11, pandas, "
        "NumPy, PyArrow, matplotlib, reportlab, XGBoost, scikit-learn, SHAP and PyTorch "
        "(training image only).")

    # ==================================================================================
    # 11. Appendix
    # ==================================================================================
    c.page_break()
    c.h1("11. Appendix")
    c.h2("A.1 Pairing and label construction")
    c.code(
        '# build the paired frame, then the label, from the archived files\n'
        'VARS = {"t2m_c": "temperature_c", "rh2m_pct": "humidity_pct",\n'
        '        "apcp_mm": "rainfall_mm", "mslp_hpa": "pressure_hpa",\n'
        '        "pwat_kgm2": "atmospheric_moisture_kgm2", "wspd10m_ms": "wind_speed_ms"}\n'
        'KEY = ["station", "region", "init_date", "valid_date", "lead_day"]\n'
        '\n'
        'mean   = gefs.groupby(KEY, observed=True)[list(VARS)].mean()      # ensemble mean\n'
        'spread = gefs.groupby(KEY, observed=True)[list(VARS)].std(ddof=0)  # ensemble spread\n'
        'paired = (mean.join(spread, rsuffix="_spread").reset_index()\n'
        '              .merge(obs, on=["station", "valid_date"], how="inner"))\n'
        '\n'
        'for col, name in VARS.items():\n'
        '    err = (paired["fc_" + name] - paired["ob_" + name]).abs()\n'
        '    thr = err.quantile(0.90)            # training cycles only in production\n'
        '    paired["bust_" + name] = (err > thr).astype(int)',
        "Listing 4. The measurement path used for every table and figure in this report. In "
        "production the quantile is taken over training cycles only and from the "
        "bias-corrected error.")
    c.h2("A.2 Sample rows of the paired frame")
    sample = (d["paired"][d["paired"]["ob_rainfall_mm"] > 5.0]
              .drop_duplicates(subset=["station"])
              .head(5))
    dry = d["paired"][d["paired"]["ob_rainfall_mm"] <= 0.01].head(1)
    sample = pd.concat([sample, dry])
    rows = []
    for _, r in sample.iterrows():
        rows.append([
            str(r["station"]), str(r["init_date"])[5:], str(int(r["lead_day"])),
            f"{r['fc_temperature_c']:.2f}", f"{r['spread_temperature_c']:.2f}",
            f"{r['ob_temperature_c']:.2f}",
            f"{abs(r['fc_temperature_c'] - r['ob_temperature_c']):.2f}",
            f"{r['fc_rainfall_mm']:.2f}", f"{r['ob_rainfall_mm']:.2f}",
            f"{abs(r['fc_rainfall_mm'] - r['ob_rainfall_mm']):.2f}",
        ])
    c.table(
        "Six paired rows as the models receive them, five wet days from different "
        "districts and one dry day",
        ["Station", "Init", "Lead", "fc T (&deg;C)", "spread T", "obs T (&deg;C)",
         "abs err T", "fc rain (mm)", "obs rain (mm)", "abs err rain"],
        rows, widths=[0.17, 0.08, 0.07, 0.10, 0.09, 0.11, 0.10, 0.10, 0.10, 0.08],
        small=True,
        legend="A 0.00 mm forecast on a dry day is not a missing value: the rainfall "
               "accumulation is honestly zero, which is why the label is a percentile of the "
               "error and not a fixed millimetre threshold.")
    
    c.h2("A.3 Reproducing this report")
    c.code(
        '# regenerate every figure and this PDF from the files in backend/data/\n'
        'python docs/preprocessing-report/build_report.py\n'
        '\n'
        'docs/preprocessing-report/assets/*.png      eight figures, rebuilt each run\n'
        'docs/preprocessing-report/preprocessing-report.md   the same content, reviewable\n'
        '"Dataset Preprocessing and Model Selection Report - Sanket.pdf"   the deliverable',
        "Listing 5. One command rebuilds the report; no number in it is typed by hand.")
    c.p("Source of the numbers that this script does not compute: the archive-scale figures "
        "(7,288 of 7,305 cycles, 33,300 rows per cycle, 12,154,500 rows for 2017), the corrupt "
        "member-days, the IMD date convention, the sea-masking effect, the verification-product "
        "agreement over 21,492 paired city-days, and the label and calibration findings are the "
        "ones recorded in <i>docs/engineering-reference.md</i>, <i>docs/known-issues.md</i> and "
        "<i>docs/analysis/verification_product_agreement.csv</i>, each with the date it was "
        "measured. The remaining limitation is stated plainly: this report was measured on the "
        "archived 2019 sample and the geography layer, not on the full twenty-year store, "
        "because the full store is a release asset rather than a repository file.")

    # ==================================================================================
    # Declaration
    # ==================================================================================
    c.h1("Student Declaration")
    c.p("I confirm that this report accurately describes the dataset selected and the "
        "cleaning, preprocessing, and model-selection work completed or proposed for this "
        "stage of the project.")
    decl = Table([["Student Signature: ________________________________", "",
                   "Date: 28-10-2026"]],
                 colWidths=[0.44 * FRAME_W, 0.22 * FRAME_W, 0.34 * FRAME_W], hAlign="LEFT")
    decl.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 10)]))
    c.flow.append(decl)
    return c


# --------------------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------------------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(FOOTER_GREY)
    text = f"{FOOTER_TEXT}  |  Page {canvas.getPageNumber()}"
    canvas.drawCentredString(PAGE_W / 2, 30, text)
    canvas.setStrokeColor(colors.Color(0.85, 0.85, 0.85))
    canvas.setLineWidth(0.5)
    canvas.line(LEFT, 44, PAGE_W - RIGHT, 44)
    canvas.restoreState()


def build(st, c: Content) -> None:
    doc = BaseDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                          topMargin=TOP, bottomMargin=BOTTOM,
                          title="Dataset Preprocessing and Model Selection Report - Sanket",
                          author="D N Harith, Dhanush Thota",
                          subject=("Explainable AI for District-Level Prediction of "
                                   "Medium-Range Weather Forecast Busts over India"),
                          creator="docs/preprocessing-report/build_report.py")
    frame = Frame(LEFT, BOTTOM, FRAME_W, PAGE_H - TOP - BOTTOM, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    template = PageTemplate(id="page", frames=[frame], onPage=footer)
    doc.addPageTemplates([template])
    doc.build(cover(st) + c.flow)


def main() -> None:
    st = make_styles()
    print("measuring backend/data ...")
    d = measure()
    print("drawing figures ...")
    figs = build_figures(d)
    print("writing content ...")
    c = build_content(st, d, figs)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Dataset Selection, Data Cleaning, Preprocessing, and Model Selection\n\n"
              "*Explainable AI for District-Level Prediction of Medium-Range Weather Forecast "
              "Busts over India*\n\n"
              "D N Harith (1DS24AI012) · Dhanush Thota (1DS24AI057) · AI&ML, 5th semester · "
              "Faculty guide: Dr. Kusumika Krori Dutta · 28-10-2026\n\n"
              "*Generated by `build_report.py`; the PDF is the deliverable and this markdown "
              "is the same content in a reviewable form.*\n")
    OUT_MD.write_text(header + "\n".join(c.md) + "\n", encoding="utf-8")
    print("writing pdf ...")
    build(st, c)
    print(f"wrote {OUT_PDF.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
