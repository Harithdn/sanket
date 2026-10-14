"""Build the Sanket dataset report PDF in the format of ``Reference dataset report.pdf``.

The reference report is a seven-page assignment report (cover page + six content pages)
written for a different project. This script reproduces its layout, typography and section
heads, replaces the cover, and writes the content for the Sanket dataset from what is
actually on disk in ``backend/data/`` and what the project documents record as measured.

What it does
------------
1.  Rebuilds the cover page by editing the reference PDF in place: title, both names and
    both USNs. Everything else on the cover (college, department, logos, guide, university
    block) is left untouched.
2.  Renders the figures in ``assets/`` from the real data files with matplotlib.
3.  Renders the six content pages with reportlab, using metrics copied from the reference:
    A4, 2.5 cm side margins, 11 pt body at 16 pt leading, 14 pt / 12.5 pt bold headings,
    bold 10 pt centred table captions, 0.5 pt rules, 90% grey header rows, continuous page
    numbers starting at 2 (the cover is page 1).
4.  Merges cover + content, applies the cover text replacements to the merged document and
    writes ``Dataset report - Sanket.pdf``.

Requirements
------------
Liberation Serif (or its metric-compatible twin, Tinos) must be installed - the reference
report is set in Liberation Serif and the cover edit re-uses the font already embedded in
it. The script looks in ``~/.local/share/fonts``, ``/usr/share/fonts/truetype/liberation``
and ``/usr/share/fonts/truetype/tinos``. Needs pymupdf, reportlab, matplotlib, pandas,
pyarrow and pillow.

Usage
-----
    python docs/dataset-report/build_report.py

The numbers baked into the prose are the ones the repository's own documents record, each
with the date it was measured where the document records one. A number that changes on
every run (metrics of the served model) is deliberately not quoted here; the live figures
are read from the deployed model by the site's About page.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REFERENCE = REPO / "Reference dataset report.pdf"
ASSETS = HERE / "assets"
OUT_PDF = REPO / "Dataset report - Sanket.pdf"
OUT_MD = HERE / "dataset-report.md"

# --------------------------------------------------------------------------------------
# Typography, copied from the reference PDF's own font metrics
# --------------------------------------------------------------------------------------
BODY_SIZE, BODY_LEADING = 11.0, 16.0
H1_SIZE, H2_SIZE = 14.0, 12.5
CAPTION_SIZE = 10.0
TABLE_SIZE, TABLE_LEADING = 9.5, 18.0
SMALL_SIZE, SMALL_LEADING = 8.0, 10.0
MARGIN = 70.9          # 2.5 cm
FRAME_W = A4[0] - 2 * MARGIN   # 453.5 pt of text width

SERIF_DIRS = [
    Path.home() / ".local/share/fonts",
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/tinos"),
]


def _font_file(stem: str) -> Path | None:
    """Locate Tinos (metric-compatible with Liberation Serif, which the reference uses)."""
    for d in SERIF_DIRS:
        for name in (f"{stem}.ttf", f"LiberationSerif-{stem}.ttf", f"Tinos-{stem}.ttf"):
            p = d / name
            if p.exists():
                return p
    return None


def register_fonts() -> dict[str, str]:
    names = {"regular": "Serif", "bold": "Serif-Bold",
             "italic": "Serif-Italic", "bolditalic": "Serif-BoldItalic"}
    mapping = {"regular": "Regular", "bold": "Bold", "italic": "Italic",
               "bolditalic": "BoldItalic"}
    found: dict[str, str] = {}
    for key, stem in mapping.items():
        p = _font_file(stem)
        if p is None:
            raise SystemExit(
                f"missing serif font {stem}.ttf - install Liberation Serif, or Tinos into "
                f"~/.local/share/fonts (the reference report is set in Liberation Serif)"
            )
        pdfmetrics.registerFont(TTFont(names[key], str(p)))
        found[key] = names[key]
    pdfmetrics.registerFontFamily("Serif", normal=names["regular"], bold=names["bold"],
                                  italic=names["italic"], boldItalic=names["bolditalic"])
    return found


# --------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------
def make_styles(f: dict[str, str]):
    body = ParagraphStyle("body", fontName=f["regular"], fontSize=BODY_SIZE,
                          leading=BODY_LEADING, alignment=TA_LEFT, spaceAfter=7,
                          spaceBefore=0)
    h1 = ParagraphStyle("h1", parent=body, fontName=f["bold"], fontSize=H1_SIZE,
                        leading=18, spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=body, fontName=f["bold"], fontSize=H2_SIZE,
                        leading=16, spaceBefore=12, spaceAfter=4)
    caption = ParagraphStyle("caption", parent=body, fontName=f["bold"],
                             fontSize=CAPTION_SIZE, leading=12, alignment=TA_CENTER,
                             spaceBefore=8, spaceAfter=3)
    figcaption = ParagraphStyle("figcaption", parent=body, fontName=f["italic"],
                                fontSize=CAPTION_SIZE, leading=12.5, alignment=TA_CENTER,
                                spaceBefore=6, spaceAfter=8)
    cell = ParagraphStyle("cell", parent=body, fontSize=TABLE_SIZE, leading=11.5)
    cell_b = ParagraphStyle("cell_b", parent=cell, fontName=f["bold"])
    cell_s = ParagraphStyle("cell_s", parent=cell, fontSize=SMALL_SIZE, leading=SMALL_LEADING)
    cell_sb = ParagraphStyle("cell_sb", parent=cell_s, fontName=f["bold"])
    legend = ParagraphStyle("legend", parent=cell_s, alignment=TA_LEFT, spaceBefore=4)
    ref = ParagraphStyle("ref", parent=body, fontSize=9.5, leading=13, leftIndent=14,
                         firstLineIndent=-14, spaceAfter=5, alignment=TA_JUSTIFY)
    return dict(body=body, h1=h1, h2=h2, caption=caption, figcaption=figcaption,
                cell=cell, cell_b=cell_b, cell_s=cell_s, cell_sb=cell_sb, legend=legend,
                ref=ref)


class Content:
    """Accumulates flowables and mirrors them into markdown for review."""

    def __init__(self, styles):
        self.s = styles
        self.flow = []
        self.md: list[str] = []
        self.table_n = 0
        self.figure_n = 0

    # -- text ---------------------------------------------------------------------------
    def h1(self, text: str):
        self.flow.append(Paragraph(text, self.s["h1"]))
        self.md.append(f"\n## {text}\n")

    def h2(self, text: str):
        self.flow.append(Paragraph(text, self.s["h2"]))
        self.md.append(f"\n### {text}\n")

    def p(self, text: str):
        self.flow.append(Paragraph(text, self.s["body"]))
        self.md.append(_md(text))

    def caption(self, text: str):
        self.flow.append(Paragraph(text, self.s["caption"]))
        self.md.append(f"\n**{_md(text)}**\n")

    # -- tables -------------------------------------------------------------------------
    def table(self, title: str, header: list[str], rows: list[list[str]],
              widths: list[float], small: bool = False, legend: str | None = None,
              padding: float = 4.0):
        self.table_n += 1
        self.caption(f"Table {self.table_n}. {title}")
        style_cell = self.s["cell_s"] if small else self.s["cell"]
        style_head = self.s["cell_sb"] if small else self.s["cell_b"]
        head = [Paragraph(h, style_head) for h in header]
        data = [[Paragraph(c, style_cell) for c in r] for r in rows]
        col = [w * FRAME_W for w in widths]
        t = Table([head] + data, colWidths=col, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.902, 0.902, 0.902)),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), padding),
            ("RIGHTPADDING", (0, 0), (-1, -1), padding),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        self.flow.append(t)
        if legend:
            self.flow.append(Paragraph(legend, self.s["legend"]))
        self.md.append("")
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "---|" * len(header))
        for r in rows:
            self.md.append("| " + " | ".join(_md(c) for c in r) + " |")
        if legend:
            self.md.append("")
            self.md.append(_md(legend))

    # -- figures ------------------------------------------------------------------------
    def figure(self, path: Path, caption: str, width: float):
        self.figure_n += 1
        self.flow.append(KeepTogether([
            Spacer(1, 6),
            Image(str(path), width=width * FRAME_W, height=_aspect(path) * width * FRAME_W,
                  hAlign="CENTER"),
            Paragraph(caption, self.s["figcaption"]),
        ]))
        self.md.append(f"\n![{_md(caption)}](assets/{path.name})\n")
        self.md.append(f"*{_md(caption)}*")

    def ref(self, text: str):
        self.flow.append(Paragraph(text, self.s["ref"]))
        self.md.append(_md(text))


def _aspect(path: Path) -> float:
    from PIL import Image as PILImage  # reportlab dependency, always present
    with PILImage.open(path) as im:
        return im.height / im.width


_MD = {"<b>": "**", "</b>": "**", "<i>": "*", "</i>": "*", "<sub>": "", "</sub>": "",
       "<super>": "^", "</super>": "", "<br/>": " ", "<font face=\"Serif\">": "`",
       "</font>": "`"}
_MD_ENTITIES = {"&mdash;": "—", "&ndash;": "–", "&rsquo;": "’", "&lsquo;": "‘",
                "&ldquo;": "“", "&rdquo;": "”", "&deg;": "°", "&sup2;": "²",
                "&minus;": "−", "&times;": "×", "&asymp;": "≈", "&rarr;": "→",
                "&nbsp;": " ", "&amp;": "&", "&gt;": ">", "&lt;": "<"}


def _md(text: str) -> str:
    """The same paragraph, as markdown: tags to emphasis, entities to their characters."""
    for k, v in _MD.items():
        text = text.replace(k, v)
    for k, v in _MD_ENTITIES.items():
        text = text.replace(k, v)
    return text


# --------------------------------------------------------------------------------------
# Cover page
# --------------------------------------------------------------------------------------
def build_cover(tmp: Path) -> None:
    """Replace the title, the two names and the two USNs on the reference cover page."""
    doc = pymupdf.open(REFERENCE)
    page = doc[0]
    to_change = {
        "title": [],
        "names": [],
    }
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t in ("Synthetic Lane-Image Dataset for", "Adversarial Robustness in",
                         "Autonomous Vehicles"):
                    to_change["title"].append(span)
                elif t in ("Pratham K", "Spoorthi", "1DS24AI037", "1DS24AI051"):
                    to_change["names"].append(span)

    title_lines = ["Explainable AI for District-Level",
                   "Prediction of Medium-Range Weather",
                   "Forecast Busts over India"]
    if len(to_change["title"]) != 3 or len(to_change["names"]) != 4:
        raise SystemExit(f"cover layout not as expected: "
                         f"{len(to_change['title'])} title, {len(to_change['names'])} name spans")

    # 1. erase the replaced text: redaction removes text *and* the bytes behind it
    for span in to_change["title"] + to_change["names"]:
        page.add_redact_annot(pymupdf.Rect(span["bbox"]) + (-2, -3, 2, 3))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)

    # 2. write the new text, centred on the same baselines, in the reference's own font
    bold = _font_file("Bold")
    measure = pymupdf.Font(fontfile=str(bold))
    for span, text in zip(to_change["title"], title_lines):
        size = span["size"]
        w = measure.text_length(text, fontsize=size)
        x = (page.rect.width - w) / 2
        y = span["origin"][1]
        page.insert_text((x, y), text, fontname="tiro", fontfile=str(bold),
                         fontsize=size, color=(0, 0, 0))

    by_text = {s["text"].strip(): s for s in to_change["names"]}
    replacements = {
        "Pratham K": "D N Harith",
        "1DS24AI037": "1DS24AI012",
        "Spoorthi": "Dhanush Thota",
        "1DS24AI051": "1DS24AI057",
    }
    for old, new in replacements.items():
        span = by_text[old]
        size = span["size"]
        if old.startswith("1DS"):
            w = measure.text_length(new, fontsize=size)
            x = span["bbox"][2] - w          # right-align the USNs where they end today
        else:
            x = span["bbox"][0]              # keep the name column where it starts
        y = span["origin"][1]
        page.insert_text((x, y), new, fontname="tiro", fontfile=str(bold),
                         fontsize=size, color=(0, 0, 0))

    doc.select([0])
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()


# --------------------------------------------------------------------------------------
# Figures - drawn from the real files in backend/data
# --------------------------------------------------------------------------------------
def build_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import font_manager, rcParams

    for d in SERIF_DIRS:
        if d.exists():
            for ttf in d.glob("*.ttf"):
                try:
                    font_manager.fontManager.addfont(str(ttf))
                except Exception:            # noqa: BLE001 - a bad font must not fail the build
                    pass
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Tinos", "Liberation Serif", "DejaVu Serif"]
    rcParams["font.size"] = 8
    ASSETS.mkdir(parents=True, exist_ok=True)

    data = REPO / "backend" / "data"
    g = pd.read_parquet(data / "samples/gefs_reforecast_india_2019.parquet")
    o = pd.read_parquet(data / "samples/era5_observations_india_2019.parquet")

    varmap = {"t2m_c": "temperature_c", "rh2m_pct": "humidity_pct",
              "apcp_mm": "rainfall_mm", "mslp_hpa": "pressure_hpa",
              "pwat_kgm2": "atmospheric_moisture_kgm2", "wspd10m_ms": "wind_speed_ms"}
    fc = (g.groupby(["city", "region", "init_date", "valid_date", "lead_day"], observed=True)
            [list(varmap)].mean().reset_index()
            .rename(columns={k: "fc_" + v for k, v in varmap.items()}))
    ob = (o[["city", "date", "t2m_c", "rh2m_pct", "precip_mm", "mslp_hpa", "pwat_kgm2",
             "wspd10m_ms"]]
          .rename(columns={"t2m_c": "ob_temperature_c", "rh2m_pct": "ob_humidity_pct",
                           "precip_mm": "ob_rainfall_mm", "mslp_hpa": "ob_pressure_hpa",
                           "pwat_kgm2": "ob_atmospheric_moisture_kgm2",
                           "wspd10m_ms": "ob_wind_speed_ms"}))
    ob["date"] = pd.to_datetime(ob["date"]).dt.date
    p = fc.merge(ob, left_on=["city", "valid_date"], right_on=["city", "date"], how="inner")

    units = {"temperature_c": "°C", "humidity_pct": "%RH", "rainfall_mm": "mm/day",
             "pressure_hpa": "hPa", "atmospheric_moisture_kgm2": "kg/m²",
             "wind_speed_ms": "m/s"}
    disp = {"temperature_c": "2 m temperature", "humidity_pct": "2 m rel. humidity",
            "rainfall_mm": "daily rainfall", "pressure_hpa": "MSL pressure",
            "atmospheric_moisture_kgm2": "column water vapour", "wind_speed_ms": "10 m wind speed"}

    # ---- Figure 1: error distributions
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.0))
    for ax, v in zip(axes.ravel(), units):
        e = (p["fc_" + v] - p["ob_" + v]).abs().dropna()
        thr, hi = e.quantile(0.90), np.quantile(e, 0.995)
        ax.hist(np.clip(e, 0, hi), bins=45, color="#3f7fb5", edgecolor="#264f6f", linewidth=0.2)
        ax.axvline(thr, color="#c0392b", linewidth=1.0, linestyle="--")
        ax.set_title(f"{disp[v]}\n$p_{{90}}$ = {thr:.2f} {units[v]}", pad=3, fontsize=8)
        ax.set_xlabel(f"|error| ({units[v]})", fontsize=7.5)
        ax.set_ylabel("event-days" if v in ("temperature_c", "pressure_hpa") else "", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", color="0.9", linewidth=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig1_error_distributions.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: coverage and sampling
    fig = plt.figure(figsize=(6.9, 2.9))
    ax1 = fig.add_axes([0.115, 0.24, 0.40, 0.62])
    cols = ["t2m_c", "rh2m_pct", "apcp_mm", "mslp_hpa", "pwat_kgm2", "wspd10m_ms",
            "soilw_vol_pct"]
    labels = ["2 m temp", "2 m RH", "rainfall", "MSLP", "water vapour", "10 m wind",
              "soil moisture"]
    M = np.zeros((len(cols), 10))
    for i, c in enumerate(cols):
        for j, ld in enumerate(range(1, 11)):
            sub = g[g.lead_day == ld]
            M[i, j] = sub[c].notna().mean() if len(sub) else 0
    im = ax1.imshow(M * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(range(10), [str(i) for i in range(1, 11)], fontsize=6.5)
    ax1.set_yticks(range(len(cols)), labels, fontsize=6.5)
    ax1.set_xlabel("forecast lead day", fontsize=7.5)
    for i in range(len(cols)):
        for j in range(10):
            ax1.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.4,
                     color="white" if M[i, j] > 0.55 else "0.25")
    ax1.set_title("(a) Non-null share of each variable by lead day, %", fontsize=7.5, pad=4)
    cb = fig.colorbar(im, ax=ax1, fraction=0.032, pad=0.015)
    cb.ax.tick_params(labelsize=6)
    ax2 = fig.add_axes([0.65, 0.24, 0.325, 0.62])
    init = pd.to_datetime(pd.Series(sorted(g.init_date.unique())))
    ax2.scatter(init.dt.dayofyear, init.dt.year, marker="o", s=20, color="#2f6f4f", zorder=3)
    ax2.set_yticks([2019])
    ax2.set_ylim(2018.55, 2019.45)
    ax2.set_xticks([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349],
                   list("JFMAMJJASOND"), fontsize=6.5)
    ax2.set_xlabel("initialisation date — 17 cycles in 2019 (fortnightly Jun–Sep,\n"
                   "monthly otherwise)", fontsize=7.2, labelpad=2)
    ax2.grid(color="0.9", linewidth=0.4)
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=6.5)
    ax2.set_title("(b) Sampled reforecast cycles", fontsize=7.5, pad=4)
    fig.savefig(ASSETS / "fig2_coverage.png", dpi=230, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: the district layer
    d = pd.read_parquet(data / "geo/district_descriptors.parquet")
    w = pd.read_parquet(data / "geo/district_grid_weights.parquet")
    cells = w.groupby("region_id").size()
    fig = plt.figure(figsize=(6.9, 3.0))
    ax1 = fig.add_axes([0.075, 0.20, 0.26, 0.62])
    sc = ax1.scatter(d.centroid_lon, d.centroid_lat, c=d.elevation_mean,
                     s=np.sqrt(d.area_km2) / 3.5, cmap="terrain", linewidths=0.12,
                     edgecolors="0.3")
    ax1.set_xlabel("longitude (°E)", fontsize=7.5)
    ax1.set_ylabel("latitude (°N)", fontsize=7.5)
    ax1.set_aspect("equal")
    ax1.tick_params(labelsize=6.5)
    ax1.set_title("(a) 666 district centroids\n(colour = mean elevation, size = area)",
                  fontsize=7.5, pad=4)
    cb = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.015)
    cb.ax.tick_params(labelsize=6)
    ax2 = fig.add_axes([0.45, 0.20, 0.21, 0.62])
    ax2.hist(cells.values, bins=30, color="#8a6fb0", edgecolor="#48355f", linewidth=0.25)
    ax2.set_xlabel("overlapped 0.25° cells", fontsize=7.5)
    ax2.set_ylabel("districts", fontsize=7.5)
    ax2.tick_params(labelsize=6.5)
    ax2.grid(axis="y", color="0.9", linewidth=0.4)
    ax2.set_axisbelow(True)
    ax2.set_title(f"(b) Cells per district\nmedian {int(np.median(cells.values))}, "
                  f"range {int(cells.values.min())}–{int(cells.values.max())}",
                  fontsize=7.5, pad=4)
    ax3 = fig.add_axes([0.745, 0.20, 0.235, 0.62])
    ax3.hist(np.log10(d.border_distance_km), bins=26, color="#d08a3e", edgecolor="#7f4d16",
             linewidth=0.25)
    ax3.set_xticks(np.log10([1, 10, 100, 600]), ["1", "10", "100", "600"], fontsize=6.5)
    ax3.set_xlabel("distance to land boundary (km, log)", fontsize=7.5)
    ax3.set_ylabel("districts", fontsize=7.5)
    ax3.tick_params(labelsize=6.5)
    ax3.grid(axis="y", color="0.9", linewidth=0.4)
    ax3.set_axisbelow(True)
    ax3.set_title(f"(c) Border distance\nmedian {d.border_distance_km.median():.0f} km, "
                  f"max {d.border_distance_km.max():.0f} km", fontsize=7.5, pad=4)
    fig.savefig(ASSETS / "fig3_districts.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------------------
def build_content(st) -> Content:
    c = Content(st)

    # ==================================================================================
    # Page 2 - title of the paper, problem statement, data life cycle
    # ==================================================================================
    c.h1("Title of the Paper to Be Published")
    c.p("<b>Working title:</b> Explainable AI for District-Level Prediction of Medium-Range "
        "Weather Forecast Busts over India")

    c.h1("Problem Statement")
    c.p("An operational centre already issues a medium-range forecast for every district, and "
        "statistical post-processing already removes the errors a model makes consistently. "
        "What survives correction is the day-to-day failure: the forecast that is badly wrong "
        "because of the particular weather pattern of that day. Following Rodwell et al. [2], "
        "this project calls such a forecast a <b>bust</b> &mdash; an error that lands in the "
        "tail of that variable&rsquo;s own historical error distribution, above the 90th "
        "percentile computed on training data only. The literature diagnoses continental "
        "upper-air busts after the event [2], [3], verifies Indian rainfall and circulation at "
        "regional scale [6]&ndash;[9], or explains the weather quantity a model predicts "
        "rather than the model&rsquo;s own error [16]&ndash;[18]. The missing integration is an "
        "operationally usable, calibrated and locally explained probability that a specific "
        "medium-range forecast will fail, for several surface variables at once, at district "
        "scale over India.")
    c.p("The problem addressed in this assignment is the <b>data part</b> of that study. It is "
        "not a new weather forecast and does not claim to be one. The dataset must make a "
        "forecast archive and a verifying truth directly comparable at district scale, span "
        "enough years that rare failures are learnable, carry the predictors that hold the "
        "predictability signal (ensemble spread, run-to-run jumps, tendencies, season, "
        "geography and the Madden&ndash;Julian Oscillation), and be clean enough that no "
        "evaluation performed on it can be accused of leakage. The dataset described below is "
        "the result: NOAA GEFSv12 reforecast for the forecast side, ERA5 reanalysis for the "
        "truth, GADM 4.1 districts as the common spatial unit, and the static and dynamical "
        "covariates that surround them.")

    c.h1("Data Life Cycle")
    c.p("The full life cycle that the project follows is shown below. Stages 1 to 5 are "
        "complete for the archived dataset and are the subject of this document; stage 6 is "
        "implemented for the model now served and continues with every retrain; stages 7 to 9 "
        "follow the dataset submission.")
    c.table(
        "Data life cycle",
        ["#", "Stage", "What is done", "Status"],
        [
            ["1", "Problem definition",
             "Research gap and objective stated (above)", "Done"],
            ["2", "Data acquisition",
             "12 GEFSv12 variables and 5 members per 00 UTC cycle from the AWS reforecast "
             "archive; ERA5 from the Copernicus CDS; 666 GADM 4.1 districts and their weight "
             "table; elevation, land&ndash;sea masks, MJO index", "Done"],
            ["3", "Data description and exploration",
             "Types, schema, counts, missingness, error distributions, coverage by lead day "
             "and district", "Done"],
            ["4", "Validation and cleaning",
             "Completeness, duplicate, plausibility-range, alignment, leakage and consistency "
             "checks (Section 2)", "Done"],
            ["5", "Preprocessing, splitting and storage",
             "District area-weighting through one weight table; daily aggregation to UTC days; "
             "chronological split with a Day-10 embargo; hive-partitioned Parquet plus SQLite "
             "lineage", "Done"],
            ["6", "Model building",
             "One XGBoost error regressor per variable; a bust classifier trained on their "
             "out-of-fold error evidence; TreeSHAP attributions; a CNN challenger over the "
             "gridded fields", "Discharging; repeats on each retrain"],
            ["7", "Baselines and calibration",
             "Climatology, lead-day, ensemble-spread, EMOS, IDR and analog baselines scored on "
             "the same held-out rows; Platt calibration; promotion gate", "In progress"],
            ["8", "Evaluation",
             "Brier skill, ROC-AUC, SEDI, CORP reliability decomposition, relative economic "
             "value, conformal coverage, SHAP stability", "In progress"],
            ["9", "Reporting and publication",
             "Results, comparison with the baseline ladder, paper", "To follow"],
        ],
        widths=[0.035, 0.275, 0.50, 0.19], small=True)

    # ==================================================================================
    # 1. DATA DESCRIPTION
    # ==================================================================================
    c.h1("1. DATA DESCRIPTION")
    c.h2("1.1 Type of data")
    c.p("The dataset has four linked parts, and all four are numeric. There is a <b>forecast "
        "archive</b> (NOAA GEFSv12 reforecast: one 00 UTC cycle a day, 5 ensemble members, "
        "0.25&deg; global grid, 3-hourly steps reduced to daily fields, lead days 1 to 10); a "
        "<b>verifying observation archive</b> (ERA5 reanalysis, on the same 0.25&deg; grid and "
        "the same UTC day convention); a <b>geography layer</b> (666 GADM 4.1 districts, the "
        "grid-cell weight table that connects districts to cells, and per-district descriptors "
        "derived from the geometry); and <b>covariate files</b> (a daily Madden&ndash;Julian "
        "Oscillation index, an elevation grid for every cell the weight table uses, and ERA5 "
        "and GEFS land&ndash;sea masks). Forecasts and observations are not kept as two "
        "differently shaped products: both are reduced to the same canonical row &mdash; one "
        "district, one variable, one valid date, one value type, one ensemble member and lead "
        "day, one value &mdash; and it is the paired canonical row that every model reads. "
        "There is no text, image, or free-form data anywhere in the training path.")
    c.p("What is predicted is not a weather value but the <b>error between the two sides</b>. "
        "Each forecast row is paired with the observation for the same district, variable and "
        "UTC day, and the absolute error of the pair is carried with it. The label is derived "
        "from those errors: a bust is an error above the 90th percentile of that "
        "variable&rsquo;s own error distribution, and the percentile is computed on training "
        "cycles only, never on the rows it will be scored against. Eight surface variables are "
        "carried (Table 4). Wind direction is an input but deliberately not a label variable: "
        "a daily resultant direction is mostly noise when the wind is light, and 86.5% of its "
        "apparent busts under plain angle arithmetic disappeared once the error was measured "
        "the short way round the circle.")
    c.p("The data is grouped and correlated in space and in time rather than being a set of "
        "independent samples. Consecutive days at one district are strongly autocorrelated, "
        "neighbouring districts share 0.25&deg; cells through the weight table, and the five "
        "members of a cycle are perturbations of one analysis rather than five observations of "
        "the atmosphere. The chronological split, the 30-day out-of-fold blocks and the "
        "Day-10 embargo described in Section 1.4 exist because of that structure, not in spite "
        "of it.")
    c.table(
        "Attribute types in the canonical store (one row = one district, variable, valid date, "
        "value type, member and lead)",
        ["Attribute type", "Attributes", "Count"],
        [
            ["Identifier (text)",
             "record_id, upload_batch_id, source_file, source_column", "4"],
            ["Categorical (nominal)",
             "variable (8 levels), value_type (forecast, observed), grain, "
             "verification_status (final, provisional)", "4"],
            ["Numeric, continuous",
             "value, lat, lon, mapping_confidence", "4"],
            ["Timestamp",
             "init_date, valid_date, ingested_at", "3"],
            ["Categorical (region)",
             "region_id, region_name (666 levels), region_resolution_method", "3"],
            ["Numeric, discrete (integer)",
             "lead_time_days (1 to 10)", "1"],
            ["Categorical (member)",
             "ensemble_member_id (c00, p01&ndash;p04 in the archive; up to 31 in the "
             "operational feed)", "1"],
            ["<b>Total</b>", "", "<b>20</b>"],
        ],
        widths=[0.30, 0.58, 0.12])

    c.h2("1.2 Number of datasets and their sources")
    c.p("There are four source families, fetched by their own scripts under "
        "<font face=\"Serif\">backend/scripts/</font>, and ten products tracked in the "
        "repository. The bulk of the archive &mdash; the twenty-year reforecast &mdash; is "
        "fetched by CI and stored as a release asset rather than committed: at 0.81 TB for a "
        "full year at daily density, it does not belong in git. Every row records where it "
        "came from, so any value can be traced back to the file and the message that produced "
        "it: the reforecast carries the GRIB2 object path and message number for each of its "
        "twelve variables, and the observation side carries the product and API that served "
        "it. <b>No value in this dataset is synthetic, simulated, interpolated or "
        "placeholder.</b> Where a number could not be fetched, there is no row.")
    c.table(
        "Datasets, their sources and their terms",
        ["Dataset", "Source", "Used for", "Term"],
        [
            ["GEFSv12 reforecast (2000&ndash;2019)",
             "NOAA GEFSv12 reforecast archive, AWS Open Data bucket "
             "<font face=\"Serif\">noaa-gefs-retrospective</font>; one 00 UTC cycle a day, "
             "5 members (c00, p01&ndash;p04), 0.25&deg;, 3-hourly to 240 h [4], [26]",
             "Training and calibration forecasts", "Public domain (NOAA Open Data)"],
            ["GEFS operational feed",
             "NOAA NOMADS and its AWS mirror, pulled every six hours; up to 31 members",
             "Live scoring, Replay and the alerts feed", "Public domain"],
            ["ERA5 reanalysis",
             "Copernicus Climate Change Service, Climate Data Store, dataset "
             "<font face=\"Serif\">reanalysis-era5-single-levels</font> [20], [25]",
             "Observations for training, verification and the bust label", "CC-BY 4.0"],
            ["ERA5 recent days",
             "Open-Meteo Historical Weather API with "
             "<font face=\"Serif\">models=era5</font>, answered from ERA5&rsquo;s own "
             "0.25&deg; cell",
             "Observations for days the CDS archive lags", "CC-BY 4.0"],
            ["IMD gridded rainfall",
             "India Meteorological Department, Pune, 0.25&deg; gauge-based daily rainfall",
             "Planned rainfall truth (the served run&rsquo;s rainfall truth is ERA5)",
             "IMD terms, credit required"],
            ["District boundaries",
             "GADM 4.1 admin-2, 666 districts, with three deliberate corrections (Ladakh "
             "reassigned out of Jammu &amp; Kashmir, disputed-territory features dissolved "
             "into their parent districts, Delhi relabelled) [24]",
             "Map geometry, the weight table, and every district descriptor",
             "Free for academic and non-commercial use"],
            ["Cell elevation",
             "Open-Elevation public API, serving CGIAR-CSI SRTM 250 m, for the 4,902 cells "
             "the weight table uses [22]",
             "Elevation descriptor and the CNN&rsquo;s terrain channel", "Acknowledge CIAT"],
            ["Land&ndash;sea masks",
             "ERA5&rsquo;s own land&ndash;sea mask, and GEFS sea cells identified from the "
             "archived gridded soil-moisture fields",
             "Refusing to compare soil moisture over the sea", "CC-BY 4.0 / public domain"],
            ["MJO index",
             "NOAA PSL OLR-based MJO index (OMI), 1991 to present, transformed to the RMM1 "
             "/ RMM2 convention by PSL&rsquo;s own documented mapping [21]",
             "MJO features (dropped from new runs while the file&rsquo;s tail is stale)",
             "NOAA open data"],
            ["Verification-product comparison",
             "MERRA-2 through the NASA POWER daily API, paired against ERA5 [23]",
             "Measuring the ground truth&rsquo;s own uncertainty",
             "NASA open data"],
        ],
        widths=[0.185, 0.475, 0.215, 0.125], small=True)
    c.p("Two details of the acquisition matter for reading the rest of this document. "
        "First, only the GRIB2 messages that are needed are downloaded, by HTTP range "
        "requests keyed off each file&rsquo;s <font face=\"Serif\">.idx</font> sidecar: "
        "1.79 GB per initialisation for 9 variables and 2.2 GB for 12, where the file "
        "holding just one of those fields is 296 MB on its own, of which 19 MB is read. "
        "Second, the twelve variables are not all of the same kind: nine have an observed "
        "counterpart and are paired; three &mdash; 500 hPa geopotential height, surface CAPE "
        "and surface CIN &mdash; have no observation to verify against, never enter the "
        "paired store, and reach only the convolutional challenger as grid channels.")
    c.table(
        "The eight predictand variables and the three grid-only predictors",
        ["Variable (canonical name)", "Unit", "GEFSv12 field", "Leads held",
         "Bust label"],
        [
            ["temperature_c", "&deg;C", "tmp_2m (t2m)", "Day 1&ndash;10", "Yes"],
            ["humidity_pct", "% RH", "spfh_2m (sh2), converted to RH after the district mean "
             "on both sides", "Day 1&ndash;10", "Yes"],
            ["rainfall_mm", "mm/day", "apcp_sfc (tp), accumulated over the model day",
             "Day 1&ndash;10", "Yes"],
            ["pressure_hpa", "hPa", "pres_msl (msl)", "Day 1&ndash;10", "Yes"],
            ["atmospheric_<br/>moisture_kgm2", "kg/m&sup2; (TCWV)", "pwat_eatm (pwat)",
             "Day 1&ndash;10", "Yes"],
            ["wind_speed_ms", "m/s", "ugrd_hgt / vgrd_hgt (u10, v10)",
             "Day 1&ndash;5 only", "Yes"],
            ["wind_direction_deg", "degrees (from)", "ugrd_hgt / vgrd_hgt (u10, v10)",
             "Day 1&ndash;5 only", "No &mdash; input only"],
            ["soil_moisture_pct", "% volumetric", "soilw_bgrnd (soilw), top layer",
             "Day 1&ndash;3 only", "Yes"],
            ["Z500 (hgt_pres_abv700mb)", "m", "gh at 500 mb", "Day 1&ndash;10",
             "No &mdash; CNN channel"],
            ["CAPE (cape_sfc)", "J/kg", "cape", "Day 1&ndash;10", "No &mdash; CNN channel"],
            ["CIN (cin_sfc)", "J/kg", "cin", "Day 1&ndash;10", "No &mdash; CNN channel"],
        ],
        widths=[0.235, 0.11, 0.40, 0.135, 0.12])
    c.p("The lead-day limits in that table are the archive&rsquo;s, not the project&rsquo;s: "
        "the GEFSv12 reforecast carries 10 m wind only to 120 h and soil moisture only to "
        "72 h, so no model can have seen those variables beyond those leads. The serving "
        "layer enforces the same limits rather than scoring a variable at a lead its model "
        "never trained on.")

    # ==================================================================================
    # 1.3, 1.4
    # ==================================================================================
    c.h2("1.3 Missing data")
    c.p("Missing data takes two forms here, and they are treated differently. In the "
        "canonical store a value that does not exist is written as an <b>absent row</b>, not "
        "as a null cell: the store is normalised on (district, variable, valid date, value "
        "type, member, lead), so an incomplete source cannot leave a half-filled record "
        "behind. The ingest refuses a cycle it cannot complete rather than storing part of "
        "one, because a short rainfall <i>sum</i> is roughly half the real accumulation and "
        "rainfall drives most busts. Where nulls do appear, they are structural, and the "
        "contract that every frame is checked against names each one and why it is allowed.")
    c.p("Measured on the two archived 2019 sample files that ship with the repository, whose "
        "numbers are reproducible from the files themselves: the reforecast file holds "
        "30,600 rows &times; 36 columns = 1,101,600 cells, of which 156,060 (14.2%) are "
        "null. The nulls are exactly two variables &mdash; 10 m wind speed and direction, "
        "held only to Day 5 (15,300 of 30,600 rows each, 50.0%), and soil moisture, held "
        "only to Day 3 (21,420 of 30,600, 70.0%). The equality is exact in both cases: the "
        "gap is a property of the archive&rsquo;s lead coverage, not of any particular "
        "cycle, and the coverage is identical for all 17 sampled cycles and all five "
        "members (Figure 2a). The ERA5 observation sample has 13,680 rows &times; 16 columns "
        "= 218,880 cells with <b>0 missing</b>: every one of its 36 stations reports on all "
        "380 days (1 January 2019 to 15 January 2020). The MJO index has 12,959 daily rows "
        "from 1 January 1991 with 0 missing and no date gaps, and its earliest date is the "
        "only boundary: it is joined to a forecast cycle as of the issue date with a "
        "five-day tolerance, so cycles before 1991 or after a stale tail simply carry no MJO "
        "value. Every one of the 666 district rows has a complete descriptor row, and every "
        "one of the 666 weight vectors sums to 1.000000.")
    c.p("The second form of missingness is at the level of the archive rather than the cell: "
        "17 of the 7,305 daily reforecast cycles of 2000&ndash;2019 (0.23%) are absent from "
        "the store &mdash; ten days of 2004, five of 2006 and two of 2008 &mdash; and "
        "training and scoring simply have no events on those dates. Nothing is imputed to "
        "cover them, and re-fetching them is an open fetch-once question rather than a quiet "
        "fix. Two further member-days (p03 of 2004-03-05 and p01 of 2019-02-09) carried "
        "values that cannot be weather rather than missing ones; those were dropped by the "
        "plausibility check described in Section 2.")
    c.table(
        "Attributes by level of missing data (measured on the files in "
        "<font face=\"Serif\">backend/data/</font>)",
        ["Missing-data level", "Attributes", "Records affected"],
        [
            ["0% (complete)",
             "All 20 canonical-store columns; all 16 observation columns; 26 of the 36 "
             "reforecast columns; the MJO index, the weight table, the descriptors and both "
             "land&ndash;sea masks",
             "0 nulls in the canonical store; 0 of 218,880 observation cells; 945,540 of "
             "1,101,600 reforecast cells (85.8%)"],
            ["up to 50%",
             "10 m wind speed and 10 m wind direction (present on Day 1&ndash;5 only)",
             "15,300 of 30,600 rows each (50.0%), identical for every cycle and member"],
            ["70%",
             "Soil moisture (present on Day 1&ndash;3 only)",
             "21,420 of 30,600 rows (70.0%)"],
            ["more than 70%", "None", "0"],
            ["<b>Total</b>", "<b>36 reforecast columns, 16 observation columns</b>",
             "<b>156,060 null cells, all of them structural</b>"],
        ],
        widths=[0.20, 0.42, 0.38], small=True)

    c.h2("1.4 Common attributes")
    c.p("All 20 canonical columns (Table 2) are common to every source, whatever format it "
        "arrived in: GRIB2 and NetCDF from the archives, CSV and XLSX from an upload, JSON "
        "for elevation, and zipped NetCDF pairs from the CDS. The parsers are "
        "format-agnostic and the schema mapper matches incoming columns to the canonical "
        "schema with a confidence score, but the schema itself is frozen: the Arrow schema "
        "is asserted equal to the field list of the canonical row model at import time, so a "
        "drift between the two fails loudly at once instead of quietly downstream.")
    c.p("Units are canonical and identical on both sides of every label: mm/day for rainfall "
        "over the model day, &deg;C for temperature, %RH for humidity, hPa for "
        "mean-sea-level pressure, kg/m&sup2; for column water vapour, % volumetric for soil "
        "moisture, and m/s and degrees (meteorological, from-direction) for wind. Derived "
        "quantities are derived the same way on both sides &mdash; relative humidity from "
        "specific humidity, temperature and pressure <i>after</i> the district mean; wind "
        "speed and direction from the two components &mdash; so a forecast row and a truth "
        "row are never two different quantities under one name. An accumulation is stamped "
        "by the end of the hour it covers, so every variable is grouped by "
        "<font face=\"Serif\">valid_time &minus; 1 h</font>; that makes an observation day "
        "the half-open window (t&minus;24 h, t], which is deliberately the same window the "
        "forecast side uses for day k, ((k&minus;1)&times;24, k&times;24]. Both sides of a "
        "bust label therefore describe the identical interval.")
    c.p("The spatial convention is common too, and it is the one design decision the whole "
        "dataset rests on: <b>there is exactly one weight table</b>. Each district&rsquo;s "
        "value is the area-weighted mean of every 0.25&deg; cell its polygon overlaps "
        "&mdash; 10,373 district&ndash;cell pairs over 4,902 distinct cells, a median of 14 "
        "cells per district (Figure 3b) &mdash; rather than the value of the nearest grid "
        "point. That is not a refinement: ten districts, among them Kolkata, Hyderabad, the "
        "Puducherry enclaves and Lakshadweep, contain no grid centre at all, and for them a "
        "nearest-point sample would mean reporting a cell that lies outside the district it "
        "is named after. Forecasts and observations go through the same table, so both sides "
        "of the label are area means over the same polygon. The same attribute structure is "
        "also common across the splits (Table 6) and across the eight variables: the frames "
        "differ in which cycles they hold, never in their columns.")
    c.table(
        "Split composition of the training data",
        ["Split", "Cycles", "Years", "What it is used for"],
        [
            ["train", "6,193 daily cycles", "2000&ndash;2016",
             "Bust thresholds; the per-district, per-variable, per-lead, per-month bias "
             "table; historical bust frequency; the regressors and the classifier"],
            ["validation", "Held out of training", "within 2000&ndash;2016",
             "Early stopping, Platt calibration and tuning (so validation scores are "
             "optimistic, and are never quoted as test results)"],
            ["test (held out)", "365 daily cycles", "2017",
             "Every metric the site publishes; the promotion gate; the baseline ladder. "
             "2018 and 2019 sit in the archive unused, and are the next evaluation"],
            ["embargo", "10 days either side of every boundary", "&mdash;",
             "No observed day appears on both sides of a split or of an out-of-fold block; "
             "out-of-fold blocks are 30-day spans of forecast cycles"],
        ],
        widths=[0.13, 0.20, 0.18, 0.49], small=True)
    return c



def build_content_part2(c: Content) -> Content:
    # ==================================================================================
    # 2. DATA CLEANING METHOD SELECTION AND JUSTIFICATION
    # ==================================================================================
    c.h1("2. DATA CLEANING METHOD SELECTION AND JUSTIFICATION")
    c.p("This dataset is fetched from other people&rsquo;s archives rather than authored by "
        "the team, so its cleaning is not about repairing typing errors. It is about the "
        "places where two individually good products will still disagree or mislead: a "
        "forecast field that has no observed counterpart, two models that fill the sea "
        "oppositely, a truth product that arrives one day late, an input that quietly uses "
        "the answer. Each candidate problem below was checked against the data, and a "
        "treatment was chosen only where a check found something. The treatments were "
        "chosen because the obvious alternatives would damage the dataset: imputing a missing "
        "forecast step fabricates the accumulation that drives most busts, deleting extreme "
        "errors removes exactly the events the label is built on, and subtracting a "
        "per-district offset without recording it hides the difference between a biased "
        "district and a failing forecast.")
    c.table(
        "Cleaning and validation methods selected, and why",
        ["Issue", "Finding", "Method selected", "Justification"],
        [
            ["Missing values",
             "No null cells in the canonical store or in the observation sample; 14.2% of "
             "the reforecast sample&rsquo;s cells are the documented structural gaps (wind "
             "beyond Day 5, soil moisture beyond Day 3)",
             "Absence of a row rather than a null cell; a completeness guard refuses a cycle "
             "that is short of its 33,300 rows; no imputation anywhere in the pipeline",
             "A short rainfall sum is about half the real accumulation, and a fabricated "
             "member would enter the ensemble spread as evidence"],
            ["Duplicates",
             "Re-ingesting a source leaves two rows for one key. Found on real data: IMD "
             "rainfall batches ingested after the ERA5 batches won the dedupe and replaced "
             "2016&ndash;2019 rainfall truth in the training store",
             "Dedupe on (region, valid_date, variable, value_type, init_date, lead, member, "
             "source_column) with the newest non-provisional row winning; quarantined and "
             "retired batches are refused before any frame is built, and a cached year is "
             "rebuilt when its batch signature changes",
             "A duplicate is not a repeated row here, it is a different truth product: it "
             "changes the label silently and inflates any score computed on it"],
            ["Implausible values (outliers)",
             "Two corrupt member-days in twenty years: p03 of 2004-03-05 with Day-2 2 m "
             "temperature from &minus;23,380,072 to +21,808,336 &deg;C in all 666 districts "
             "(other members &minus;18.5 to 32.4); p01 of 2019-02-09 with Day-1 column water "
             "vapour of 12,500&ndash;12,546 kg/m&sup2; (others 0.7&ndash;55.3)",
             "Range check against a per-variable plausible range, applied at the head of the "
             "one frame-building path that training, scoring, live and Replay all share; a "
             "corrupt member is dropped and the ensemble is the members left (666 rows on "
             "each affected cycle, none on a clean one)",
             "One such member put that day&rsquo;s temperature RMSE at 34,612 &deg;C. A "
             "plausible range is not a smoothing filter: real extremes stay, because the "
             "label lives in the tail"],
            ["Data leakage",
             "A time-lagged error input existed that no live forecast can have (it reads the "
             "previous lead&rsquo;s realised error of the same forecast), and 2016&ndash;2019 "
             "rainfall truth came from a one-day-late IMD file that had won the dedupe",
             "Observation-derived columns are refused as model inputs (a test enforces it); "
             "the four IMD batches were quarantined and the store restored to ERA5; "
             "thresholds, bias tables and calibrators are fitted on training cycles only; "
             "the split is chronological with a Day-10 embargo",
             "A held-out score must be the score live scoring would produce. The leak was "
             "found by an audit after the model was served, and the model it affected is "
             "replaced by the next retrain rather than defended"],
            ["Skewed labels (rainfall)",
             "Rainfall is zero on 44.8% of observation days and its error is heavy tailed: "
             "median 1.53 mm/day, 90th percentile 14.38, 99th 48.14, maximum 170.8 mm/day in "
             "the archived sample",
             "A percentile definition of a bust &mdash; the variable&rsquo;s own training-set "
             "90th percentile &mdash; instead of a fixed threshold; zero-inflation reported "
             "with the variable rather than corrected by resampling",
             "A fixed millimetre threshold would label almost nothing in a dry district and "
             "almost everything in the monsoon; resampling dry days would repeat "
             "near-identical rows instead of adding information"],
            ["Skewed labels (steady bias)",
             "64% of squared error for temperature and humidity, and 90% for soil moisture, "
             "is a fixed per-district offset that ordinary bias correction removes; rainfall "
             "is 6% and its error grows with lead as forecast failure should",
             "Busts are defined on bias-corrected error: the training-period mean error per "
             "(district, variable, lead, month) is removed first, with backoff to the "
             "district&rsquo;s month across leads and then to its year when a cell holds "
             "fewer than 30 events; a key that is thin at every level is not compared at all",
             "Otherwise a &ldquo;bust&rdquo; means &ldquo;this district is always "
             "off&rdquo;, which NCMRWF&rsquo;s own post-processing already removes &mdash; "
             "the failure that remains is the one worth predicting"],
            ["Inconsistent format or units",
             "Inputs arrive as GRIB2, NetCDF, CSV, XLSX and JSON; ERA5 stamps an "
             "accumulation with the end of its hour; humidity is specific humidity on the "
             "forecast side and relative humidity on the observation side; wind arrives as "
             "two components; the CDS returns a zip of two NetCDF files split by step type",
             "Format-agnostic parsers into one canonical schema with confidence-scored "
             "column matching; accumulation grouped by valid_time &minus; 1 h; relative "
             "humidity derived from specific humidity, temperature and pressure after the "
             "district mean on both sides; speed and direction derived from u and v; the CDS "
             "zip is unzipped and the two files read separately",
             "One schema and one set of units is what makes a forecast row and a truth row "
             "comparable at all; deriving RH after the mean on both sides keeps a bounded "
             "quantity from being averaged as a ratio"],
            ["Misplaced or lost geography",
             "For eight 2017 districts the schema mapper did not recognise the region column "
             "and placed rows by coordinate; five of them vanished and three held a "
             "neighbour&rsquo;s values. Separately, ten districts contain no 0.25&deg; grid "
             "centre at all",
             "One area-overlap weight table used by both sides; rows are keyed by district "
             "identifier, not by position; the mapper is fixed and the affected 2017 "
             "districts are re-ingested",
             "A nearest-point sample cannot represent a district that contains no grid "
             "centre, and a mislabelled district corrupts both its own rows and its "
             "neighbour&rsquo;s"],
            ["Sea read as land (soil moisture)",
             "GEFS writes the sea as saturated soil (~98% volumetric, median 99.78) while "
             "ERA5 writes it as 0. Paired through the weight table this labelled four "
             "districts (Nicobar Islands, Lakshadweep, Diu, Mumbai City) a soil-moisture "
             "bust on 100% of their rows against 0.9% elsewhere, and the effect is a "
             "gradient in how much sea each coastal district contains",
             "A per-district land&ndash;sea mask built from ERA5&rsquo;s own mask and from "
             "GEFS sea cells identified in the archived gridded fields; soil moisture is "
             "dropped on both sides for any district more than 5% sea in either product, and "
             "the exclusion list is required to exist rather than defaulted",
             "Masking is honest and imputation is not: a district that is mostly sea has no "
             "soil-moisture value to compare, and leaving the pair in teaches a model a "
             "district-identity shortcut"],
            ["Incomplete cycles",
             "NOAA drops forecast steps mid-pull; a short rainfall sum silently halves the "
             "quantity that drives most busts, and a partial pull can leave an observation "
             "fetch half-done",
             "A completeness check per cycle refuses an incomplete cycle rather than "
             "publishing part of one, with an S3 byte-range fallback that refills missing "
             "messages before the daily reduction; observation pulls report completeness "
             "rather than assuming it",
             "Publishing nothing for a day is recoverable; publishing a thin cycle teaches "
             "the model that a real rain day was dry"],
            ["Scaling",
             "Not applied to the stored data, and not applied to the tree models",
             "None for the stored dataset or for XGBoost, which is invariant to a monotone "
             "rescaling of a feature; the convolutional challenger normalises its channels "
             "with training statistics; the classifier&rsquo;s probabilities are mapped to "
             "observed frequencies by a Platt fit on validation, with a non-positive slope "
             "refused",
             "Rescaling the stored values would add a version to every file and change "
             "nothing the served model reads; calibrating the probability is needed, because "
             "an uncalibrated classifier&rsquo;s lowest reliability bin predicted 0.158 "
             "against ~0.07 observed"],
        ],
        widths=[0.115, 0.30, 0.315, 0.27], small=True)

    # ==================================================================================
    # 3. RESULTS
    # ==================================================================================
    c.h1("3. RESULTS")
    c.h2("3.1 Verification results")
    c.p("Every cycle is verified before it is stored, and the check is structural rather "
        "than statistical. At district scale one cycle is 666 districts &times; 5 members "
        "&times; 10 lead days = 33,300 wide rows, and a cycle that does not match that shape "
        "is refused rather than written short. Measured end to end on 2017: <b>365 cycles, "
        "12,154,500 rows, zero refusals, every cycle exactly 33,300 rows</b>. Across "
        "2000&ndash;2019 the store holds 7,288 of the 7,305 daily cycles (99.77%), the gaps "
        "being the 17 dates listed in Section 1.3.")
    c.p("The two files reproduced for this report were verified the same way and pass the "
        "same checks. The reforecast sample has 30,600 rows = 17 cycles &times; 36 stations "
        "&times; 5 members &times; 10 lead days, with 1,800 rows on every cycle and no cycle "
        "short. The observation file has 13,680 rows = 36 stations &times; 380 days with no "
        "null in any of its 16 columns. Most important for pairing, the two files name the "
        "same 36 stations, so all 6,120 archived forecast rows have a truth for the same "
        "district and UTC day &mdash; the join that a bust label is built on is lossless "
        "here, as it is at district scale, where the forecast and the truth are reduced "
        "through the same weight table.")
    c.p("Two alignment results deserve to be stated because the dataset is unusable if they "
        "fail. First, dates: IMD&rsquo;s gauge day runs 0830&ndash;0830 IST, so IMD&rsquo;s "
        "date D is model day D&minus;1. A join that treated them as the same day put every "
        "IMD value one model day late. Settled from data rather than from either "
        "documentation: the Spearman correlation of IMD district rainfall on IMD date D "
        "against independent reanalysis rainfall summed to UTC day D+k peaks at "
        "k&nbsp;=&nbsp;&minus;1 for 8 of 8 districts against ERA5 and for the same 8 of 8 "
        "against MERRA-2 (means 0.698 and 0.810, against 0.556 and 0.621 at k&nbsp;=&nbsp;0), "
        "and the merge now moves every IMD date back one day and refuses to write a merge "
        "whose rainfall does not correlate best with ERA5 at lag 0. Second, product "
        "agreement: ERA5 and MERRA-2 are close on average, and a comparison of 21,492 paired "
        "city-days measured bias to within 0.06 &deg;C, 2.4 %RH, 0.1 mm/day and 0.75 m/s. The "
        "same comparison is what measures the truth&rsquo;s own uncertainty, below.")
    c.table(
        "Forecast&ndash;observation error statistics of the archived 2019 paired sample "
        "(6,120 district-day pairs; the 90th percentile is the bust threshold under the "
        "training-only rule)",
        ["Variable", "Unit", "Bias", "MAE", "Median", "p90 (train.)", "p99", "Maximum"],
        [
            ["temperature_c", "&deg;C", "+0.64", "1.81", "1.25", "4.16", "8.54", "17.34"],
            ["humidity_pct", "%RH", "&minus;9.04", "11.11", "9.11", "23.57", "37.57", "49.29"],
            ["rainfall_mm", "mm/day", "+0.25", "5.30", "1.53", "14.38", "48.14", "170.81"],
            ["pressure_hpa", "hPa", "&minus;0.62", "1.31", "0.88", "2.98", "6.37", "19.69"],
            ["atmospheric_<br/>moisture_kgm2", "kg/m&sup2;", "&minus;1.30", "3.20", "2.33",
             "7.00", "14.27", "34.95"],
            ["wind_speed_ms", "m/s", "+0.71", "0.97", "0.66", "2.19", "4.77", "10.37"],
        ],
        widths=[0.25, 0.095, 0.08, 0.075, 0.10, 0.135, 0.075, 0.19],
        legend="Wind is paired on 3,060 of the 6,120 pairs: the archive holds it only to Day "
               "5. The humidity bias (&minus;9.0 %RH, the forecast drier than ERA5) is the "
               "bias that motivated per-district, per-lead bias correction before the bust "
               "label is defined.")
    c.p("Table 8 shows the shape of each variable&rsquo;s error, and two things stand out. "
        "Rainfall is the hardest and the most skewed: nearly half of its observation days are "
        "dry, its median error is 1.5 mm/day, and its 99th percentile is 32 times that, with "
        "16.6% of events above the threshold &mdash; the tail that matters is long and it is "
        "where the label lives. Humidity carries a systematic dry bias of about 9 %RH, so "
        "more of its &lsquo;large errors&rsquo; are a standing offset than a forecast "
        "failure. That distinction is measurable, and it is what the bias-corrected label "
        "version exists to separate.")
    c.table(
        "Absolute error by lead day (MAE of the ensemble mean against ERA5, archived 2019 "
        "sample)",
        ["Variable", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
        [
            ["temperature_c (&deg;C)", "1.70", "1.77", "1.74", "1.83", "1.80", "1.84", "1.91",
             "1.83", "1.87", "1.86"],
            ["rainfall_mm (mm/day)", "4.59", "4.60", "5.11", "5.11", "4.82", "5.31", "5.71",
             "6.59", "5.53", "5.64"],
            ["humidity_pct (%RH)", "10.8", "10.8", "10.6", "10.7", "10.6", "10.9", "11.6",
             "11.7", "11.8", "11.4"],
            ["wind_speed_ms (m/s)", "0.82", "0.91", "1.02", "1.03", "1.05", "&mdash;",
             "&mdash;", "&mdash;", "&mdash;", "&mdash;"],
        ],
        widths=[0.24, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076],
        legend="Rainfall error grows with lead, as a forecast failure should, and roughly "
               "doubles from Day 1 to Day 8. Wind stops at Day 5 because the archive does. "
               "Temperature and humidity barely grow at all, which is the signature of steady "
               "per-district bias rather than of a forecast coming apart.")

    c.h2("3.2 Label statistics")
    c.p("The stored quantity is a continuous error and the label is its binary tail: a bust "
        "is an error above the training-set 90th percentile for that variable. So the "
        "distribution in Figure 1 <i>is</i> the label&rsquo;s support, and the dashed line "
        "in each panel is the threshold it will be cut at. The distributions are all "
        "right-skewed, which is what makes a percentile definition necessary rather than "
        "convenient: for temperature the threshold is 4.16 &deg;C against a median error of "
        "1.25, while for rainfall the threshold is 14.4 mm/day against a median of 1.5, and "
        "the same millimetre figure would mean something completely different in a desert "
        "and in Kerala.")
    c.p("Two rates are worth stating, and they are different numbers for a reason. In this "
        "in-sample illustration the thresholds are quantiles of the same 6,120 rows they "
        "label, so each variable busts on exactly 10% of them by construction; an event "
        "(any variable) busts on 38.6% of the 6,120 paired rows. In production the "
        "thresholds and the bias table are fitted on training cycles only and the label is "
        "computed on bias-corrected error, and the published event base rate on held-out "
        "data is <b>43&ndash;49%</b> depending on the run and on how many variables exist at "
        "that lead. The reason is structural: an event busts if <i>any</i> of about eight "
        "variables is over its own threshold, so 1 &minus; 0.9<sup>8</sup> &asymp; 57% "
        "before dependence between the variables, and fewer than eight variables are "
        "available from Day 4 and Day 6. Any analysis that assumed a 10% base rate would "
        "mis-weight the problem; that is precisely why the served classifier is not "
        "re-weighted for rarity.")
    c.figure(ASSETS / "fig1_error_distributions.png",
             "Figure 1. Absolute error of the GEFSv12 ensemble mean against ERA5 for the six "
             "continuous pairings of the archived 2019 sample (6,120 pairs, wind 3,060). The "
             "dashed line is the variable&rsquo;s own 90th percentile, which is the bust "
             "threshold; the clipped right-hand bar in each panel collects the tail beyond "
             "the 99.5th percentile.", 1.0)
    c.figure(ASSETS / "fig2_coverage.png",
             "Figure 2. Source coverage of the archived 2019 sample. (a) The non-null share "
             "of each variable by lead day: the two white blocks are the archive&rsquo;s "
             "lead limits (10 m wind beyond Day 5, soil moisture beyond Day 3), identical "
             "for every cycle and member. (b) The 17 sampled initialisation dates of 2019: "
             "fortnightly through the monsoon, monthly otherwise, so each year samples the "
             "full seasonal cycle without pretending that neighbouring cycles are "
             "independent.", 1.0)
    c.figure(ASSETS / "fig3_districts.png",
             "Figure 3. The static geography layer. (a) All 666 district centroids, coloured "
             "by mean elevation and sized by area; the Himalayan districts are both the "
             "largest and the highest, and elevation is a real driver of pressure error "
             "(Leh runs about ten times the typical district). (b) How many 0.25&deg; cells "
             "each district&rsquo;s polygon overlaps &mdash; the weight table in one "
             "picture. (c) Distance from each district to the outer edge of the dissolved "
             "landmass (coastline and international borders together).", 1.0)

    c.h2("3.3 Effect of the cleaning steps on the label distribution")
    c.p("The table compares each quantity before and after the treatment that targeted it. "
        "Every pair of numbers was measured on real data, and each is the reason the "
        "treatment exists rather than an argument for it. The last three rows are the ones "
        "that changed the label most: they are the difference between a label that measures "
        "forecast failure and one that measures a district&rsquo;s permanent offset or a "
        "mis-scaled angle.")
    c.table(
        "Before and after the cleaning steps (measured on real data)",
        ["Quantity", "Before the treatment", "After the treatment"],
        [
            ["2004-03-05 Day-2 temperature RMSE, with the corrupt p03 member in",
             "34,612 &deg;C",
             "Within the plausible range: exactly 666 rows dropped on each affected cycle "
             "and none on a clean one"],
            ["Rainfall truth for 2016&ndash;2019 in the live run&rsquo;s validation and test "
             "years", "One-day-late IMD-merged rainfall, differing from ERA5 in 72.5% of "
             "2018 rows and 79.8% of 2019 rows", "ERA5 CDS batches; the IMD batches are "
             "quarantined and any frame build that finds them now stops instead of choosing"],
            ["Humidity bust threshold, Nov 2017, 666 districts",
             "27.6 %RH on raw error (a dry-biased forecast made an ordinary day look like a "
             "failure)", "10.4 %RH after per-district, per-lead bias correction"],
            ["Wind-direction error p90, real CI sample",
             "191.0&deg; under plain angle arithmetic (a northerly is a bust by "
             "construction; 86.5% of direction busts were this artefact)",
             "115.1&deg; with the error taken the short way round the circle; ensemble "
             "spread p90 falls from 118.0&deg; to 55.0&deg;"],
            ["Event bust rate by lead, leaving out each district&rsquo;s own bias",
             "0.31 at Day 1 rising to 0.50 at Day 10 only after the offset is removed; "
             "correlation of lead with bust ~0.02 before",
             "0.25 at Day 1 to 0.50 at Day 10 with the bias removed leave-one-cycle-out "
             "(no in-sample fitting), and rainfall growing with lead throughout"],
            ["Soil moisture for the four sea-covered districts",
             "Labelled a bust on 100% of their paired rows against 0.9% for the other 662 "
             "districts &mdash; a guaranteed fabricated label",
             "Not compared at all: 18 districts exceed the 5% sea threshold in GEFS or ERA5 "
             "and are excluded on both sides"],
            ["Member count used for spread",
             "5 members a day in the archive (11 on Wednesdays) against 31 in the "
             "operational feed",
             "Stated as a limitation rather than corrected: spread features remain a noisy "
             "estimate, and a time-lagged pool across earlier cycles is added alongside them"],
        ],
        widths=[0.28, 0.36, 0.36], small=True)

    # ==================================================================================
    # 4. CONCLUSION
    # ==================================================================================
    c.h1("4. CONCLUSION")
    c.p("The dataset is four linked, all-numeric parts on a single spatial unit: the NOAA "
        "GEFSv12 reforecast (2000&ndash;2019, one 00 UTC cycle a day, five members, 0.25&deg;, "
        "twelve variables, Day 1&ndash;10) as the forecast side; ERA5 reanalysis, reduced "
        "through the same weight table, as the truth; GADM 4.1 as 666 districts with one "
        "area-overlap weight table of 10,373 district&ndash;cell pairs over 4,902 cells; and "
        "the covariates that surround them (elevation, two land&ndash;sea masks, the MJO "
        "index). Both sides are joined into one canonical 20-column row, the error of each "
        "pair is carried with it, and the label &mdash; a bust above that variable&rsquo;s "
        "own training-set 90th percentile &mdash; is derived from it. It is complete: no "
        "null cells in the store, no nulls anywhere in the observation sample, no duplicate "
        "keys after dedupe, all 666 districts with a complete weight vector summing to 1.0, "
        "and 7,288 of 7,305 archive cycles present. The only missingness is structural and "
        "is exactly the archive&rsquo;s own lead limits: 50% of rows for 10 m wind (Day 1&ndash;5) "
        "and 70% for soil moisture (Day 1&ndash;3).")
    c.p("The cleaning was chiefly verification, because the usual treatments would damage "
        "products that are correct in themselves: the real problems found were not missing "
        "values but two products disagreeing &mdash; a sea convention read as saturated "
        "soil, an IMD date convention one day out, an angle difference that made a northerly "
        "a bust, a truth file that won a dedupe it should have lost, and one input that used "
        "an observation from after issue time. Each was measured, treated, and where a "
        "trained model still carries the old behaviour the model is to be replaced rather "
        "than defended. The effect on the label is visible: the humidity threshold falls "
        "from 27.6 to 10.4 %RH, wind-direction error p90 from 191&deg; to 115&deg;, and the "
        "four sea-covered districts stop being guaranteed busts.")
    c.p("The main limitations are stated with the numbers, because they bound what the "
        "dataset can support. The truth is a reanalysis, not a gauge network: two leading "
        "products disagree per day by 24&ndash;43% of the bust threshold depending on the "
        "variable (rainfall 23.5%, humidity 36.5%, wind 38.0%, temperature 43.5% of the "
        "threshold over 21,492 paired city-days), and that disagreement is a floor under "
        "every label. Only five of the thirty-one operational ensemble members exist in the "
        "archive, so spread is a noisy estimate. Two variables cannot be scored at long "
        "lead at all. One held-out year (2017) has been evaluated so far; 2018 and 2019 are "
        "in the archive and unused. And the bust is defined on surface-variable error rather "
        "than on the synoptic 500 hPa criterion of Rodwell et al. [2] &mdash; deliberately, "
        "because surface error is what reaches agriculture and disaster response. The next "
        "stages of the life cycle are to complete the baseline ladder and calibration on the "
        "bias-corrected label, to evaluate on the held-out multi-year window, and to publish "
        "the results with the explanation stability that the literature survey requires.")
    return c

def build_references(c: Content) -> Content:
    c.h1("5. REFERENCES")
    c.p("The companion document for this assignment is the project&rsquo;s literature survey, "
        "<font face=\"Serif\">docs/literature-survey-explainable-ai-weather-busts.pdf</font>, "
        "which reviews the twenty studies the numbered references above correspond to and "
        "positions this dataset within them. The report below cites those references by the "
        "same numbers.")
    c.p("<b>Sources of the datasets.</b> Every number in this report traces to one of the "
        "sources below. Each was fetched by its own script in "
        "<font face=\"Serif\">backend/scripts/</font>, which records the source in the data "
        "it writes, and each keeps its own licence and attribution &mdash; the project&rsquo;s "
        "MIT licence covers the project&rsquo;s own code and documents only. Boundary "
        "geometry for a political map of India raises questions of its own, reviewed "
        "separately in the repository.")
    c.ref("Forecast archive: NOAA GEFSv12 reforecast (2000&ndash;2019), AWS Open Data bucket "
          "<font face=\"Serif\">noaa-gefs-retrospective</font> &mdash; "
          "https://registry.opendata.aws/noaa-gefs-reforecast/ [26] and "
          "https://noaa-gefs-retrospective.s3.amazonaws.com (public domain). "
          "Live feed: NOAA NOMADS and its AWS mirror (public domain).")
    c.ref("Observations: ERA5 reanalysis, Copernicus Climate Change Service (C3S) Climate "
          "Data Store, dataset <font face=\"Serif\">reanalysis-era5-single-levels</font> "
          "&mdash; https://cds.climate.copernicus.eu/datasets (CC-BY 4.0; &ldquo;Generated "
          "using Copernicus Climate Change Service information&rdquo;). Recent days are "
          "served through the Open-Meteo Historical Weather API with "
          "<font face=\"Serif\">models=era5</font> &mdash; https://open-meteo.com/ "
          "(CC-BY 4.0).")
    c.ref("Madden&ndash;Julian Oscillation: NOAA PSL OLR-based MJO Index (OMI), daily, 1991 "
          "&rarr; present &mdash; https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt, transformed "
          "to the RMM1/RMM2 convention by PSL&rsquo;s own documented mapping (NOAA open data; "
          "cite Kiladis et al. [21]).")
    c.ref("Ground-truth uncertainty: MERRA-2 through the NASA POWER daily point API &mdash; "
          "https://power.larc.nasa.gov/api/temporal/daily/point (NASA open data; cite "
          "NASA POWER [23]).")
    c.ref("District elevation: Open-Elevation public API, serving the CGIAR-CSI SRTM 250 m "
          "product &mdash; https://api.open-elevation.com/api/v1/lookup (acknowledge CIAT; "
          "cite Jarvis et al. [22]).")
    c.ref("District boundaries: GADM 4.1 admin-2 for India, 666 districts &mdash; "
          "https://gadm.org/ [24] (free for academic and other non-commercial use; "
          "redistribution or commercial use requires GADM&rsquo;s prior permission). "
          "Claimed-territory outlines on the map: Natural Earth &mdash; "
          "https://www.naturalearthdata.com/ (public domain).")
    c.ref("Rainfall truth (planned, and the truth for the IMD-verified years): India "
          "Meteorological Department gridded daily rainfall, 0.25&deg; &mdash; "
          "https://imdpune.gov.in/cmpg/Griddata/rainfall.php (IMD terms; credit required).")

    c.p("<b>Works cited.</b> The numbered entries are the references of the corresponding "
        "literature survey, kept in the same order so the two documents can be read "
        "together.")
    refs = [
        "Potvin, C. K., et al. (2024). Using machine learning to predict convection-allowing "
        "ensemble forecast skill: Evaluation with the NSSL Warn-on-Forecast System. "
        "<i>Artificial Intelligence for the Earth Systems</i>, 3, e230106. "
        "https://doi.org/10.1175/AIES-D-23-0106.1",
        "Rodwell, M. J., et al. (2013). Characteristics of occasional poor medium-range "
        "weather forecasts for Europe. <i>Bulletin of the American Meteorological Society</i>, "
        "94(9), 1393&ndash;1405. https://doi.org/10.1175/BAMS-D-12-00099.1",
        "Parsons, D. B., et al. (2019). The role of continental mesoscale convective systems "
        "in forecast busts within global weather prediction systems. <i>Atmosphere</i>, "
        "10(11), 681. https://doi.org/10.3390/atmos10110681",
        "Guan, H., et al. (2022). GEFSv12 reforecast dataset for supporting subseasonal and "
        "hydrometeorological applications. <i>Monthly Weather Review</i>, 150(3), "
        "647&ndash;665. https://doi.org/10.1175/MWR-D-21-0245.1",
        "Hamill, T. M., et al. (2022). The reanalysis for the Global Ensemble Forecast "
        "System, version 12. <i>Monthly Weather Review</i>, 150(1), 59&ndash;79. "
        "https://doi.org/10.1175/MWR-D-21-0023.1",
        "Dube, A., Ashrit, R., Singh, H., Iyengar, G., &amp; Rajagopal, E. N. (2016). "
        "Verification of medium range probabilistic rainfall forecasts over India. <i>Pure "
        "and Applied Geophysics</i>, 173, 2489&ndash;2510. "
        "https://doi.org/10.1007/s00024-016-1278-z",
        "Chakraborty, P., Sarkar, A., Bhatla, R., &amp; Singh, R. (2021). Assessing the skill "
        "of NCMRWF global ensemble prediction system in predicting Indian summer monsoon "
        "during 2018. <i>Atmospheric Research</i>, 251, 105255. "
        "https://doi.org/10.1016/j.atmosres.2020.105255",
        "Dube, A., Karunasagar, S., Ashrit, R., &amp; Mitra, A. K. (2022). Spatial "
        "verification of ensemble rainfall forecasts over India. <i>Atmospheric Research</i>, "
        "273, 106169. https://doi.org/10.1016/j.atmosres.2022.106169",
        "Roy Bhowmik, S. K., &amp; Durai, V. R. (2012). Development of multimodel ensemble "
        "based district level medium range rainfall forecast system for Indian region. "
        "<i>Journal of Earth System Science</i>, 121, 273&ndash;285. "
        "https://doi.org/10.1007/s12040-012-0160-1",
        "Rasp, S., &amp; Lerch, S. (2018). Neural networks for postprocessing ensemble "
        "weather forecasts. <i>Monthly Weather Review</i>, 146(11), 3885&ndash;3900. "
        "https://doi.org/10.1175/MWR-D-18-0187.1",
        "Gneiting, T., &amp; Raftery, A. E. (2007). Strictly proper scoring rules, "
        "prediction, and estimation. <i>Journal of the American Statistical Association</i>, "
        "102(477), 359&ndash;378. https://doi.org/10.1198/016214506000001437",
        "Br&ouml;cker, J., &amp; Smith, L. A. (2007). Increasing the reliability of "
        "reliability diagrams. <i>Weather and Forecasting</i>, 22(3), 651&ndash;661. "
        "https://doi.org/10.1175/WAF993.1",
        "Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model "
        "predictions. <i>Advances in Neural Information Processing Systems</i> 30, "
        "4765&ndash;4774. "
        "https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions",
        "Lundberg, S. M., et al. (2020). From local explanations to global understanding "
        "with explainable AI for trees. <i>Nature Machine Intelligence</i>, 2, 56&ndash;67. "
        "https://doi.org/10.1038/s42256-019-0138-9",
        "Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
        "<i>Proceedings of the 22nd ACM SIGKDD Conference</i>, 785&ndash;794. "
        "https://doi.org/10.1145/2939672.2939785",
        "Dutta, S., &amp; Pal, S. K. (2022). Interpretation of black box for short-term "
        "predictions of pre-monsoon cumulonimbus cloud events over Kolkata. <i>SN Computer "
        "Science</i>, 3, 352. https://doi.org/10.1007/s42979-022-01255-x",
        "Silva, S. J., Keller, C. A., &amp; Hardin, J. (2022). Using an explainable machine "
        "learning approach to characterize Earth system model errors: Application of SHAP "
        "analysis to modeling lightning flash occurrence. <i>Journal of Advances in Modeling "
        "Earth Systems</i>, 14, e2021MS002881. https://doi.org/10.1029/2021MS002881",
        "Daif, N., Di Nunno, F., Granata, F., et al. (2025). Forecasting maximal and minimal "
        "air temperatures using explainable machine learning: SHAP versus LIME. "
        "<i>Stochastic Environmental Research and Risk Assessment</i>, 39, "
        "2551&ndash;2581. https://doi.org/10.1007/s00477-025-02984-4",
        "Price, I., et al. (2025). Probabilistic weather forecasting with machine learning. "
        "<i>Nature</i>, 637, 84&ndash;90. https://doi.org/10.1038/s41586-024-08252-9",
        "Hersbach, H., et al. (2020). The ERA5 global reanalysis. <i>Quarterly Journal of "
        "the Royal Meteorological Society</i>, 146(730), 1999&ndash;2049. "
        "https://doi.org/10.1002/qj.3803",
        "Kiladis, G. N., Dias, J., Straub, K. H., Wheeler, M. C., Tulich, S. N., Kikuchi, K., "
        "Weickmann, K. M., &amp; Ventrice, M. J. (2014). A comparison of OLR and circulation-"
        "based indices for tracking the MJO. <i>Monthly Weather Review</i>, 142(5), "
        "1697&ndash;1715. https://doi.org/10.1175/MWR-D-13-00301.1",
        "Jarvis, A., Reuter, H. I., Nelson, A., &amp; Guevara, E. (2008). Hole-filled "
        "seamless SRTM data V4. International Centre for Tropical Agriculture (CIAT). "
        "https://srtm.csi.cgiar.org",
        "NASA Langley Research Center (2024). Prediction Of Worldwide Energy Resources "
        "(POWER) project, daily point API. "
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        "Global Administrative Areas (GADM) (2022). GADM database of Global Administrative "
        "Areas, version 4.1. https://gadm.org/",
        "Copernicus Climate Change Service (C3S) (2023). ERA5 hourly data on single levels "
        "from 1940 to present. Copernicus Climate Change Service Climate Data Store. "
        "https://cds.climate.copernicus.eu/datasets",
        "National Oceanic and Atmospheric Administration (NOAA). GEFSv12 retrospective "
        "forecast archive (reforecast), AWS Registry of Open Data. "
        "https://registry.opendata.aws/noaa-gefs-reforecast/",
    ]
    for i, r in enumerate(refs, 1):
        c.ref(f"[{i}] {r}")
    return c


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def render_content_pdf(flow: list, tmp: Path, fonts: dict[str, str]) -> None:
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont(fonts["regular"], 10)
        canvas.drawCentredString(A4[0] / 2, 30, str(canvas.getPageNumber() + 1))
        canvas.restoreState()

    doc = BaseDocTemplate(str(tmp), pagesize=A4, title="Sanket dataset report",
                          author="D N Harith, Dhanush Thota",
                          leftMargin=MARGIN, rightMargin=MARGIN, topMargin=62,
                          bottomMargin=54)
    frame = Frame(MARGIN, doc.bottomMargin, FRAME_W,
                  A4[1] - doc.topMargin - doc.bottomMargin, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])
    doc.build(flow)


def main() -> None:
    fonts = register_fonts()
    st = make_styles(fonts)
    print("rendering figures from backend/data ...")
    build_figures()

    c = build_content(st)
    c = build_content_part2(c)
    c = build_references(c)

    cover_pdf = HERE / "_cover.pdf"
    content_pdf = HERE / "_content.pdf"
    print("building cover (edits the reference PDF in place) ...")
    build_cover(cover_pdf)
    print("building content pages ...")
    render_content_pdf(c.flow, content_pdf, fonts)

    doc = pymupdf.open(cover_pdf)
    doc.insert_pdf(pymupdf.open(content_pdf))
    doc.set_metadata({
        "title": "Dataset report - Explainable AI for District-Level Prediction of "
                 "Medium-Range Weather Forecast Busts over India (Sanket)",
        "author": "D N Harith (1DS24AI012), Dhanush Thota (1DS24AI057)",
        "subject": "Assignment report: dataset description, cleaning and results",
        "keywords": "GEFSv12, ERA5, GADM, forecast bust, SHAP, SIH26079",
    })
    doc.save(OUT_PDF, garbage=4, deflate=True)
    pages = doc.page_count
    doc.close()
    for p in (cover_pdf, content_pdf):
        p.unlink(missing_ok=True)

    OUT_MD.write_text(_markdown_header + "\n".join(c.md) + "\n", encoding="utf-8")
    size = OUT_PDF.stat().st_size / 1024
    print(f"wrote {OUT_PDF.relative_to(REPO)} ({pages} pages, {size:.0f} KB)")
    print(f"wrote {OUT_MD.relative_to(REPO)} ({len(c.md)} blocks)")


_markdown_header = """# Dataset report — Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India

Sanket · Smart India Hackathon 2026 · Problem Statement SIH26079 · Ministry of Earth Sciences

D N Harith (1DS24AI012) · Dhanush Thota (1DS24AI057) · Guided by Dr. Kusumika Krori Dutta

*Generated from `build_report.py`; the PDF is the deliverable and this markdown is the same
content in a reviewable form.*

---

"""


if __name__ == "__main__":
    sys.exit(main())
