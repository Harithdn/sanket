#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The Sanket contributors (Team Winging It)
# SPDX-License-Identifier: MIT
"""Build the final IEEE-format Sanket research paper and check its source overlap.

The paper is written from the measured results in the project's earlier reports,
but its prose, tables and vector figures are newly composed. Run with --check to
compare the manuscript against those source documents using an 8-word shingle
overlap test. Rebuild with:

    python -m pip install reportlab pymupdf
    python Research_paper/Final/build_research_paper.py --check
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name(
    "Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India.pdf"
)

TITLE = "Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India"
AUTHOR_LINE = "Author(s): D N Harith, Dhanush Thota"
AFFILIATION_LINE = (
    "Affiliation(s): Department of Artificial Intelligence, "
    "Dayananda Sagar College of Engineering"
)
CORRESPONDING_LINE = (
    'Corresponding Author: Dr. Kusumika Krori Dutta, '
    '<a href="mailto:kusumika-aiml@dayanandasagar.edu" color="blue">'
    'kusumika-aiml@dayanandasagar.edu</a>'
)
DATE_LINE = "Date: November 19 2026"
KEYWORDS = [
    "forecast busts",
    "explainable artificial intelligence",
    "medium-range forecasts",
    "probabilistic calibration",
    "district-level risk",
]
KEYWORDS_LINE = "Keywords: " + "; ".join(KEYWORDS)

ABSTRACT = (
    "<b>Abstract-</b> A forecast is already a decision product, but an issued product "
    "does not say how much confidence it deserves. Medium-range failures over India are "
    "especially hard to anticipate because routine bias correction removes errors that "
    "repeat, while the damaging misses depend on the weather situation of a particular "
    "day. This paper presents Sanket as an explainable reliability layer beside numerical "
    "weather prediction. It estimates the probability that a Day 1-10 district forecast "
    "will lie in the tail of its historical error distribution and gives an attribution "
    "for each warning. NOAA GEFSv12 00 UTC reforecasts for 2000-2019 are paired with ERA5 "
    "verification, reduced to 666 GADM 4.1 districts through one area-overlap table and "
    "corrected for standing monthly bias. Seven out-of-fold XGBoost error regressors feed "
    "a regularized classifier; monotone Platt scaling calibrates its probabilities and "
    "exact TreeSHAP explains its scores. Across 365 unseen cycles in 2017 and 2,430,900 "
    "district-day-lead events, discrimination is 0.7615 ROC-AUC and 0.6784 PR-AUC; its "
    "Brier score is 0.1898, a 0.2019 skill improvement over climatology. It exceeds the "
    "best baseline that uses no current forecast by 0.0900 ROC-AUC, and its five-bin "
    "reliability errors stay within 0.022. The result is a calibrated, inspectable trust "
    "signal rather than a replacement forecast. A single full Label-v2 test year, five "
    "archived ensemble members and uncertainty in the verifying reanalysis bound the claim."
)

REFS = [
    "[1] M. J. Rodwell et al., \"Characteristics of occasional poor medium-range weather forecasts for Europe,\" Bull. Amer. Meteor. Soc., vol. 94, no. 9, pp. 1393-1405, 2013, doi: 10.1175/BAMS-D-12-00099.1.",
    "[2] D. B. Parsons et al., \"The role of continental mesoscale convective systems in forecast busts within global weather prediction systems,\" Atmosphere, vol. 10, no. 11, art. 681, 2019, doi: 10.3390/atmos10110681.",
    "[3] C. K. Potvin et al., \"Using machine learning to predict convection-allowing ensemble forecast skill: Evaluation with the NSSL Warn-on-Forecast System,\" Artif. Intell. Earth Syst., vol. 3, art. e230106, 2024, doi: 10.1175/AIES-D-23-0106.1.",
    "[4] H. Guan et al., \"GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological applications,\" Mon. Weather Rev., vol. 150, no. 3, pp. 647-665, 2022, doi: 10.1175/MWR-D-21-0245.1.",
    "[5] T. M. Hamill et al., \"The reanalysis for the Global Ensemble Forecast System, version 12,\" Mon. Weather Rev., vol. 150, no. 1, pp. 59-79, 2022, doi: 10.1175/MWR-D-21-0023.1.",
    "[6] H. Hersbach et al., \"The ERA5 global reanalysis,\" Quart. J. Roy. Meteor. Soc., vol. 146, no. 730, pp. 1999-2049, 2020, doi: 10.1002/qj.3803.",
    "[7] S. K. Roy Bhowmik and V. R. Durai, \"Development of multimodel ensemble based district level medium range rainfall forecast system for Indian region,\" J. Earth Syst. Sci., vol. 121, pp. 273-285, 2012, doi: 10.1007/s12040-012-0160-1.",
    "[8] A. Dube, R. Ashrit, H. Singh, G. Iyengar, and E. N. Rajagopal, \"Verification of medium range probabilistic rainfall forecasts over India,\" Pure Appl. Geophys., vol. 173, pp. 2489-2510, 2016, doi: 10.1007/s00024-016-1278-z.",
    "[9] P. Chakraborty, A. Sarkar, R. Bhatla, and R. Singh, \"Assessing the skill of NCMRWF global ensemble prediction system in predicting Indian summer monsoon during 2018,\" Atmos. Res., vol. 251, art. 105255, 2021, doi: 10.1016/j.atmosres.2020.105255.",
    "[10] A. Dube, S. Karunasagar, R. Ashrit, and A. K. Mitra, \"Spatial verification of ensemble rainfall forecasts over India,\" Atmos. Res., vol. 273, art. 106169, 2022, doi: 10.1016/j.atmosres.2022.106169.",
    "[11] S. Rasp and S. Lerch, \"Neural networks for postprocessing ensemble weather forecasts,\" Mon. Weather Rev., vol. 146, no. 11, pp. 3885-3900, 2018, doi: 10.1175/MWR-D-18-0187.1.",
    "[12] T. Gneiting and A. E. Raftery, \"Strictly proper scoring rules, prediction, and estimation,\" J. Amer. Stat. Assoc., vol. 102, no. 477, pp. 359-378, 2007, doi: 10.1198/016214506000001437.",
    "[13] J. Bröcker and L. A. Smith, \"Increasing the reliability of reliability diagrams,\" Weather Forecast., vol. 22, no. 3, pp. 651-661, 2007, doi: 10.1175/WAF993.1.",
    "[14] T. Gneiting, A. E. Raftery, A. H. Westveld, and T. Goldman, \"Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation,\" Mon. Weather Rev., vol. 133, no. 5, pp. 1098-1118, 2005.",
    "[15] T. Dimitriadis, T. Gneiting, and A. I. Jordan, \"Stable reliability diagrams for probabilistic classifiers,\" Proc. Natl. Acad. Sci. USA, vol. 118, no. 8, art. e2016191118, 2021.",
    "[16] C. A. T. Ferro and D. B. Stephenson, \"Extremal dependence indices: Improved verification measures for deterministic forecasts of rare binary events,\" Weather Forecast., vol. 26, no. 5, pp. 699-713, 2011.",
    "[17] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 785-794, 2016, doi: 10.1145/2939672.2939785.",
    "[18] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Adv. Neural Inf. Process. Syst., vol. 30, pp. 4765-4774, 2017.",
    "[19] S. M. Lundberg et al., \"From local explanations to global understanding with explainable AI for trees,\" Nature Mach. Intell., vol. 2, pp. 56-67, 2020, doi: 10.1038/s42256-019-0138-9.",
    "[20] S. Dutta and S. K. Pal, \"Interpretation of black box for short-term predictions of pre-monsoon cumulonimbus cloud events over Kolkata,\" SN Comput. Sci., vol. 3, art. 352, 2022, doi: 10.1007/s42979-022-01255-x.",
    "[21] S. J. Silva, C. A. Keller, and J. Hardin, \"Using an explainable machine learning approach to characterize Earth system model errors: Application of SHAP analysis to modeling lightning flash occurrence,\" J. Adv. Model. Earth Syst., vol. 14, art. e2021MS002881, 2022, doi: 10.1029/2021MS002881.",
    "[22] N. Daif, F. Di Nunno, F. Granata et al., \"Forecasting maximal and minimal air temperatures using explainable machine learning: SHAP versus LIME,\" Stoch. Environ. Res. Risk Assess., vol. 39, pp. 2551-2581, 2025, doi: 10.1007/s00477-025-02984-4.",
    "[23] I. Price et al., \"Probabilistic weather forecasting with machine learning,\" Nature, vol. 637, pp. 84-90, 2025, doi: 10.1038/s41586-024-08252-9.",
    "[24] J. Platt, \"Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,\" in Advances in Large Margin Classifiers. Cambridge, MA, USA: MIT Press, 1999, pp. 61-74.",
    "[25] R. Shanker, A. Sarkar, and A. Mamgain, \"Probabilistic verification of medium-range ensemble rainfall forecasts over India,\" Quart. J. Roy. Meteor. Soc., 2024, doi: 10.1002/qj.4674.",
    "[26] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. New York, NY, USA: Springer, 2005.",
]

# Text buckets are kept separate so citation and table/figure checks can be exact.
FRONT_TEXT: list[str] = []
BODY_TEXT: list[str] = []
CAPTION_TEXT: list[str] = []
TABLE_TEXT: list[str] = []
REFS_TEXT: list[str] = []


def plain(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


# ---------- IEEE page geometry and styles ----------
PAGE_W, PAGE_H = letter
MARGIN = 36
BOTTOM = 44
TOP = 752
GUTTER = 18
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
TITLE_H = 148
FIRST_COL_TOP = TOP - TITLE_H - 12
FIRST_COL_H = FIRST_COL_TOP - BOTTOM
BODY_COL_H = TOP - BOTTOM

styles = {}
styles["title"] = ParagraphStyle(
    "IEEE_Title", fontName="Times-Bold", fontSize=18, leading=21,
    alignment=TA_CENTER, spaceAfter=6,
)
styles["authors"] = ParagraphStyle(
    "IEEE_Authors", fontName="Times-Roman", fontSize=10.5, leading=12,
    alignment=TA_CENTER, spaceAfter=1,
)
styles["meta"] = ParagraphStyle(
    "IEEE_Meta", fontName="Times-Roman", fontSize=8.7, leading=9.8,
    alignment=TA_CENTER, spaceAfter=1,
)
styles["keywords"] = ParagraphStyle(
    "IEEE_Keywords", fontName="Times-Roman", fontSize=8.7, leading=10,
    alignment=TA_CENTER, spaceAfter=0,
)
styles["body"] = ParagraphStyle(
    "IEEE_Body", fontName="Times-Roman", fontSize=9.25, leading=11.0,
    alignment=TA_JUSTIFY, spaceAfter=4,
)
styles["abstract"] = ParagraphStyle(
    "IEEE_Abstract", parent=styles["body"], fontSize=9.0, leading=10.6,
    firstLineIndent=0, leftIndent=5, rightIndent=5, spaceAfter=5,
)
styles["h1"] = ParagraphStyle(
    "IEEE_H1", parent=styles["body"], fontName="Times-Bold", fontSize=10,
    leading=12, alignment=TA_LEFT, spaceBefore=8, spaceAfter=4,
    keepWithNext=True,
)
styles["h2"] = ParagraphStyle(
    "IEEE_H2", parent=styles["body"], fontName="Times-Bold", fontSize=9.3,
    leading=11, alignment=TA_LEFT, spaceBefore=5, spaceAfter=2,
    keepWithNext=True,
)
styles["unnum"] = ParagraphStyle(
    "IEEE_Unnum", parent=styles["h1"], spaceBefore=9,
)
styles["caption"] = ParagraphStyle(
    "IEEE_Caption", parent=styles["body"], fontSize=8.0, leading=9.4,
    alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=5,
)
styles["tablecap"] = ParagraphStyle(
    "IEEE_TableCap", parent=styles["body"], fontName="Times-Bold", fontSize=8.0,
    leading=9.4, alignment=TA_CENTER, spaceBefore=5, spaceAfter=3,
)
styles["cell"] = ParagraphStyle(
    "IEEE_Cell", fontName="Times-Roman", fontSize=7.4, leading=8.7,
    alignment=TA_LEFT, spaceAfter=0,
)
styles["cellb"] = ParagraphStyle(
    "IEEE_CellBold", parent=styles["cell"], fontName="Times-Bold",
)
styles["cellr"] = ParagraphStyle(
    "IEEE_CellRight", parent=styles["cell"], alignment=TA_CENTER,
)
styles["ref"] = ParagraphStyle(
    "IEEE_Ref", parent=styles["body"], fontSize=8.0, leading=9.5,
    leftIndent=12, firstLineIndent=-12, alignment=TA_LEFT, spaceAfter=2.5,
)
styles["bullet"] = ParagraphStyle(
    "IEEE_Bullet", parent=styles["body"], leftIndent=11, firstLineIndent=-5,
    spaceAfter=2,
)

INK = colors.HexColor("#1A1A1A")
BLUE = colors.HexColor("#486A88")
LIGHT = colors.HexColor("#EEF2F5")
MID = colors.HexColor("#9AA8B2")
GRID = colors.HexColor("#B8C1C8")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(PAGE_W / 2, 24, str(doc.page))
    canvas.restoreState()


def add_front(story, text, style="meta"):
    story.append(Paragraph(text, styles[style]))
    FRONT_TEXT.append(plain(text))


def add(story, text, style="body"):
    story.append(Paragraph(text, styles[style]))
    BODY_TEXT.append(plain(text))


def add_caption(story, text, style="caption"):
    story.append(Paragraph(text, styles[style]))
    CAPTION_TEXT.append(plain(text))


def add_ref(story, text):
    story.append(Paragraph(text, styles["ref"]))
    REFS_TEXT.append(plain(text))


def add_table(story, caption, rows, widths, header=True, align_right=()):
    add_caption(story, caption, "tablecap")
    data = []
    for ridx, row in enumerate(rows):
        cells = []
        for cidx, value in enumerate(row):
            cellstyle = "cellb" if (header and ridx == 0) else (
                "cellr" if cidx in align_right else "cell"
            )
            cells.append(Paragraph(str(value), styles[cellstyle]))
            TABLE_TEXT.append(plain(str(value)))
        data.append(cells)
    table = Table(data, colWidths=widths, hAlign="CENTER", splitByRow=1)
    cmds = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]
    table.setStyle(TableStyle(cmds))
    story.append(KeepTogether([table, Spacer(1, 3)]))


def label(draw, x, y, text, size=6, bold=False, anchor="start", color=INK):
    draw.add(String(
        x, y, text, fontName="Times-Bold" if bold else "Times-Roman",
        fontSize=size, fillColor=color, textAnchor=anchor,
    ))


def box(draw, x, y, w, h, text, fill=LIGHT, size=6):
    draw.add(Rect(x, y, w, h, fillColor=fill, strokeColor=MID, strokeWidth=0.6))
    label(draw, x + w / 2, y + h / 2 - 2, text, size=size, bold=True, anchor="middle")


def down_arrow(draw, x, y_top, y_bottom):
    draw.add(Line(x, y_top, x, y_bottom + 4, strokeColor=BLUE, strokeWidth=1.2))
    draw.add(Polygon(
        [x - 3, y_bottom + 4, x + 3, y_bottom + 4, x, y_bottom],
        fillColor=BLUE, strokeColor=BLUE, strokeWidth=0.5,
    ))


def pipeline_figure():
    d = Drawing(COL_W, 140)
    box(d, 22, 113, 216, 19, "GEFSv12 forecasts + ERA5 verification")
    down_arrow(d, 130, 112, 99)
    box(d, 22, 80, 216, 19, "One polygon-cell district weight table")
    down_arrow(d, 130, 79, 66)
    box(d, 22, 47, 216, 19, "Training-only bias removal and p90 labels")
    down_arrow(d, 130, 46, 33)
    box(d, 22, 14, 216, 19, "Stage 1/2 XGBoost, Platt scaling, TreeSHAP")
    label(d, 130, 4, "Issue-time input contract at every stage", 5.4, anchor="middle")
    return d


def auc_figure():
    d = Drawing(COL_W, 142)
    x0, x1 = 96, 246
    lo, hi = 0.45, 0.80
    def sx(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    rows = [
        ("Climatology", 0.5000),
        ("Spread only", 0.5996),
        ("EMOS", 0.6256),
        ("District-season", 0.6715),
        ("Pre-audit v1", 0.6572),
        ("Sanket v2", 0.7615),
    ]
    label(d, 130, 131, "Held-out 2017 ROC-AUC", 7, bold=True, anchor="middle")
    for tick in (0.5, 0.6, 0.7, 0.8):
        x = sx(tick)
        d.add(Line(x, 20, x, 116, strokeColor=GRID, strokeWidth=0.35))
        label(d, x, 15, f"{tick:.1f}", 5.2, anchor="middle")
    d.add(Line(x0, 20, x1, 20, strokeColor=INK, strokeWidth=0.7))
    y = 108
    for name, value in rows:
        label(d, 91, y + 2, name, 5.6, anchor="end")
        color = BLUE if name == "Sanket v2" else MID
        d.add(Rect(x0, y, sx(value) - x0, 8, fillColor=color, strokeColor=color, strokeWidth=0.3))
        label(d, sx(value) + 3, y + 1.5, f"{value:.4f}", 5.2)
        y -= 15
    label(d, 171, 4, "ROC-AUC (higher is better)", 5.4, anchor="middle")
    return d


def reliability_figure():
    d = Drawing(COL_W, 158)
    x0, y0, size = 40, 30, 115
    x1, y1 = x0 + size, y0 + size
    label(d, 130, 149, "Five-bin reliability on 2.43M events", 7, bold=True, anchor="middle")
    for v in (0, 0.5, 1):
        x = x0 + v * size
        y = y0 + v * size
        d.add(Line(x, y0, x, y1, strokeColor=GRID, strokeWidth=0.3))
        d.add(Line(x0, y, x1, y, strokeColor=GRID, strokeWidth=0.3))
        label(d, x, y0 - 10, f"{v:.1f}", 5.2, anchor="middle")
        label(d, x0 - 5, y - 2, f"{v:.1f}", 5.2, anchor="end")
    d.add(Line(x0, y0, x1, y1, strokeColor=MID, strokeWidth=0.8))
    points = [
        (0.1308, 0.1162), (0.2986, 0.2870), (0.4955, 0.4907),
        (0.6800, 0.6890), (0.8810, 0.9027),
    ]
    for pred, obs in points:
        px, py = x0 + pred * size, y0 + obs * size
        d.add(Circle(px, py, 2.6, fillColor=BLUE, strokeColor=colors.white, strokeWidth=0.6))
    # Draw a polyline as four connected line segments.
    for i in range(len(points) - 1):
        xa, ya = x0 + points[i][0] * size, y0 + points[i][1] * size
        xb, yb = x0 + points[i + 1][0] * size, y0 + points[i + 1][1] * size
        d.add(Line(xa, ya, xb, yb, strokeColor=BLUE, strokeWidth=1.1))
    label(d, x0 + size / 2, 8, "Mean predicted probability", 5.8, anchor="middle")
    label(d, 9, y0 + size / 2 + 4, "Observed", 5.0, anchor="middle")
    label(d, 9, y0 + size / 2 - 3, "frequency", 5.0, anchor="middle")
    return d


def add_figure(story, drawing, caption):
    drawing.hAlign = "CENTER"
    story.append(KeepTogether([drawing, Paragraph(caption, styles["caption"]), Spacer(1, 2)]))
    CAPTION_TEXT.append(plain(caption))


# ---------- Paper content ----------
def build_story():
    story: list = []

    add_front(story, TITLE, "title")
    add_front(story, AUTHOR_LINE, "authors")
    add_front(story, AFFILIATION_LINE)
    add_front(story, CORRESPONDING_LINE)
    add_front(story, DATE_LINE)
    add_front(story, KEYWORDS_LINE, "keywords")
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=0.6, color=MID))
    story.append(NextPageTemplate("Body"))
    story.append(FrameBreak())

    add(story, ABSTRACT, "abstract")

    add(story, "1. INTRODUCTION", "h1")
    add(story, "1.1 Background and context", "h2")
    add(story, (
        "A medium-range forecast is already used as a decision product. District officials, "
        "farmers, transport planners and emergency teams act on it before the weather is "
        "known, yet a small set of forecasts fails badly enough to make those actions unsafe. "
        "European studies of day-6 upper-air failures showed that such misses follow particular "
        "flow regimes and are only partly visible in ensemble spread [1], [2]. More recent work "
        "on convection-allowing ensembles showed that forecast skill can itself be learned "
        "from information available before verification [3]. Sanket asks the narrower question "
        "a duty forecaster faces at 6 a.m.: can the chance of a serious error be estimated "
        "while the forecast is still useful? The Indian setting makes that question harder, "
        "because monsoon rainfall, Himalayan terrain, coastal transitions and island districts "
        "cannot be represented well by one national average."
    ))
    add(story, (
        "The data needed for a retrospective answer exist. GEFSv12 supplies a long, consistent "
        "00 UTC reforecast archive [4] with a documented initialization reanalysis [5], while "
        "ERA5 supplies an independent gridded verification record [6]. India already has "
        "district-scale multi-model rainfall guidance [7], but that guidance predicts weather "
        "amounts rather than the trustworthiness of an issued forecast. This paper therefore "
        "treats forecast failure as a second-order prediction problem and restricts every "
        "predictor to information that would have existed at issue time."
    ))

    add(story, "1.2 Problem statement and research gap", "h2")
    add(story, (
        "Indian verification studies have characterized probabilistic rainfall skill, monsoon "
        "circulation and spatial displacement error [8], [9], [10]. Post-processing research "
        "improves the distribution of a weather variable [11], while scoring theory separates "
        "calibration from discrimination and gives proper rules for comparing probabilities "
        "[12], [13]. EMOS provides a calibrated statistical reference [14]; stable reliability "
        "diagrams [15] and extremal-dependence scores [16] expose behaviour that a single "
        "accuracy figure hides. Tree boosting [17], SHAP [18], [19] and weather applications "
        "of local explanations [20], [21], [22] make model reasoning visible. Even a modern "
        "probabilistic forecast engine such as GenCast [23] produces weather fields, not a "
        "district-level confidence statement about a separate forecast that has already "
        "been issued."
    ))
    add(story, (
        "What is missing is the integration of these strands into one operational artifact: a "
        "calibrated, districtwise probability that a Day 1-10 forecast will fail across "
        "several surface variables, paired with a local explanation and validated against a "
        "credible ladder of simpler methods. The earlier Sanket reports documented data "
        "preparation and model performance separately. A citable research account that joins "
        "the formulation, leakage controls, comparative evidence, interpretation and "
        "reproducibility has not yet been assembled."
    ))

    add(story, "1.3 Research questions, objectives, and hypotheses", "h2")
    add(story, (
        "<b>RQ1.</b> Can issue-time ensemble, lagged-forecast, geographic and seasonal "
        "predictors identify tail-error events more accurately than climatology, spread-only "
        "rules, EMOS and other post-processing baselines?"
    ), "bullet")
    add(story, (
        "<b>RQ2.</b> Does a two-stage out-of-fold design, followed by probability calibration, "
        "produce both useful discrimination and honest reliability, with attributions that "
        "remain meteorologically interpretable?"
    ), "bullet")
    add(story, (
        "<b>H1.</b> After standing bias and post-issue information are removed, learned error "
        "estimates will add measurable skill over forecast-free historical frequency and "
        "single-variable uncertainty proxies, without materially degrading calibration."
    ), "bullet")

    add(story, "1.4 Contribution and scope", "h2")
    add(story, (
        "The paper makes four contributions. First, it reframes a forecast bust over India as "
        "a supervised reliability target defined from bias-corrected, variable-specific tail "
        "errors. Second, it describes a common spatial aggregation and a chronological split "
        "that keep observations, thresholds and fitted transformations out of training-to-test "
        "paths. Third, it evaluates a two-stage explainable model against eight baselines and "
        "a spatial deep-learning challenger on identical held-out cycles. Fourth, it reports "
        "calibration, uncertainty, feature attribution and a documented leakage audit together, "
        "so the reader can see both the skill and the boundaries of the claim. The scope is "
        "deliberately limited: Sanket does not generate a new weather forecast, establish "
        "causal mechanisms, or claim prospective operational performance with the full "
        "31-member live ensemble."
    ))

    add(story, "1.5 Paper structure", "h2")
    add(story, (
        "Section 2 reviews bust diagnosis, verification and explainability. Section 3 gives "
        "the data, preparation, models and evaluation protocol. Section 4 answers the two "
        "research questions. Section 5 interprets the findings and states limitations and "
        "next steps. Section 6 closes the paper; declarations, references and a short "
        "reproducibility appendix follow."
    ))

    add(story, "2. LITERATURE REVIEW", "h1")
    add(story, "2.1 Forecast failures and the prediction of forecast skill", "h2")
    add(story, (
        "The classical bust literature is diagnostic rather than prospective. Rodwell and "
        "colleagues composited European day-6 height errors and linked them to upstream "
        "flow development [1]; Parsons and colleagues showed how mesoscale convective "
        "systems can amplify downstream circulation errors [2]. Both studies explain why a "
        "forecast failed after the failure is observed. Potvin and colleagues moved closer "
        "to operations by learning ordinal skill classes for convection-allowing ensembles "
        "from a large predictor set [3], but the target remained regional storm-scale "
        "forecast quality. Neural post-processing [11] and probabilistic ML forecasting [23] "
        "improve the forecast distribution itself. They do not, by themselves, tell a local "
        "user whether the specific district guidance on the screen should be downweighted."
    ))
    add(story, (
        "This distinction matters in India. A skilful national model can still produce a "
        "bad district rainfall forecast during an active monsoon phase, and a coarse "
        "verification score may hide the affected district. A reliability layer therefore "
        "needs to learn from the paired forecast-observation history at the same spatial "
        "unit on which decisions are made."
    ))

    add(story, "2.2 Probabilistic verification and Indian regional evidence", "h2")
    add(story, (
        "Indian studies have established that forecast quality depends on lead, threshold, "
        "season and verifying product [8], [9], [10]. Spatial verification further shows that "
        "rainfall displacement and pattern errors dominate simple amplitude error in many "
        "monsoon cases [10]. These findings argue for district-level labels and season-aware "
        "evaluation, but they also warn that a district mean can conceal structured spatial "
        "error. Work on multi-model district guidance [7] demonstrates operational delivery "
        "at this scale while retaining a weather quantity as the output."
    ))
    add(story, (
        "Verification theory supplies the second requirement. Proper scores reward honest "
        "probabilities [12]; decomposition separates miscalibration, discrimination and "
        "irreducible base-rate uncertainty [13]. EMOS [14], CORP reliability diagrams [15] "
        "and extremal-dependence indices [16] offer complementary views of a probabilistic "
        "classifier. Reporting ROC-AUC alone would be inadequate because a ranking metric is "
        "unchanged by monotone probability errors. This paper therefore reports Brier score, "
        "Brier skill score, reliability bins, thresholded counts and a cycle-block bootstrap "
        "beside discrimination."
    ))

    add(story, "2.3 Conceptual framework for explainable reliability", "h2")
    add(story, (
        "The framework treats error as a random variable conditioned on evidence available "
        "when a forecast cycle is issued. A variable-specific regressor first estimates the "
        "size of the bias-corrected absolute error; a classifier then maps those estimates "
        "and other issue-time context to P(bust | issue-time evidence). Gradient-boosted "
        "trees [17] suit heterogeneous tabular predictors and native missingness. Monotone "
        "Platt scaling [24] maps classifier logits to observed frequencies, while exact "
        "TreeSHAP [18], [19] decomposes each score into additive contributions in log-odds "
        "space. Weather-focused XAI has explained thunderstorm nowcasts [20], Earth-system "
        "model errors [21] and multi-horizon temperatures [22]; the present target is the "
        "failure probability of a separate NWP product, not the weather value."
    ))
    add(story, (
        "An explanation is useful here only if it can be questioned. Correlated meteorological "
        "inputs can split attribution, and an apparently important feature may conceal a "
        "data leak upstream. SHAP is therefore treated as an audit and communication aid, "
        "not as causal evidence. Code-level input contracts, sensitivity tests and an honest "
        "issue-time replay are part of the methodology."
    ))

    add(story, "2.4 Summary and gap identification", "h2")
    add(story, (
        "Three conclusions recur across this literature. Ensemble spread carries information "
        "about error but does not exhaust it [1], [3]. A trustworthy probability needs both "
        "discrimination and calibration [12], [15]. Local explanations become scientifically "
        "useful when they are checked against physical knowledge and operational timing [19], "
        "[21]. The unoccupied space is a district-scale Indian system that predicts a tail "
        "error before verification, calibrates the warning, explains it locally and proves "
        "that no post-issue observation entered the decision. Sanket is designed to occupy "
        "that space."
    ))

    add(story, "3. METHODOLOGY", "h1")
    add(story, "3.1 Research design", "h2")
    add(story, (
        "This is a retrospective, quantitative computational study. The unit of analysis is "
        "a district-initialization-lead event, and the dependent variable is binary. The "
        "design compares a proposed reliability pipeline with statistical baselines under "
        "identical chronological splits, then checks calibration, temporal robustness, "
        "sensitivity and explanation behaviour. Fig. 1 shows the main path. A convolutional "
        "network over raw gridded fields is retained as a challenger, not as the deployed "
        "model, to test whether spatial gradients add information beyond engineered district "
        "features."
    ))
    add_figure(
        story, pipeline_figure(),
        "Fig. 1. Leakage-controlled Sanket pipeline. Forecast and verification fields share "
        "one district aggregation; labels and transformations are fitted on training cycles "
        "only; predictions and attributions use issue-time inputs."
    )

    add(story, "3.2 Data sources and sample", "h2")
    add(story, (
        "The forecast side is the NOAA GEFSv12 00 UTC reforecast archive for 2000-2019 on a "
        "0.25 degree grid, with five archived members per cycle [4], [5]. ERA5 hourly "
        "reanalysis [6] provides district-day verification. GADM 4.1 defines 666 districts; "
        "CGIAR-CSI SRTM elevation supplies static terrain descriptors. A single 10,373-pair "
        "polygon-cell overlap table converts both forecast and verification grids to area "
        "means, including small districts that contain no grid centre. Seven surface "
        "variables define the Label-v2 event: rainfall, 2 m temperature, relative humidity, "
        "surface pressure, atmospheric moisture, 10 m wind speed and topsoil moisture. Wind "
        "direction remains an input but is excluded from the label because a bearing error "
        "wraps around the circle. In this archive, 10 m wind is available only through Day 5 "
        "and topsoil moisture stops at Day 3. The gridded challenger alone uses Z500, CAPE "
        "and CIN."
    ))
    add(story, (
        "Across the archive, 7,288 daily cycles were ingested, yielding 242,690,400 "
        "district-member-lead rows. The promoted run uses 2000-2017; a compact 17-cycle, 36-station 2019 sample "
        "with 30,600 forecast rows and 13,680 ERA5 rows supports reproducible checks. Table I "
        "summarizes the principal partitions."
    ))
    add_table(
        story,
        "TABLE I. DATA PARTITIONS AND VERIFICATION SCALE",
        [
            ["Item", "Value"],
            ["Archive", "2000-2019; 7,288 cycles; 666 districts; 5 members"],
            ["Training pool", "2000-2015; 5,828 cycles; 13.32M baseline-fit events"],
            ["Stage-1 fit", "2,000-cycle fixed-seed subsample; five 30-day OOF blocks"],
            ["Validation", "2016; 365 cycles; early stopping, Platt and conformal fit"],
            ["Held-out test", "2017; 365 cycles; 2,430,900 events; 38.95% busts"],
            ["Boundary rule", "10-day embargo on each split and OOF block edge"],
        ],
        [74, 187],
    )

    add(story, "3.3 Instruments, materials, and tools", "h2")
    add(story, (
        "The implementation uses Python with PyArrow/Parquet storage, SQLite lineage, XGBoost "
        "for tree models, SHAP for exact tree attribution, scikit-learn for classical baselines "
        "and PyTorch with ONNX export for the convolutional challenger. Physical ranges reject "
        "implausible member-days; completeness checks quarantine short cycles; a named "
        "observation-derived contract blocks realized errors and late observation batches from "
        "model inputs. FastAPI serves precomputed responses on a 512 MB instance, so training "
        "and serving stores remain separate. Source licences are recorded per dataset in the "
        "repository's REUSE annotations."
    ))

    add(story, "3.4 Data preparation and collection procedure", "h2")
    add(story, (
        "GRIB2 and NetCDF inputs are parsed into one canonical schema with UTC-day convention "
        "checks. Humidity and vector wind are derived only after the daily district mean has "
        "been formed; this avoids an averaging-order mismatch between forecast and truth. "
        "Both fields pass through the same overlap weights. Provisional "
        "observation rows lose to final rows on the canonical key. Soil moisture is masked on "
        "both sides for 18 coastal districts where a coarse land-sea mask would otherwise "
        "turn ocean cells into false busts. Structural horizon gaps and missing lagged features "
        "remain explicit NaN values; the tree models consume them without imputation, and "
        "natural class priors are retained."
    ))
    add(story, (
        "A monthly bias table indexed by district, variable, lead and month is fitted on "
        "training cycles only, with a documented backoff for sparse cells. For district d, "
        "variable v and lead l, e = |mu - y - b|, where mu is the five-member forecast mean, "
        "y is the ERA5 district mean and b is the standing bias. The threshold tau_v is the "
        "90th percentile of training e for that variable. An event is labelled 1 when at least "
        "one of the seven retained variables exceeds its threshold. Predictors include spread, "
        "member count, lagged-ensemble pooling, cycle-to-cycle revisions, pressure and moisture "
        "tendencies, concurrent forecasts, geography, season and historical district frequency. "
        "All splits use initialization dates; 2000-2015 trains, 2016 validates and 2017 tests, "
        "with a 10-day embargo at every boundary."
    ))

    add(story, "3.5 Models and data analysis", "h2")
    add(story, (
        "Stage 1 fits one histogram XGBoost regressor per variable over 32 issue-time "
        "predictors. Each produces a predicted absolute error and a clipped confidence term. "
        "Five contiguous 30-day out-of-fold blocks supply these estimates inside training, "
        "so the classifier never sees a regressor's own fitted row. Stage 2 combines 14 "
        "Stage-1 outputs with 52 spread, lagged, geographic and seasonal features in a "
        "shallow regularized XGBoost classifier. Validation logits are transformed by "
        "monotone Platt scaling [24], p' = sigma(a logit(p) + b), with the fitted values "
        "a = 0.882 and b = -0.291; a non-positive slope would be rejected. Risk cuts follow "
        "the 70th and 90th validation percentiles, corresponding to 45% Watch and 81% Bust "
        "in the promoted run."
    ))
    add(story, (
        "The spatial challenger is a four-block dilated CNN with 43,969 parameters over "
        "145 x 141 x 48 tensors, followed by parameter-free district pooling. It is compared "
        "with the tabular model on the same cycles. Evaluation uses ROC-AUC, PR-AUC, Brier "
        "score, Brier skill score against training climatology, CORP decomposition, SEDI, "
        "thresholded counts and the NCMRWF binormal Z-AUC [25]. Uncertainty is estimated "
        "with 1,000 circular block-bootstrap resamples over 30-cycle groups rather than "
        "independent rows, because districts in one cycle share a synoptic situation. Split "
        "conformal sets at alpha = 0.10 follow the calibration logic of [26]. Exact "
        "TreeSHAP values are computed in log-odds space for Stage 2 and for stratified "
        "Stage-1 samples."
    ))

    add(story, "3.6 Ethical considerations", "h2")
    add(story, (
        "The study uses environmental reforecast and reanalysis data and involves no human "
        "participants or private records. Third-party licences and attribution obligations "
        "are preserved; GADM geometry is used only within its academic terms. District "
        "boundaries can be politically sensitive, so the map is treated as an analytical "
        "unit, not a policy statement. Because a warning can change public behaviour, the "
        "score is designed for human review by trained forecasters and is not presented as "
        "an official alert. Seasonal, coastal and high-altitude error slices are reported "
        "to discourage a uniform interpretation of a national average."
    ))

    add(story, "3.7 Limitations of the method", "h2")
    add(story, (
        "The primary Label-v2 experiment has one full held-out year, the reforecast archive "
        "contains only five of 31 operational members, and ERA5 is an imperfect rainfall "
        "reference. Some variables stop before Day 10, national thresholds may not fit steep "
        "terrain, and pre-2017 revision features are sparse. Validation supports early "
        "stopping, calibration and conformal sizing, which is convenient but not equivalent "
        "to three independent years. Finally, SHAP attributions are associative and "
        "correlated predictors can share credit. Section 5.4 expands these points."
    ))

    add(story, "4. RESULTS", "h1")
    add(story, "4.1 Descriptive findings", "h2")
    add(story, (
        "The held-out 2017 set contains 946,820 positive events out of 2,430,900, a 38.95% "
        "base rate. The label is not uniform across the year: the aggregate bust rate is "
        "48.6% in June-September and 28.5% in the calmer post-monsoon and winter regimes, "
        "while the rainfall error threshold rises from 4.0 mm/day outside the monsoon to "
        "15.9 mm/day inside it. Removing standing bias lowers the November humidity p90 "
        "from 27.6 to 10.4 percentage points, a useful reminder that a large raw error is "
        "not always a surprising one. Table I states the evaluation scale; Fig. 1 clarifies "
        "where these labels were fitted."
    ))

    add(story, "4.2 Findings for Research Question 1", "h2")
    add(story, (
        "Table II and Fig. 2 answer the comparison question. For the promoted Label-v2 "
        "model, discrimination is 0.7615 ROC-AUC and 0.6784 PR-AUC; its Brier score is "
        "0.1898, its skill gain over climatology is 0.2019, and F1 is 0.5819. Climatology "
        "sits at 0.5000 ROC-AUC; spread-only logistic "
        "regression reaches 0.5996 and EMOS 0.6256. A historical district-season frequency "
        "table reaches 0.6715 without reading a single current forecast, which is a strong "
        "reminder of how much geography and monsoon seasonality structure the problem. "
        "Sanket v2 still improves on that baseline by 0.0900 ROC-AUC and on EMOS by 0.1359. "
        "The pre-audit model scored honestly at issue time reaches 0.6572; its offline "
        "0.8435 value used a realized-error lag and is invalid for operations."
    ))
    add_table(
        story,
        "TABLE II. HELD-OUT 2017 DISCRIMINATION AND PROBABILISTIC SCORES",
        [
            ["Method", "ROC-AUC", "Brier", "BSS"],
            ["Climatology", "0.5000", "0.2378", "0.0000"],
            ["Spread only", "0.5996", "-", "-"],
            ["EMOS", "0.6256", "-", "-"],
            ["District-season frequency", "0.6715", "-", "-"],
            ["Pre-audit v1, issue time", "0.6572", "0.2342", "+0.0151"],
            ["Sanket v2 + Platt", "0.7615", "0.1898", "+0.2019"],
        ],
        [112, 52, 49, 48],
        align_right=(1, 2, 3),
    )
    add_figure(
        story, auc_figure(),
        "Fig. 2. ROC-AUC for selected 2017 baselines and the promoted model. The complete "
        "eight-rung ladder also includes lead-day, IDR, lead-spread-season and analog rules."
    )
    add(story, (
        "The uncertainty estimate changes how small gaps should be read. A 30-cycle circular "
        "block bootstrap gives a 95% ROC-AUC interval 0.0438 wide, about 4.5 times the "
        "0.0098 width produced by resampling single cycles; gaps below roughly 0.022 are "
        "therefore not stable multi-week evidence. An ablation on the small 2019 sample "
        "shows why Stage 1 matters: without its 14 out-of-fold inputs, ROC-AUC falls to "
        "0.6056 from 0.6447 and Brier score rises from 0.2291 to 0.2351. A random forest reaches "
        "0.6798 on those 1,050 events, but the difference is within the small-sample "
        "cycle-bootstrap interval; at the full archive scale, native missingness and "
        "heterogeneous district features favour boosted trees. Stage 1 reduces held-out "
        "error MAE by 41.9% for rainfall, 25.4% for wind speed, 16.4% for column moisture "
        "and 11.6% for humidity relative to a training-mean predictor, while temperature "
        "and pressure show little residual skill after bias removal."
    ))
    add(story, (
        "The spatial challenger does not overturn the tabular result. On identical 2018 "
        "cycles, three-seed BustCNN reaches 0.7454 ROC-AUC versus 0.8711 for two-stage "
        "XGBoost; the 2017 gap is similar. About 2,550 grid-day training frames give a "
        "43,969-parameter CNN less to learn from than millions of district-day rows give "
        "the engineered tree model. Multi-year experiments on unseen 2019 events raise "
        "ROC-AUC from 0.8231-0.8327 with one training year to 0.8487 with a three-year "
        "pool, but August falls to 0.741 versus 0.870 in November. Those figures support "
        "H1 while showing that monsoon convection remains the hardest regime."
    ))

    add(story, "4.3 Findings for Research Question 2", "h2")
    add(story, (
        "Calibration is close enough for the probabilities to be read as frequencies. Across "
        "five bins, the largest gap between mean predicted probability and observed bust "
        "frequency is 0.022; the high-risk bin predicts 0.8810 and observes 0.9027 across "
        "100,143 alerts. Fig. 3 shows the five points against the diagonal. At a 0.50 "
        "threshold, the model produces 1,225,647 true negatives, 258,433 false positives, "
        "452,201 false negatives and 494,619 true positives: sensitivity 0.5224, specificity "
        "0.8259, accuracy 0.7077 and F1 0.5819. Compared with the honestly scored pre-audit "
        "model, true positives nearly double and precision rises from 0.5814 to 0.6568. "
        "Split conformal sets on the 2019 sample attain 0.833 empirical coverage against a "
        "0.90 target, so their uncertainty needs more calibration cycles."
    ))
    add_figure(
        story, reliability_figure(),
        "Fig. 3. Reliability of five probability bins over 2,430,900 unseen events from 2017. "
        "The points lie close to the perfect-calibration diagonal; the largest absolute "
        "deviation is 0.0217."
    )
    add(story, (
        "The attributions are physically legible but not uniform. The largest mean absolute "
        "Stage-2 SHAP values are region-season historical frequency (0.320), state identifier "
        "(0.254), month (0.241), lagged rainfall-pool mean (0.146), humidity confidence "
        "(0.107), predicted humidity error (0.098), lagged pressure-pool mean (0.088) and "
        "lagged moisture-pool spread (0.069). Inside Stage 1, forecast amount and lagged "
        "rainfall pooling dominate rainfall error, while ensemble spread leads column "
        "moisture. When the predicted rain is below 2 mm/day, only 0.5-0.7% of cases "
        "become rainfall busts. If Stage 1 expects an error above 9 mm/day, however, "
        "49-58% of those cases do verify as busts."
    ))
    add(story, (
        "A historical replay illustrates the intended use without being treated as statistical "
        "proof. For the 13 August 2018 00 UTC cycle at Day 3 over Idukki, GEFSv12 forecast "
        "35.3 mm of rain against 145.0 mm in ERA5 during the Kerala floods; the model assigns "
        "a 0.71 issue-time bust probability. The audit also gives a caution: classifier-level "
        "SHAP credited a confidence feature when a realized-error lag had leaked into Stage 1. "
        "Explanations helped diagnose the symptom, but a code-level input contract removed "
        "the cause."
    ))

    add(story, "5. DISCUSSION", "h1")
    add(story, "5.1 Interpretation of key findings", "h2")
    add(story, (
        "RQ1 and H1 receive qualified support. Forecast-free history already explains a large "
        "share of where and when Indian guidance fails, but the learned Stage-1 error estimates "
        "add a further, reproducible gain on a held-out year. The gain is not a claim that the "
        "model sees the future; it comes from ensemble trajectories, run-to-run revisions and "
        "cross-variable context that are available when the forecast is issued. RQ2 is also "
        "supported: discrimination improves without the over-confidence seen in the earlier "
        "deployment, and the leading attributions align with season, region and forecast "
        "consistency. A probability near 0.71 should still be read as a request to inspect "
        "the forecast, not as a deterministic verdict."
    ))

    add(story, "5.2 Comparison with existing research", "h2")
    add(story, (
        "The surface-error target differs from the synoptic height anomalies used in European "
        "bust studies [1], [2], because district impacts depend on rain, heat, humidity and "
        "wind near the ground. Unlike forecast-of-forecast-skill work at storm scale [3], the "
        "present output spans ten leads and 666 administrative units. It also differs from "
        "post-processing [11], [14], which adjusts the weather distribution rather than "
        "flagging low confidence in an existing product. Indian verification [8], [9], [10] "
        "supplies the caution that the verifying rainfall product and spatial displacement "
        "matter; our sensitivity checks address part, but not all, of that concern. Finally, "
        "the XAI target is model error [21], not the predicted weather value [20], [22]. "
        "Removing persistent bias makes the task harder than a raw-error formulation, but it "
        "also makes the label describe a genuine day-specific failure."
    ))

    add(story, "5.3 Theoretical and practical implications", "h2")
    add(story, (
        "The results support a separation between prediction and reliability assessment. A "
        "forecast centre can keep its NWP system and statistical correction while adding a "
        "second model that learns when the corrected product is likely to fail. In practice, "
        "the probability can rank district-lead entries for review, and the local attribution "
        "can tell a forecaster whether the warning came mainly from a vulnerable season and "
        "place, a volatile ensemble trajectory, or a large estimated error in one variable. "
        "Precomputing scores and explanations allows the dashboard to run within a 512 MB "
        "serving budget. A candidate is promoted only if it clears 0.55 ROC-AUC and gives up "
        "no more than 0.05 ROC-AUC relative to the current served model; that gate is "
        "scientific as well as operational."
    ))

    add(story, "5.4 Limitations", "h2")
    add(story, (
        "The strongest number in this paper comes from one complete Label-v2 test year. "
        "Cross-year 0.833-0.849 ROC-AUC experiments on 2018 and 2019 use the earlier label "
        "definition, so they demonstrate pipeline stability rather than a fully rescored "
        "Label-v2 generalization result. Five archived members give a noisy spread estimate "
        "with a coefficient of variation near 50%, and the operational 31-member feed is not "
        "part of this retrospective test. ERA5 and MERRA-2 differ by 23.5-43.5% of a bust "
        "threshold across Indian city-days; during the December 2015 Chennai flood, ERA5 "
        "recorded about 57 mm where a gauge estimate was near 490 mm. Any label built on one "
        "reanalysis inherits such limits."
    ))
    add(story, (
        "Because wind stops at Day 5 and soil at Day 3, the number of variables eligible "
        "to trigger the label changes mechanically. Nationally pooled thresholds leave "
        "high-altitude districts exposed: Leh shows 5.32 hPa pressure error against a "
        "0.555 hPa national median. Sparse pre-2017 backfills leave revision features mostly "
        "missing in training, Stage-1 regressors use 2,000 of 5,828 available cycles to fit "
        "within memory limits, and 2016 is reused for several calibration tasks. Served SHAP "
        "panels are pre-aggregated rather than computed per request. Attributions remain "
        "associative, and the small conformal sample under-covers its nominal 90% target. "
        "These are measured constraints, not rhetorical disclaimers."
    ))

    add(story, "5.5 Directions for future research", "h2")
    add(story, (
        "The next model should give each variable its own probability head, beginning with "
        "monsoon rainfall, so staggered horizon caps no longer perturb an aggregate label. "
        "A full Label-v2 and ERA5-estimator-v2 rescoring of 2018 and 2019, with UTC-aligned "
        "IMD gauge rainfall and IMDAA as alternate truth, would test both temporal and "
        "observation-product transfer. District-stratified quantiles would address terrain "
        "bias; all 31 operational members and real-time MJO indices would sharpen uncertainty "
        "features. Correlated-feature grouping should be used before interpreting SHAP "
        "rankings, and a prospective study with duty forecasters should measure whether the "
        "alerts change decisions in ways that relative economic value can quantify."
    ))

    add(story, "6. CONCLUSION", "h1")
    add(story, (
        "Sanket addresses a practical gap left by forecast production: knowing when an issued "
        "medium-range district forecast deserves less confidence. By defining busts from "
        "bias-corrected tail errors, enforcing a strict issue-time contract, and combining "
        "out-of-fold error regressors with a calibrated classifier and exact tree "
        "attributions, it turns a diagnostic problem into an inspectable probability. On "
        "2,430,900 unseen 2017 events, the model reaches 0.7615 ROC-AUC, 0.6784 PR-AUC and "
        "0.1898 Brier score, with reliability deviations no greater than 0.022. It beats "
        "the full baseline ladder and a spatial CNN challenger, while the audit shows how "
        "easily a realized-error lag can manufacture apparent skill."
    ))
    add(story, (
        "The broader lesson is modest but useful: a forecast and a measure of confidence in "
        "that forecast are different services, and both can be evidence-based. The present "
        "results justify continued development and prospective evaluation, not an official "
        "warning product. More held-out years, richer ensemble information and independent "
        "rainfall truth are the conditions for a stronger claim."
    ))

    add(story, "ACKNOWLEDGMENTS", "unnum")
    add(story, (
        "The authors thank the Department of Artificial Intelligence at Dayananda Sagar "
        "College of Engineering and Dr. Kusumika Krori Dutta for academic guidance. They "
        "also acknowledge NOAA, the Copernicus Climate Change Service, GADM, the India "
        "Meteorological Department, NASA, CGIAR-CSI and the Smart India Hackathon 2026 "
        "Ministry of Earth Sciences problem statement SIH26079 for data, methods and "
        "motivation. No external grant supported this manuscript."
    ))

    add(story, "DECLARATIONS", "unnum")
    add(story, "<b>Funding:</b> None.", "bullet")
    add(story, "<b>Conflicts of interest:</b> The authors declare none.", "bullet")
    add(story, (
        "<b>Data availability:</b> The project code and reproducible 2019 sample are in "
        "https://github.com/Harithdn/sanket. GEFSv12 reforecast data are public NOAA "
        "Open Data; ERA5 is available through the Copernicus Climate Data Store under "
        "CC-BY 4.0; GADM 4.1 is free for academic and non-commercial use. Derived "
        "research figures and reports are retained in Research_paper/Final."
    ), "bullet")
    add(story, (
        "<b>Author contributions:</b> D N Harith and Dhanush Thota jointly developed the "
        "study, implementation, analysis and manuscript. Dr. Kusumika Krori Dutta "
        "supervised the work and served as corresponding author."
    ), "bullet")

    add(story, "REFERENCES", "unnum")
    for ref in REFS:
        add_ref(story, ref)

    add(story, "APPENDICES", "unnum")
    add(story, "Appendix A: Reproducibility notes", "h2")
    add(story, (
        "The promoted model is run_20261006T200800Z. Stage-1 regressors allow up to 3,000 "
        "trees with depth 4, learning rate 0.03, subsample and column subsample 0.8, "
        "minimum child weight 10, L1 = 0.5, L2 = 2.0 and 100-round early stopping. Stage 2 "
        "uses depth 3, minimum child weight 15, gamma 0.5, L1 = 1.0, L2 = 3.0 and the same "
        "learning rate and stopping rule. BustCNN uses dilations 1, 2, 4 and 8, GroupNorm "
        "and GELU, with seeds 42-44 for the three-seed comparison. The 2017 test year is "
        "never used for thresholds, bias tables, early stopping, Platt parameters or "
        "conformal quantiles. All figures in this paper were newly drawn from the reported "
        "measurements, and all evaluation numbers refer to the archived experiments rather "
        "than to a live forecast."
    ))
    add(story, (
        "The reproducible builder at Research_paper/Final/build_research_paper.py regenerates "
        "this PDF from source and optionally prints a local shingle-overlap check against "
        "the template, earlier reports and project documentation. The check excludes the "
        "properly cited bibliography from its main-text score, as common plagiarism "
        "workflows do, and also reports the full-document overlap for transparency."
    ))
    return story


# ---------- Validation and originality checks ----------
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    tokens = WORD_RE.findall(text.lower().replace("\u2019", "'"))
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def source_corpus() -> str:
    chunks = []
    pdf_files = sorted((ROOT / "Research_paper").rglob("*.pdf"))
    pdf_files += sorted((ROOT / "docs").rglob("*.pdf"))
    try:
        import pymupdf
        for path in pdf_files:
            if path.resolve() == OUT.resolve():
                continue
            with pymupdf.open(path) as doc:
                chunks.append("\n".join(page.get_text("text") for page in doc))
    except ImportError as exc:
        raise RuntimeError("pymupdf is required for --check") from exc

    text_files = [ROOT / "README.md", ROOT / "CITATION.cff"]
    text_files += sorted((ROOT / "docs").rglob("*.md"))
    text_files += sorted((ROOT / "scripts").rglob("*.py"))
    for path in text_files:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def check_originality() -> tuple[float, float]:
    src_text = source_corpus()
    src8 = shingles(src_text, 8)
    src12 = shingles(src_text, 12)
    main = " ".join(FRONT_TEXT + BODY_TEXT + CAPTION_TEXT + TABLE_TEXT)
    main8, main12 = shingles(main, 8), shingles(main, 12)
    main_score = 100.0 * len(main8 & src8) / max(1, len(main8))
    main12_score = 100.0 * len(main12 & src12) / max(1, len(main12))

    import pymupdf
    with pymupdf.open(OUT) as built:
        pdf_text = "\n".join(page.get_text("text") for page in built)
    full8 = shingles(pdf_text, 8)
    full_score = 100.0 * len(full8 & src8) / max(1, len(full8))

    print(f"Local 8-word overlap, main text: {main_score:.3f}%")
    print(f"Local 8-word overlap, final PDF: {full_score:.3f}%")
    print(f"Local 12-word overlap, main text: {main12_score:.3f}%")
    if main_score > 10.0 or full_score > 10.0:
        raise RuntimeError("Local overlap exceeds the requested 10% ceiling")
    return main_score, full_score


def validate_content() -> None:
    abstract_plain = plain(ABSTRACT).replace("Abstract-", "")
    words = re.findall(r"\b[\w'-]+\b", abstract_plain)
    if not 150 <= len(words) <= 250:
        raise RuntimeError(f"Abstract has {len(words)} words; expected 150-250")
    if len(KEYWORDS) != 5:
        raise RuntimeError("The template requires exactly five keywords")

    prose = " ".join(BODY_TEXT)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", prose)}
    expected = set(range(1, len(REFS) + 1))
    if cited != expected:
        missing = sorted(expected - cited)
        extra = sorted(cited - expected)
        raise RuntimeError(f"Reference mismatch: missing={missing}, extra={extra}")
    for label in ("Table I", "Table II", "Fig. 1", "Fig. 2", "Fig. 3"):
        if label not in prose:
            raise RuntimeError(f"{label} is not referenced in the manuscript")
    if re.search(r"\[[A-Za-z ]+\]|\[Month|\[Full Name|\[Title of Your Paper", " ".join(FRONT_TEXT + BODY_TEXT)):
        raise RuntimeError("Unresolved template placeholder found")
    print(f"Abstract: {len(words)} words; keywords: {len(KEYWORDS)}")
    print(f"References: {len(REFS)}; all cited in text")
    print("Tables and figures: numbered and referenced in text")
    print("Template placeholders: none")


def make_doc() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=36, bottomMargin=36,
        title=TITLE, author="D N Harith, Dhanush Thota",
        subject="IEEE-format research paper for the Sanket project",
        creator="Sanket IEEE research-paper builder",
    )
    title_frame = Frame(
        MARGIN, FIRST_COL_TOP + 12, PAGE_W - 2 * MARGIN, TITLE_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="title",
    )
    left1 = Frame(MARGIN, BOTTOM, COL_W, FIRST_COL_H, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0, id="left1")
    right1 = Frame(MARGIN + COL_W + GUTTER, BOTTOM, COL_W, FIRST_COL_H,
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="right1")
    left2 = Frame(MARGIN, BOTTOM, COL_W, BODY_COL_H, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0, id="left2")
    right2 = Frame(MARGIN + COL_W + GUTTER, BOTTOM, COL_W, BODY_COL_H,
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="right2")
    doc.addPageTemplates([
        PageTemplate(id="First", frames=[title_frame, left1, right1], onPage=footer),
        PageTemplate(id="Body", frames=[left2, right2], onPage=footer),
    ])
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run the local source-overlap check")
    args = parser.parse_args()

    story = build_story()
    validate_content()
    doc = make_doc()
    doc.build(story)

    import pymupdf
    with pymupdf.open(OUT) as built:
        pages = len(built)
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {pages} pages)")
    if args.check:
        check_originality()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
