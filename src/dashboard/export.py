"""
src/dashboard/export.py
────────────────────────
Generates a PDF audit report from risk flags and forecast data.
Uses ReportLab — pure Python, no external dependencies.

Report structure:
  Page 1  — Cover: title, date, portfolio summary
  Page 2  — Cash flow forecast summary table
  Page 3+ — Top N flagged invoices with risk scores
            and audit narratives (if requested)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import EXPORTS_DIR, DASHBOARD_CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


def generate_pdf_report(
    flagged: pd.DataFrame,
    forecast_df: pd.DataFrame,
    meta: dict,
    title: str = "AR Risk & Cash Flow Report",
    include_narratives: bool = True,
    top_n: int = 20,
) -> Path:
    """
    Generates a PDF audit report and saves to EXPORTS_DIR.
    Returns the path to the generated file.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        log.error("reportlab not installed — run: pip install reportlab")
        raise

    # ── File path ────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path  = EXPORTS_DIR / f"ar_risk_report_{timestamp}.pdf"

    # ── Styles ───────────────────────────────────
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "CoverTitle",
        fontSize=22, leading=28, alignment=TA_CENTER,
        textColor=colors.HexColor("#185FA5"), spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        "CoverSub",
        fontSize=12, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor("#5F5E5A"), spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        "SectionHead",
        fontSize=13, leading=18,
        textColor=colors.HexColor("#185FA5"),
        spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "NarrativeText",
        fontSize=8, leading=12,
        textColor=colors.HexColor("#3d3d3a"),
        fontName="Courier", spaceBefore=2
    ))
    styles.add(ParagraphStyle(
        "BodySmall",
        fontSize=9, leading=13,
        textColor=colors.HexColor("#3d3d3a")
    ))

    # ── Color helpers ─────────────────────────────
    def risk_color(prob: float) -> colors.Color:
        if prob >= 0.7:  return colors.HexColor("#FCEBEB")
        if prob >= 0.5:  return colors.HexColor("#FAEEDA")
        return colors.HexColor("#EAF3DE")

    def risk_text_color(prob: float) -> colors.Color:
        if prob >= 0.7:  return colors.HexColor("#A32D2D")
        if prob >= 0.5:  return colors.HexColor("#854F0B")
        return colors.HexColor("#3B6D11")

    # ── Build content ─────────────────────────────
    doc   = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)
    story = []

    # ── Cover page ────────────────────────────────
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph(title, styles["CoverTitle"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}",
        styles["CoverSub"]
    ))
    story.append(Paragraph(
        f"Model: XGBoost + LightGBM  |  Explainability: SHAP TreeExplainer",
        styles["CoverSub"]
    ))
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#B5D4F4")))
    story.append(Spacer(1, 1*cm))

    # Portfolio summary
    total_ar   = flagged["DMBTR"].sum()
    avg_risk   = flagged["LATE_PROB"].mean()
    n_flagged  = len(flagged)

    summary_data = [
        ["Metric", "Value"],
        ["Flagged invoices",         f"{n_flagged:,}"],
        ["Total at-risk AR",         f"${total_ar:,.0f}"],
        ["Average risk score",       f"{avg_risk:.1%}"],
        ["Model CV AUC-ROC",         f"{meta.get('cv_auc', 0):.4f}"],
        ["Risk threshold",           f"{meta.get('threshold', 0):.2f}"],
    ]

    t = Table(summary_data, colWidths=[9*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#185FA5")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#E6F1FB")]),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#B5D4F4")),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── Forecast section ──────────────────────────
    story.append(Paragraph("Cash Flow Forecast", styles["SectionHead"]))
    story.append(Paragraph(
        "Probabilistic 30/60/90-day cash inflow projections with 80% confidence intervals.",
        styles["BodySmall"]
    ))
    story.append(Spacer(1, 0.3*cm))

    fc_data = [["Horizon", "Lower (10th pct)", "Median (50th pct)", "Upper (90th pct)", "Uncertainty"]]
    for _, row in forecast_df.iterrows():
        fc_data.append([
            row["Horizon"],
            f"${row['Lower_10pct']:,.0f}",
            f"${row['Median_50pct']:,.0f}",
            f"${row['Upper_90pct']:,.0f}",
            f"{row['Uncertainty_Pct']:.1f}%",
        ])

    ft = Table(fc_data, colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3*cm])
    ft.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#185FA5")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ALIGN",        (1,0), (-1,-1), "RIGHT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#E6F1FB")]),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#B5D4F4")),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(ft)
    story.append(PageBreak())

    # ── Flagged invoices ──────────────────────────
    story.append(Paragraph(
        f"High-Risk Invoices — Top {min(top_n, n_flagged)}",
        styles["SectionHead"]
    ))
    story.append(Spacer(1, 0.2*cm))

    inv_data = [["Invoice", "Customer", "Industry", "Amount", "Due Date", "Risk Score"]]
    for _, row in flagged.head(top_n).iterrows():
        inv_data.append([
            str(row.get("BELNR", "")),
            str(row.get("KUNNR", ""))[:12],
            str(row.get("INDUSTRY", "")),
            f"${row['DMBTR']:,.0f}",
            str(row.get("FAEDT", ""))[:10],
            f"{row['LATE_PROB']:.1%}",
        ])

    # Color-code rows by risk
    inv_table = Table(inv_data, colWidths=[2.5*cm, 3*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    row_styles = [
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#185FA5")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#B5D4F4")),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("ALIGN",        (3,0), (5,-1), "RIGHT"),
    ]
    for i, (_, row) in enumerate(flagged.head(top_n).iterrows(), start=1):
        bg = risk_color(row["LATE_PROB"])
        row_styles.append(("BACKGROUND", (0,i), (-1,i), bg))

    inv_table.setStyle(TableStyle(row_styles))
    story.append(inv_table)

    # ── Audit narratives ──────────────────────────
    if include_narratives and "NARRATIVE" in flagged.columns:
        story.append(PageBreak())
        story.append(Paragraph("Audit Narratives", styles["SectionHead"]))
        story.append(Paragraph(
            "AI-generated, SHAP-backed explanations for each flagged invoice.",
            styles["BodySmall"]
        ))

        for i, (_, row) in enumerate(flagged.head(top_n).iterrows()):
            if "NARRATIVE" not in row or pd.isna(row.get("NARRATIVE", None)):
                continue
            story.append(Spacer(1, 0.4*cm))
            story.append(Paragraph(
                f"Invoice {row.get('BELNR', i+1)} — ${row['DMBTR']:,.0f} — Risk: {row['LATE_PROB']:.1%}",
                styles["SectionHead"]
            ))
            story.append(Paragraph(
                row["NARRATIVE"].replace("\n", "<br/>"),
                styles["NarrativeText"]
            ))
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#D3D1C7")
            ))

    # ── Footer disclaimer ─────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#D3D1C7")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "This report is generated by an ML model and is intended to assist, not replace, "
        "professional judgment. All flagged invoices should be reviewed by a qualified "
        "finance professional before action is taken. Model performance metrics are "
        f"available in the dashboard (CV AUC-ROC: {meta.get('cv_auc', 'N/A')}).",
        styles["BodySmall"]
    ))

    doc.build(story)
    log.success(f"PDF report generated → {pdf_path}")
    return pdf_path
