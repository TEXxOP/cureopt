"""
CureOpt AI — PDF Report Export
Generates professional PDF report using ReportLab.
"""

import io
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# Brand colors
BRAND_BLUE = HexColor("#1a73e8")
BRAND_DARK = HexColor("#202124")
BRAND_GRAY = HexColor("#5f6368")
BRAND_LIGHT = HexColor("#f8f9fa")
BRAND_GREEN = HexColor("#34a853")
BRAND_RED = HexColor("#ea4335")


def _build_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="BrandTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=BRAND_BLUE,
        spaceAfter=6 * mm,
    ))

    styles.add(ParagraphStyle(
        name="BrandSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=BRAND_GRAY,
        alignment=TA_CENTER,
        spaceAfter=10 * mm,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=BRAND_DARK,
        spaceBefore=8 * mm,
        spaceAfter=4 * mm,
        borderColor=BRAND_BLUE,
        borderWidth=2,
        borderPadding=2,
    ))

    styles.add(ParagraphStyle(
        name="MetricLabel",
        parent=styles["Normal"],
        fontSize=10,
        textColor=BRAND_GRAY,
    ))

    styles.add(ParagraphStyle(
        name="MetricValue",
        parent=styles["Normal"],
        fontSize=14,
        textColor=BRAND_DARK,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=BRAND_GRAY,
        alignment=TA_CENTER,
    ))

    return styles


def generate_report(recommendation, cost_breakdown, demould_info,
                    input_params, region, scenarios=None):
    """
    Generate a PDF report buffer.

    Args:
        recommendation: dict from optimizer
        cost_breakdown: dict from cost model
        demould_info: dict from predict module
        input_params: dict of input parameters
        region: str
        scenarios: optional list of comparison scenarios

    Returns:
        bytes — PDF content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = _build_styles()
    story = []

    # ─── Header ─────────────────────────────────────────────────
    story.append(Paragraph("🏗️ CureOpt AI", styles["BrandTitle"]))
    story.append(Paragraph(
        "AI-Powered Cycle Time Optimization Report",
        styles["BrandSubtitle"]
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')} | Region: {region}",
        styles["BrandSubtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_BLUE))
    story.append(Spacer(1, 6 * mm))

    # ─── Input Parameters ───────────────────────────────────────
    story.append(Paragraph("Input Parameters", styles["SectionHeader"]))

    input_data = [
        ["Parameter", "Value"],
        ["Cement Content", f"{input_params.get('cement_pct', 'N/A')}%"],
        ["Fly Ash Content", f"{input_params.get('fly_ash_pct', 'N/A')}%"],
        ["Water-Cement Ratio", f"{input_params.get('water_cement_ratio', 'N/A')}"],
        ["Curing Method", f"{input_params.get('curing_method', 'N/A')}"],
        ["Admixture", f"{input_params.get('admixture_type', 'N/A')}"],
        ["Ambient Temperature", f"{input_params.get('ambient_temp_C', 'N/A')}°C"],
        ["Relative Humidity", f"{input_params.get('humidity_pct', 'N/A')}%"],
        ["Required Strength", f"{input_params.get('required_mpa', 'N/A')} MPa"],
    ]

    t = Table(input_data, colWidths=[200, 250])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
        ("BACKGROUND", (0, 1), (-1, -1), BRAND_LIGHT),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), BRAND_LIGHT]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))

    # ─── Recommendation ─────────────────────────────────────────
    story.append(Paragraph("🎯 Optimal Recommendation", styles["SectionHeader"]))

    rec_data = [
        ["Metric", "Value"],
        ["Optimal Curing Method", recommendation.get("curing_method", "N/A")],
        ["Optimal Mix (w/c)", str(recommendation.get("water_cement_ratio", "N/A"))],
        ["Admixture", recommendation.get("admixture_type", "N/A")],
        ["Predicted De-mould Time", f"{recommendation.get('cycle_time_hr', 'N/A')} hours"],
        ["Predicted Strength at De-mould", f"{recommendation.get('predicted_strength', 'N/A')} MPa"],
        ["Cycle Time Reduction", f"{recommendation.get('time_reduction_pct', 'N/A')}%"],
        ["Cost Index (vs baseline)", str(recommendation.get("cost_index", "N/A"))],
    ]

    t2 = Table(rec_data, colWidths=[200, 250])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#e8f5e9")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6 * mm))

    # ─── Cost Breakdown ──────────────────────────────────────────
    story.append(Paragraph("💰 Cost Breakdown", styles["SectionHeader"]))

    cost_data = [
        ["Component", "Amount (₹)"],
        ["Material", f"₹{cost_breakdown.get('material', 0):,.2f}"],
        ["Energy (Curing)", f"₹{cost_breakdown.get('energy', 0):,.2f}"],
        ["Labor", f"₹{cost_breakdown.get('labor', 0):,.2f}"],
        ["Mold Occupancy", f"₹{cost_breakdown.get('mold_occupancy', 0):,.2f}"],
        ["TOTAL", f"₹{cost_breakdown.get('total', 0):,.2f}"],
    ]

    t3 = Table(cost_data, colWidths=[200, 250])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
        ("BACKGROUND", (0, 1), (-1, -2), BRAND_LIGHT),
        ("BACKGROUND", (0, -1), (-1, -1), HexColor("#e3f2fd")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6 * mm))

    # ─── Scenario Comparison (if available) ──────────────────────
    if scenarios and len(scenarios) > 1:
        story.append(Paragraph("📊 Scenario Comparison (Top 4)", styles["SectionHeader"]))

        sc_data = [["#", "Curing", "Admixture", "w/c", "Cycle (h)", "Strength", "Cost Index", "ΔTime"]]
        for i, sc in enumerate(scenarios[:4], 1):
            sc_data.append([
                str(i),
                sc.get("curing_method", ""),
                sc.get("admixture_type", ""),
                str(sc.get("water_cement_ratio", "")),
                str(sc.get("cycle_time_hr", "")),
                f"{sc.get('predicted_strength', '')} MPa",
                str(sc.get("cost_index", "")),
                f"{sc.get('time_reduction_pct', '')}%",
            ])

        t4 = Table(sc_data, colWidths=[25, 70, 70, 40, 55, 65, 60, 50])
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), BRAND_LIGHT]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t4)
        story.append(Spacer(1, 6 * mm))

    # ─── Footer ──────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_GRAY))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "CureOpt AI — AI-Powered Cycle Time Optimization for Precast Yards | "
        "L&T CreaTech 2025 | Confidential",
        styles["Footer"]
    ))

    # Build
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
