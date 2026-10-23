"""PDF report generation with ReportLab."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SYSTEM_NAME = "Osteoporosis Screening-Support Prototype (Knee X-ray)"

DISCLAIMER = (
    "This report was produced by an experimental academic prototype. It is NOT a medical "
    "diagnosis and must not be used for clinical decisions. It does not replace a DXA scan "
    "or assessment by a qualified clinician."
)

LIMITATIONS = [
    "Scores are model outputs, not calibrated probabilities of disease.",
    "X-ray-only mode was close to chance on held-out data. The combined mode (X-ray + patient "
    "details) performed better but was still wrong about one time in three; most of its signal "
    "comes from the patient details, especially age.",
    "The model was trained on a small public dataset (about 210 images from a single "
    "collection site in India), labelled from quantitative-ultrasound T-scores rather than DXA.",
    "The system checks that the file is a valid image, but it cannot verify that the image "
    "is a knee X-ray. Images of other anatomy will still produce a (meaningless) output.",
    "The Grad-CAM heatmap shows which regions influenced the score. It does not prove that "
    "the model's reasoning is clinically valid.",
]


def build_pdf(path: Path, *, report_id: int, scan_id: int, predicted_at: datetime, label: str,
              scores: dict[str, float], model_version: str, preprocessing_version: str,
              original_image: Path | None, overlay_image: Path | None,
              heatmap_informative: bool, mode: str = "image_only",
              component_scores: dict | None = None, clinical_inputs: dict | None = None) -> None:
    """Write the PDF to `path`. Missing images are replaced by a note, not an error."""
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, leading=11)
    warning = ParagraphStyle("warning", parent=styles["BodyText"], fontSize=9, leading=12,
                             textColor=colors.HexColor("#8a1c1c"), backColor=colors.HexColor("#fdecea"),
                             borderPadding=6)
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=15, leading=19, spaceAfter=6)
    story = [
        Paragraph(SYSTEM_NAME, title),
        Paragraph(DISCLAIMER, warning),
        Spacer(1, 0.4 * cm),
    ]

    details = [
        ["Report ID", f"R-{report_id:06d}"],
        ["Scan ID", f"S-{scan_id:06d}"],
        ["Prediction time (UTC)", predicted_at.strftime("%Y-%m-%d %H:%M:%S")],
        ["Mode", "X-ray + patient details (hybrid)" if mode == "hybrid" else "X-ray only (low reliability)"],
        ["Model output", label + ("  (screening rule: Osteoporosis score x%.1f)" % (component_scores or {}).get("osteoporosis_boost", 1.0)
                                  if mode == "hybrid" and (component_scores or {}).get("osteoporosis_boost", 1.0) > 1 else "")],
        ["Model score for that class", f"{scores[label]:.3f}  (uncalibrated)"],
        ["Model version", model_version],
        ["Preprocessing version", preprocessing_version],
    ]
    story.append(_table(details, col_widths=[5.5 * cm, 11 * cm], header=False))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Model scores for all classes", styles["Heading3"]))
    parts = component_scores or {}
    if mode == "hybrid" and parts.get("clinical"):
        rows = [["Class", "Patient details", "X-ray", "Combined"]] + [
            [name, f"{parts['clinical'][name]:.3f}", f"{parts['image'][name]:.3f}", f"{value:.3f}"]
            for name, value in scores.items()]
        story.append(_table(rows, col_widths=[4 * cm, 3.5 * cm, 3 * cm, 3 * cm], header=True))
    else:
        rows = [["Class", "Model score"]] + [[name, f"{value:.3f}"] for name, value in scores.items()]
        story.append(_table(rows, col_widths=[5.5 * cm, 4 * cm], header=True))
    if clinical_inputs:
        flags = [k.replace("_", " ") for k in ("smoker", "diabetic", "hypothyroidism", "estrogen_use",
                                               "fracture_history", "family_history") if clinical_inputs.get(k)]
        text = (f"Patient details used: age {clinical_inputs['age']}, {clinical_inputs['sex']}, "
                f"{clinical_inputs['height_cm']} cm, {clinical_inputs['weight_kg']} kg"
                + (f", menopause at {clinical_inputs['menopause_age']}" if clinical_inputs.get("menopause_age") else "")
                + (f"; {', '.join(flags)}" if flags else "") + ".")
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(text, small))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Images", styles["Heading3"]))
    cells = [_image_or_note(original_image, "Input (preprocessed view)"),
             _image_or_note(overlay_image, "Grad-CAM overlay" if heatmap_informative
                            else "Grad-CAM overlay (no positive evidence found)")]
    image_table = Table([cells], colWidths=[6.2 * cm, 6.2 * cm])
    image_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(image_table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Limitations", styles["Heading3"]))
    for item in LIMITATIONS:
        story.append(Paragraph(f"• {item}", small))

    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title=f"Report R-{report_id:06d}", author=SYSTEM_NAME).build(story)


def _table(rows, col_widths, header: bool) -> Table:
    table = Table(rows, colWidths=col_widths)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced6")),
             ("FONTSIZE", (0, 0), (-1, -1), 9.5),
             ("BACKGROUND", (0, 0), (0 if not header else -1, -1 if not header else 0), colors.HexColor("#eef2f7"))]
    table.setStyle(TableStyle(style))
    return table


def _image_or_note(path: Path | None, caption: str):
    styles = getSampleStyleSheet()
    if path is not None and path.is_file():
        return [Image(str(path), width=5.6 * cm, height=5.6 * cm), Paragraph(caption, styles["BodyText"])]
    return [Paragraph(f"{caption}: image not available.", styles["BodyText"])]
