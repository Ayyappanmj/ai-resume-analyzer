"""PDF report generation with ReportLab (ASCII-safe text only)."""
import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..schemas import AnalysisResult

ACCENT = colors.HexColor("#6d28d9")


def _p(text: str, style) -> Paragraph:
    return Paragraph(escape(text), style)


def build_pdf(r: AnalysisResult) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="Resume Analysis Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=ACCENT, fontSize=22, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=ACCENT, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=0)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    def bullets(items: list[str]) -> list:
        return [Paragraph(escape(i), bullet, bulletText="-") for i in items] or [_p("None", body)]

    s: list = [_p("AI Resume Analysis Report", h1),
               _p(f"File: {r.filename}  |  Generated: {r.created_at:%Y-%m-%d %H:%M} UTC", small),
               Spacer(1, 8)]

    score_tbl = Table(
        [[f"{r.ats_score}/100", r.grade, f"{r.similarity_score:.1f}%", str(len(r.matched_skills)) + "/" + str(len(r.jd_skills))],
         ["ATS score", "Grade", "JD similarity", "JD skills matched"]],
        colWidths=[42 * mm] * 4,
    )
    score_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, 0), 18), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT), ("FONTSIZE", (0, 1), (-1, 1), 8),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey), ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    s += [score_tbl, _p("Score breakdown", h2)]

    rows = [["Category", "Score", "Details"]] + [
        [i.name, f"{i.score:g} / {i.max:g}", Paragraph(escape(i.detail), small)] for i in r.breakdown]
    t = Table(rows, colWidths=[45 * mm, 25 * mm, 100 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    s += [t, _p("Summary", h2), _p(r.summary, body)]

    s += [_p("Matched skills", h2), _p(", ".join(r.matched_skills) or "None", body),
          _p("Missing skills", h2), _p(", ".join(r.missing_skills) or "None - all detected job skills are present", body),
          _p("Top keywords in resume", h2), _p(", ".join(r.keywords), body),
          _p("Keywords in the job description missing from resume", h2), _p(", ".join(r.missing_keywords) or "None", body),
          _p("Sections", h2),
          _p("Detected: " + (", ".join(r.sections_detected) or "none"), body),
          _p("Missing: " + (", ".join(r.sections_missing) or "none"), body)]

    if r.strengths:
        s += [_p("Strengths", h2), *bullets(r.strengths)]
    if r.weaknesses:
        s += [_p("Weaknesses", h2), *bullets(r.weaknesses)]
    s += [_p("Recommendations", h2), *bullets(r.suggestions)]
    if r.improved_bullets:
        s += [_p("Suggested bullet rewrites", h2), *bullets(r.improved_bullets)]
    if r.interview_questions:
        s += [_p("Likely interview questions", h2), *bullets(r.interview_questions)]

    doc.build(s)
    return buf.getvalue()
