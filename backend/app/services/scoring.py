"""Transparent 0-100 ATS score. Weights: skills 40 / semantic 25 / sections 15 / format 10 / impact 10."""
import re

from ..schemas import ScoreItem
from .sections import SECTION_ALIASES

ACTION_VERBS = {
    "achieved", "architected", "automated", "built", "collaborated", "created", "delivered",
    "deployed", "designed", "developed", "drove", "enhanced", "engineered", "established",
    "implemented", "improved", "increased", "integrated", "launched", "led", "managed",
    "mentored", "migrated", "optimized", "optimised", "orchestrated", "owned", "reduced",
    "refactored", "resolved", "scaled", "shipped", "streamlined", "trained", "transformed",
}

_METRIC = re.compile(
    r"\d+(?:\.\d+)?\s?%|[$€£]\s?\d[\d,.]*\s?[kmb]?\b|\b\d+x\b|"
    r"\b\d{2,}(?:,\d{3})*\+?\s+(?:users|customers|clients|requests|engineers|members|projects|"
    r"services|models|records|stores|teams|ms|hours|tickets)\b",
    re.I,
)
_BULLET = re.compile(r"^\s*[•▪●◦·\-–*]\s+", re.M)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

SECTION_WEIGHTS = {"experience": 4.0, "education": 3.0, "skills": 3.0, "summary": 2.0, "projects": 1.5}


def grade_for(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Needs work"


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def count_metrics(text: str) -> int:
    return len(_METRIC.findall(text))


def count_action_verbs(text: str) -> int:
    return sum(1 for w in re.findall(r"[a-z]+", text.lower()) if w in ACTION_VERBS)


def compute_ats(
    *,
    text: str,
    sections_detected: list[str],
    contact: dict[str, str],
    matched: list[str],
    jd_skills: list[str],
    resume_skills: list[str],
    keyword_coverage: float,
    similarity: float,
    method: str,
) -> tuple[int, list[ScoreItem]]:
    items: list[ScoreItem] = []

    # 1) Skills + keyword match (40)
    if jd_skills:
        skill_cov = len(matched) / len(jd_skills)
        s = 40 * (0.7 * skill_cov + 0.3 * keyword_coverage)
        detail = f"{len(matched)}/{len(jd_skills)} JD skills found; {keyword_coverage:.0%} keyword coverage"
    else:
        s = 40 * (0.5 * _clip(len(resume_skills) / 15) + 0.5 * keyword_coverage)
        detail = f"No known skills in JD; {len(resume_skills)} resume skills, {keyword_coverage:.0%} keyword coverage"
    items.append(ScoreItem(name="Skills & Keywords", score=round(s, 1), max=40, detail=detail))

    # 2) Semantic similarity (25)
    lo, span = (0.15, 0.55) if method == "sentence-transformers" else (0.05, 0.50)
    sem = _clip((similarity - lo) / span)
    items.append(ScoreItem(name="Semantic Match", score=round(25 * sem, 1), max=25,
                           detail=f"Cosine similarity {similarity:.2f} ({method})"))

    # 3) Sections + contact (15)
    present = set(sections_detected)
    sec_score = sum(w for name, w in SECTION_WEIGHTS.items() if name in present)
    c = 0.0
    c += 0.75 if "email" in contact else 0
    c += 0.5 if "phone" in contact else 0
    c += 0.25 if ("linkedin" in contact or "github" in contact) else 0
    items.append(ScoreItem(name="Sections & Contact", score=round(sec_score + c, 1), max=15,
                           detail=f"Sections: {', '.join(sorted(present)) or 'none'}; contact fields: {len(contact)}"))

    # 4) Formatting / parseability (10)
    words = len(text.split())
    bullets = len(_BULLET.findall(text))
    years = len(_YEAR.findall(text))
    f = 4 if 350 <= words <= 1000 else 2 if (200 <= words < 350 or 1000 < words <= 1400) else 0
    f += 3 if bullets >= 6 else 1.5 if bullets >= 3 else 0
    f += 3 if years >= 2 else 1.5 if years == 1 else 0
    items.append(ScoreItem(name="Formatting", score=float(f), max=10,
                           detail=f"{words} words, {bullets} bullets, {years} dates"))

    # 5) Impact (10)
    m, v = count_metrics(text), count_action_verbs(text)
    imp = 5 * _clip(m / 5) + 5 * _clip(v / 8)
    items.append(ScoreItem(name="Impact & Action Verbs", score=round(imp, 1), max=10,
                           detail=f"{m} quantified results, {v} action verbs"))

    total = int(round(sum(i.score for i in items)))
    return max(0, min(100, total)), items


def build_suggestions(*, items: list[ScoreItem], missing_skills: list[str], missing_sections: list[str],
                      missing_keywords: list[str], contact: dict[str, str]) -> list[str]:
    tips: list[str] = []
    if missing_skills:
        tips.append("Add these job-required skills only if you genuinely have them: "
                    + ", ".join(missing_skills[:8]) + ".")
    if missing_keywords:
        tips.append("Mirror the job's wording where truthful (e.g. " + ", ".join(missing_keywords[:6]) + ").")
    for s in missing_sections:
        label = SECTION_ALIASES[s][0].title()
        tips.append(f"Add a clearly labelled '{label}' section - ATS parsers rely on standard headings.")
    by_name = {i.name: i for i in items}
    if by_name["Impact & Action Verbs"].score < 6:
        tips.append("Quantify results (%, $, time saved, users) and start bullets with strong action verbs.")
    if by_name["Formatting"].score < 7:
        tips.append("Use simple bullets, include dates for each role, and keep length to ~1-2 pages (350-1000 words).")
    if "email" not in contact or "phone" not in contact:
        tips.append("Put email and phone as plain text at the top of the resume.")
    if "linkedin" not in contact and "github" not in contact:
        tips.append("Add a LinkedIn and/or GitHub URL.")
    if not tips:
        tips.append("Strong match. Tailor the summary to the role and keep quantifying achievements.")
    return tips
