"""Lightweight skill extraction, keyword extraction and contact parsing."""
import re
from collections import Counter
from functools import lru_cache

from .skills import PATTERN_TO_SKILL, SKILL_TO_CATEGORY

GENERIC_WORDS = {
    "experience", "work", "team", "year", "years", "ability", "skill", "skills", "role", "job",
    "company", "candidate", "requirement", "requirements", "responsibility", "responsibilities",
    "knowledge", "opportunity", "position", "level", "field", "world", "way", "time", "day",
    "environment", "plus", "etc", "well", "strong", "excellent", "good", "great", "new", "use",
    "working", "using", "related", "including", "preferred", "required", "minimum", "degree",
    "benefit", "salary", "location", "equal", "employer", "application", "people", "help",
}
STOP_WORDS = {
    "about", "after", "again", "also", "among", "and", "any", "are", "because", "been", "before",
    "being", "between", "both", "but", "can", "could", "did", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "her", "here",
    "hers", "him", "his", "how", "into", "its", "itself", "just", "more", "most", "other",
    "our", "out", "over", "own", "same", "should", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "under",
    "until", "very", "was", "were", "what", "when", "where", "which", "while", "who", "will",
    "with", "would", "you", "your",
}


@lru_cache
def _skill_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple(
        (
            re.compile(
                r"(?<![A-Za-z0-9_])"
                + re.sub(r"\\ ", r"\\s+", re.escape(pattern))
                + r"(?![A-Za-z0-9_])",
                re.IGNORECASE,
            ),
            skill,
        )
        for pattern, skill in PATTERN_TO_SKILL.items()
    )


def extract_skills(text: str) -> dict[str, int]:
    """Return {canonical_skill: mention_count} using boundary-aware regexes."""
    counts: Counter[str] = Counter()
    for pattern, skill in _skill_patterns():
        matches = sum(1 for _ in pattern.finditer(text))
        if matches:
            counts[skill] += matches
    return dict(counts)


def group_skills(skills: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for s in skills:
        grouped.setdefault(SKILL_TO_CATEGORY.get(s, "Other"), []).append(s)
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """Return frequency-ranked terms while excluding common and generic filler."""
    words = re.findall(r"[a-z][a-z+#.]{1,}", text[:20000].lower())
    counter: Counter[str] = Counter(
        word for word in words if word not in STOP_WORDS and word not in GENERIC_WORDS
    )
    for skill, n in extract_skills(text).items():
        counter[skill.lower()] += 3 * n

    return [word for word, _ in counter.most_common(max(0, top_n))]


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}")
_LINKEDIN = re.compile(r"linkedin\.com/[\w\-/%]+", re.I)
_GITHUB = re.compile(r"github\.com/[\w\-]+", re.I)


def extract_contact(text: str) -> dict[str, str]:
    head = "\n".join(text.splitlines()[:15])
    out: dict[str, str] = {}
    if m := _EMAIL.search(text):
        out["email"] = m.group(0)
    if m := _PHONE.search(head):
        if len(re.sub(r"\D", "", m.group(0))) >= 9:
            out["phone"] = m.group(0).strip()
    if m := _LINKEDIN.search(text):
        out["linkedin"] = m.group(0)
    if m := _GITHUB.search(text):
        out["github"] = m.group(0)
    return out
