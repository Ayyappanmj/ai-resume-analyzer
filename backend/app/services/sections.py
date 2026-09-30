"""Rule-based resume section detection."""
import re

SECTION_ALIASES: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective",
                "about me", "about", "executive summary"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "employment", "internships", "internship experience",
                   "relevant experience"],
    "education": ["education", "academic background", "academics", "qualifications",
                  "education and training"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "skills & tools",
               "skills and tools", "technologies", "tech stack", "competencies"],
    "projects": ["projects", "personal projects", "academic projects", "selected projects",
                 "key projects"],
    "certifications": ["certifications", "certificates", "licenses", "courses", "training"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "awards and honors"],
    "publications": ["publications", "research", "papers"],
    "languages": ["languages"],
    "interests": ["interests", "hobbies", "volunteering", "volunteer experience"],
}

CORE_SECTIONS = ["summary", "experience", "education", "skills", "projects"]

_LOOKUP = {alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases}


def _heading_key(line: str) -> str | None:
    if not line or len(line.split()) > 5:
        return None
    cleaned = re.sub(r"[^a-z& ]", " ", line.lower())
    return _LOOKUP.get(" ".join(cleaned.split()))


def detect_sections(text: str) -> dict[str, str]:
    """Split resume text into {section: body}. Text before the first heading is 'header'."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        key = _heading_key(line.strip())
        if key:
            current = key
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def detected_names(sections: dict[str, str]) -> list[str]:
    return [k for k, v in sections.items() if k != "header" and v.strip()]


def missing_core(sections: dict[str, str]) -> list[str]:
    have = set(detected_names(sections))
    return [s for s in CORE_SECTIONS if s not in have]
