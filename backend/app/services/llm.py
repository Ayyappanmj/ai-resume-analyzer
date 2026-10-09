"""Groq-powered resume feedback with rule-based fallbacks."""
import json
import logging
import os

from groq import Groq, GroqError

log = logging.getLogger(__name__)
MODEL = "llama-3.1-8b-instant"

SYSTEM = (
    "You are an expert technical recruiter and resume coach. Be specific, honest and concise. "
    "NEVER invent employers, titles, technologies, or numbers that are not in the resume. "
    "When a metric is unknown, use a placeholder like [X%]. Reply with JSON only."
)

PROMPT = """RESUME:
\"\"\"
{resume}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{jd}
\"\"\"

Skills required by the job but missing from the resume: {missing}

Return a JSON object with exactly these keys:
- "summary": string, 3-4 sentence neutral summary of the candidate and their fit for this job
- "strengths": array of 3-5 short strings
- "weaknesses": array of 3-5 short strings
- "improved_bullets": array of 5 rewritten resume bullets (action verb + task + tool + measurable result, based only on the resume's real content)
- "interview_questions": array of 8 questions likely to be asked for this job (mix technical and behavioural, tailored to the gaps and strengths above)
"""


def _as_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_with_llm(resume: str, jd: str, missing: list[str]) -> dict | None:
    """Return parsed Groq feedback or None when unavailable or invalid."""
    if not is_configured():
        log.warning("GROQ_API_KEY is not configured; using rule-based LLM fallbacks")
        return None

    prompt = PROMPT.format(
        resume=resume[:4000], jd=jd[:2500], missing=", ".join(missing[:12]) or "none")
    try:
        completion = _groq_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        if not content:
            return None
        data = json.loads(content)
    except (GroqError, json.JSONDecodeError, IndexError, AttributeError, TypeError) as exc:
        log.warning("Groq call failed (%s)", type(exc).__name__)
        return None

    out = {
        "summary": str(data.get("summary", "")).strip(),
        "strengths": _as_list(data.get("strengths")),
        "weaknesses": _as_list(data.get("weaknesses")),
        "improved_bullets": _as_list(data.get("improved_bullets")),
        "interview_questions": _as_list(data.get("interview_questions")),
    }
    return out if out["summary"] else None


def fallback_summary(*, filename: str, words: int, sections: list[str], skills: list[str],
                     ats: int, matched: list[str], jd_skills: list[str]) -> str:
    top = ", ".join(skills[:6]) or "no recognised technical skills"
    sec = ", ".join(sections) or "no standard sections"
    fit = f"matches {len(matched)} of {len(jd_skills)} skills named in the job description" if jd_skills else \
        "could not be compared against specific skills in the job description"
    return (f"Resume of {words} words with sections: {sec}. Most prominent skills: {top}. "
            f"It scores {ats}/100 for ATS compatibility and {fit}. "
            "(Rule-based summary - configure GROQ_API_KEY for an AI-written one.)")


def fallback_questions(matched: list[str], missing: list[str]) -> list[str]:
    qs = [f"Walk me through a project where you used {s}. What trade-offs did you make?" for s in matched[:4]]
    qs += [f"This role needs {s}, which I don't see on your resume. How would you get productive with it?"
           for s in missing[:3]]
    qs += ["Tell me about a time you disagreed with a teammate. How was it resolved?",
           "Describe the most technically challenging problem you solved recently."]
    return qs[:9]


def fallback_bullets(matched: list[str]) -> list[str]:
    tools = matched[:5] or ["[tool]"]
    return [f"Built [what] using {t}, improving [metric] by [X%] for [users/team]." for t in tools]
