from app.services import nlp, scoring
from app.services import sections as sec

RESUME = """Jane Doe
jane@doe.com | +1 415 555 0134 | github.com/janedoe
SUMMARY
Backend engineer with 5 years of experience.
EXPERIENCE
Acme Corp 2021 - 2024
- Built REST APIs with Python and FastAPI serving 50,000 users
- Reduced latency by 40% by adding Redis caching
- Deployed services with Docker on AWS
EDUCATION
BSc Computer Science, 2019
SKILLS
Python, SQL, PostgreSQL, Docker
"""


def test_section_detection():
    s = sec.detect_sections(RESUME)
    assert {"summary", "experience", "education", "skills"} <= set(sec.detected_names(s))
    assert "projects" in sec.missing_core(s)


def test_skill_extraction():
    found = nlp.extract_skills(RESUME)
    for skill in ["Python", "FastAPI", "Redis", "Docker", "AWS", "PostgreSQL", "SQL"]:
        assert skill in found


def test_ambiguous_words_not_matched():
    assert "Go" not in nlp.extract_skills("We go to market and rest of the team can go home")


def test_contact():
    c = nlp.extract_contact(RESUME)
    assert c["email"] == "jane@doe.com" and "github" in c and "phone" in c


def test_metrics_and_verbs():
    assert scoring.count_metrics(RESUME) >= 2
    assert scoring.count_action_verbs(RESUME) >= 3


def test_score_bounds_and_weights():
    total, items = scoring.compute_ats(
        text=RESUME, sections_detected=["summary", "experience", "education", "skills"],
        contact={"email": "a@b.c", "phone": "1"}, matched=["Python"], jd_skills=["Python", "Go"],
        resume_skills=["Python"], keyword_coverage=0.5, similarity=0.5, method="sentence-transformers")
    assert 0 <= total <= 100
    assert sum(i.max for i in items) == 100
    assert all(0 <= i.score <= i.max for i in items)


def test_grade():
    assert scoring.grade_for(90) == "Excellent" and scoring.grade_for(10) == "Needs work"
