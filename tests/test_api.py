import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app import main
from app.services import embeddings, llm

JD = ("We are hiring a backend engineer with strong Python, FastAPI, Docker and AWS experience. "
      "You will design REST APIs, work with PostgreSQL and Redis, and collaborate in an Agile team. "
      "Kubernetes and Terraform are a plus.")


def make_pdf(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for ln in lines:
        c.drawString(50, y, ln)
        y -= 16
    c.save()
    return buf.getvalue()


RESUME_LINES = (
    ["Jane Doe", "jane@doe.com  +1 415 555 0134  github.com/janedoe", "SUMMARY",
     "Backend engineer building reliable Python services for high-traffic products."]
    + ["EXPERIENCE", "Acme Corp 2021 - 2024"]
    + ["- Built REST APIs with Python and FastAPI for 50,000 users",
       "- Reduced latency by 40% using Redis caching",
       "- Deployed services with Docker on AWS and PostgreSQL databases",
       "- Led an Agile team of 4 engineers and mentored two juniors",
       "- Automated CI/CD pipelines with GitHub Actions",
       "- Designed microservices that improved uptime to 99.9%"]
    + ["EDUCATION", "BSc Computer Science 2019", "SKILLS", "Python, SQL, PostgreSQL, Docker, AWS, Git"]
    + ["Additional context line to make sure the resume has enough words for the parser."] * 4
)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(embeddings, "similarity", lambda a, b: (0.55, "tfidf"))
    monkeypatch.setattr(llm, "analyze_with_llm", lambda *a, **k: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    response = client.get("/health").json()
    assert response["status"] == "ok"
    assert "groq_configured" in response and response["model"] == llm.MODEL


def test_groq_response_is_parsed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    payload = (
        '{"summary":"Candidate summary","strengths":["Python"],"weaknesses":["Cloud"],'
        '"improved_bullets":["Built an API"],"interview_questions":["How?"]}'
    )
    request = {}

    def create(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(llm, "_groq_client", lambda: fake_client)

    result = llm.analyze_with_llm("resume", "job description", ["AWS"])

    assert result is not None and result["summary"] == "Candidate summary"
    assert request["model"] == "llama-3.1-8b-instant"
    assert request["temperature"] == 0.2
    assert request["response_format"] == {"type": "json_object"}


def test_similarity_uses_lightweight_tfidf():
    score, method = embeddings.similarity(
        "Python APIs with Redis caching",
        "Backend engineering with Python and Redis",
    )

    assert 0 < score < 1
    assert method == "tfidf"


def test_similarity_with_empty_text_is_zero():
    assert embeddings.similarity("", "Python backend role") == (0.0, "tfidf")


def test_groq_missing_api_key_returns_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_groq_client", lambda: pytest.fail("client should not be created"))

    assert llm.analyze_with_llm("resume", "job description", []) is None


def test_analyze_and_report(client):
    pdf = make_pdf(RESUME_LINES)
    r = client.post("/analyze", files={"resume": ("cv.pdf", pdf, "application/pdf")},
                    data={"jd_text": JD, "use_llm": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["ats_score"] <= 100
    assert "Python" in body["matched_skills"]
    assert "Kubernetes" in body["missing_skills"]
    assert body["llm_used"] is False and body["interview_questions"]

    rep = client.post("/report", json=body)
    assert rep.status_code == 200 and rep.content.startswith(b"%PDF")


def test_rejects_non_pdf(client):
    r = client.post("/analyze", files={"resume": ("cv.txt", b"hello", "text/plain")}, data={"jd_text": JD})
    assert r.status_code == 415


def test_requires_jd(client):
    pdf = make_pdf(RESUME_LINES)
    r = client.post("/analyze", files={"resume": ("cv.pdf", pdf, "application/pdf")}, data={"jd_text": "short"})
    assert r.status_code == 422
