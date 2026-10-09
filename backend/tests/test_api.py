import io

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
    monkeypatch.setattr(embeddings, "similarity", lambda a, b: (0.55, "sentence-transformers"))
    monkeypatch.setattr(llm, "analyze_with_llm", lambda *a, **k: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize(("enabled", "expected_calls"), [(False, 0), (True, 1)])
def test_embedding_model_preload_follows_configuration(monkeypatch, enabled, expected_calls):
    settings = main.get_settings().model_copy(update={"use_embeddings": enabled})
    preloaded = []
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main.embeddings, "preload_model", lambda: preloaded.append(True))

    with TestClient(main.app):
        pass

    assert len(preloaded) == expected_calls


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


def test_analysis_error_returns_json_500(client, monkeypatch):
    def fail_analysis(_text):
        raise RuntimeError("internal processing failure")

    monkeypatch.setattr(main.nlp, "extract_contact", fail_analysis)
    pdf = make_pdf(RESUME_LINES)
    r = client.post("/analyze", files={"resume": ("cv.pdf", pdf, "application/pdf")},
                    data={"jd_text": JD, "use_llm": "false"})

    assert r.status_code == 500
    assert r.json() == {"detail": "Resume analysis failed. Please try again."}
