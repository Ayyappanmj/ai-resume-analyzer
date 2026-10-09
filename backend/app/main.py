"""FastAPI application."""
import logging
import os
from contextlib import asynccontextmanager

# Limit numerical-library thread pools before importing spaCy, NumPy, or PyTorch.
for _thread_limit in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_limit] = "1"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import db
from .config import get_settings
from .schemas import AnalysisResult, HistoryItem
from .services import embeddings, llm, nlp, pdf_parser, report, scoring
from .services import sections as sec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("resume-analyzer")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    nlp.get_nlp()  # warm up
    if get_settings().use_embeddings:
        embeddings.preload_model()
    yield


app = FastAPI(title="AI Resume Analyzer", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload(f: UploadFile, max_mb: int) -> bytes:
    data = f.file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {max_mb} MB).")
    return data


def _jd_from_file(f: UploadFile, max_mb: int) -> str:
    data = _read_upload(f, max_mb)
    if data.startswith(b"%PDF"):
        try:
            return pdf_parser.extract_text_from_pdf(data)[0]
        except Exception as exc:
            raise HTTPException(422, f"Could not read job description PDF: {exc}")
    return pdf_parser.normalize_text(data.decode("utf-8", errors="ignore"))


@app.get("/")
def root():
    return {"message": "AI Resume Analyzer API is running", "status": "healthy", "docs": "/docs"}


@app.get("/health")
def health():
    s = get_settings()
    ollama_ok = False
    try:
        import httpx

        ollama_ok = httpx.get(f"{s.ollama_host}/api/tags", timeout=2).status_code == 200
    except Exception:
        pass
    return {"status": "ok", "ollama": ollama_ok, "model": s.ollama_model, "database": db.enabled()}


@app.post("/analyze", response_model=AnalysisResult)
def analyze(
    resume: UploadFile = File(..., description="Resume PDF"),
    jd_text: str = Form("", description="Job description text"),
    jd_file: UploadFile | None = File(None, description="Job description (.pdf or .txt)"),
    use_llm: bool = Form(True),
):
    s = get_settings()

    data = _read_upload(resume, s.max_upload_mb)
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Resume must be a valid PDF file.")
    try:
        text, pages = pdf_parser.extract_text_from_pdf(data)
    except Exception as exc:
        raise HTTPException(422, f"Could not read PDF (encrypted or corrupt?): {exc}")
    words = len(text.split())
    if words < 50:
        raise HTTPException(422, "Almost no text found. Scanned/image-only PDFs are not supported - export a text PDF.")

    jd = jd_text.strip()
    if not jd and jd_file is not None and jd_file.filename:
        jd = _jd_from_file(jd_file, s.max_upload_mb).strip()
    if len(jd.split()) < 20:
        raise HTTPException(422, "Please provide a job description (at least ~20 words).")

    try:
        # --- NLP ---
        sections = sec.detect_sections(text)
        detected, missing_sections = sec.detected_names(sections), sec.missing_core(sections)
        contact = nlp.extract_contact(text)

        resume_counts = nlp.extract_skills(text)
        jd_counts = nlp.extract_skills(jd)
        resume_set = set(resume_counts)
        jd_skills = sorted(jd_counts, key=lambda k: (-jd_counts[k], k))
        matched = [k for k in jd_skills if k in resume_set]
        missing = [k for k in jd_skills if k not in resume_set]

        keywords = nlp.extract_keywords(text, 20)
        jd_keywords = nlp.extract_keywords(jd, 25)
        lower = text.lower()
        missing_kw_all = [k for k in jd_keywords if k not in lower]
        coverage = 1 - len(missing_kw_all) / len(jd_keywords) if jd_keywords else 0.0

        sim, method = embeddings.similarity(text, jd)

        ats, breakdown = scoring.compute_ats(
            text=text, sections_detected=detected, contact=contact, matched=matched,
            jd_skills=jd_skills, resume_skills=sorted(resume_set), keyword_coverage=coverage,
            similarity=sim, method=method,
        )
        suggestions = scoring.build_suggestions(
            items=breakdown, missing_skills=missing, missing_sections=missing_sections,
            missing_keywords=missing_kw_all, contact=contact,
        )

        # --- LLM (optional, graceful fallback) ---
        ai = llm.analyze_with_llm(text, jd, missing) if use_llm else None
        result = AnalysisResult(
            filename=resume.filename or "resume.pdf",
            ats_score=ats, grade=scoring.grade_for(ats), breakdown=breakdown,
            similarity_score=round(sim * 100, 1), similarity_method=method,
            word_count=words, pages=pages, contact=contact,
            sections_detected=detected, sections_missing=missing_sections,
            resume_skills=nlp.group_skills(sorted(resume_set)),
            jd_skills=jd_skills, matched_skills=matched, missing_skills=missing,
            keywords=keywords, jd_keywords=jd_keywords, missing_keywords=missing_kw_all[:15],
            summary=(ai or {}).get("summary") or llm.fallback_summary(
                filename=resume.filename or "", words=words, sections=detected, skills=sorted(resume_set),
                ats=ats, matched=matched, jd_skills=jd_skills),
            strengths=(ai or {}).get("strengths", []),
            weaknesses=(ai or {}).get("weaknesses", []),
            suggestions=suggestions,
            improved_bullets=(ai or {}).get("improved_bullets") or llm.fallback_bullets(matched),
            interview_questions=(ai or {}).get("interview_questions") or llm.fallback_questions(matched, missing),
            llm_used=ai is not None,
        )
        result.id = db.save_analysis(result)
        return result
    except Exception as exc:
        log.exception("Resume analysis failed")
        raise HTTPException(500, "Resume analysis failed. Please try again.") from exc


@app.post("/report", response_class=Response)
def make_report(result: AnalysisResult):
    pdf = report.build_pdf(result)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="resume-analysis-report.pdf"'})


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20):
    return db.list_history(min(max(limit, 1), 100))
