"""FastAPI application."""
import logging
import multiprocessing
import threading
import time
from contextlib import asynccontextmanager
from multiprocessing.connection import Connection

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
ANALYSIS_TIMEOUT_SECONDS = 60
ANALYSIS_WORKER_STARTUP_TIMEOUT_SECONDS = 90


def _process_analysis(
    resume_data: bytes,
    resume_filename: str,
    jd_text: str,
    jd_file_data: bytes | None,
    use_llm: bool,
) -> dict:
    if not resume_data.startswith(b"%PDF"):
        raise HTTPException(415, "Resume must be a valid PDF file.")
    try:
        text, pages = pdf_parser.extract_text_from_pdf(resume_data)
    except Exception as exc:
        raise HTTPException(422, f"Could not read PDF (encrypted or corrupt?): {exc}") from exc
    words = len(text.split())
    if words < 50:
        raise HTTPException(422, "Almost no text found. Scanned/image-only PDFs are not supported - export a text PDF.")

    jd = jd_text.strip()
    if not jd and jd_file_data is not None:
        if jd_file_data.startswith(b"%PDF"):
            try:
                jd = pdf_parser.extract_text_from_pdf(jd_file_data)[0].strip()
            except Exception as exc:
                raise HTTPException(422, f"Could not read job description PDF: {exc}") from exc
        else:
            jd = pdf_parser.normalize_text(jd_file_data.decode("utf-8", errors="ignore")).strip()
    if len(jd.split()) < 20:
        raise HTTPException(422, "Please provide a job description (at least ~20 words).")

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

    ai = llm.analyze_with_llm(text, jd, missing) if use_llm else None
    result = AnalysisResult(
        filename=resume_filename,
        ats_score=ats, grade=scoring.grade_for(ats), breakdown=breakdown,
        similarity_score=round(sim * 100, 1), similarity_method=method,
        word_count=words, pages=pages, contact=contact,
        sections_detected=detected, sections_missing=missing_sections,
        resume_skills=nlp.group_skills(sorted(resume_set)),
        jd_skills=jd_skills, matched_skills=matched, missing_skills=missing,
        keywords=keywords, jd_keywords=jd_keywords, missing_keywords=missing_kw_all[:15],
        summary=(ai or {}).get("summary") or llm.fallback_summary(
            filename=resume_filename, words=words, sections=detected, skills=sorted(resume_set),
            ats=ats, matched=matched, jd_skills=jd_skills),
        strengths=(ai or {}).get("strengths", []),
        weaknesses=(ai or {}).get("weaknesses", []),
        suggestions=suggestions,
        improved_bullets=(ai or {}).get("improved_bullets") or llm.fallback_bullets(matched),
        interview_questions=(ai or {}).get("interview_questions") or llm.fallback_questions(matched, missing),
        llm_used=ai is not None,
    )
    return result.model_dump(mode="json")


def _analysis_worker(connection: Connection) -> None:
    try:
        connection.send(("ready",))
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                connection.send(("ok", _process_analysis(*request)))
            except HTTPException as exc:
                connection.send(("http_error", exc.status_code, exc.detail))
            except Exception:
                log.exception("Resume analysis worker failed")
                connection.send(("error",))
    except EOFError:
        return
    finally:
        connection.close()


class AnalysisWorker:
    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None

    def start(self) -> None:
        with self._lock:
            self._start()

    def _start(self, deadline: float | None = None) -> None:
        parent, child = self._context.Pipe()
        process = self._context.Process(target=_analysis_worker, args=(child,), daemon=True)
        process.start()
        child.close()
        self._process, self._connection = process, parent
        startup_timeout = ANALYSIS_WORKER_STARTUP_TIMEOUT_SECONDS
        if deadline is not None:
            startup_timeout = max(0, deadline - time.monotonic())
        if not parent.poll(startup_timeout):
            self._stop()
            if deadline is not None:
                raise HTTPException(504, "Resume analysis exceeded its time limit.")
            raise RuntimeError("Analysis worker did not start before the startup deadline.")
        try:
            message = parent.recv()
        except EOFError as exc:
            self._stop()
            raise RuntimeError("Analysis worker exited during startup.") from exc
        if message != ("ready",):
            self._stop()
            raise RuntimeError(f"Analysis worker failed to start: {message!r}")

    def run(self, *payload: object) -> dict:
        deadline = time.monotonic() + ANALYSIS_TIMEOUT_SECONDS
        if not self._lock.acquire(timeout=ANALYSIS_TIMEOUT_SECONDS):
            raise HTTPException(504, "Resume analysis exceeded its time limit.")
        try:
            if self._process is None or not self._process.is_alive():
                self._stop()
                self._start(deadline)
            connection = self._connection
            if connection is None:
                raise RuntimeError("Analysis worker is not available.")
            try:
                connection.send(payload)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not connection.poll(remaining):
                    self._stop()
                    raise HTTPException(504, "Resume analysis exceeded its time limit.")
                message = connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop()
                raise RuntimeError("Analysis worker exited unexpectedly.") from exc

            if message[0] == "ok":
                return message[1]
            if message[0] == "http_error":
                raise HTTPException(message[1], message[2])
            self._stop()
            raise RuntimeError("Analysis worker failed while processing the resume.")
        finally:
            self._lock.release()

    def _stop(self) -> None:
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        if process is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()

    def close(self) -> None:
        with self._lock:
            self._stop()


analysis_worker = AnalysisWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    analysis_worker.start()
    try:
        yield
    finally:
        analysis_worker.close()


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


@app.get("/")
def root():
    return {"message": "AI Resume Analyzer API is running", "status": "healthy", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_configured": llm.is_configured(),
        "model": llm.MODEL,
        "database": db.enabled(),
    }


@app.post("/analyze", response_model=AnalysisResult)
def analyze(
    resume: UploadFile = File(..., description="Resume PDF"),
    jd_text: str = Form("", description="Job description text"),
    jd_file: UploadFile | None = File(None, description="Job description (.pdf or .txt)"),
    use_llm: bool = Form(True),
):
    s = get_settings()
    resume_data = _read_upload(resume, s.max_upload_mb)
    jd_file_data = None
    if not jd_text.strip() and jd_file is not None and jd_file.filename:
        jd_file_data = _read_upload(jd_file, s.max_upload_mb)
    try:
        data = analysis_worker.run(
            resume_data,
            resume.filename or "resume.pdf",
            jd_text,
            jd_file_data,
            use_llm,
        )
        result = AnalysisResult.model_validate(data)
        result.id = db.save_analysis(result)
        return result
    except HTTPException:
        raise
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
