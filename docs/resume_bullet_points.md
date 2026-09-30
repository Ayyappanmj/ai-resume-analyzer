# Resume bullet points for this project

Use these if you want to list this project on your own resume.

- Designed and built a full-stack AI resume analyzer (FastAPI + Streamlit) that scores resumes
  against job descriptions using a transparent, weighted 0-100 ATS algorithm.
- Implemented NLP-based skill-gap and keyword-coverage detection with spaCy across a 90+ term
  taxonomy spanning 8 skill categories.
- Integrated Sentence-Transformers embeddings for resume-to-job-description semantic similarity,
  with an automatic lexical-overlap fallback for offline resilience.
- Integrated a locally-hosted Llama 3.1 model (via Ollama) to generate resume summaries, bullet
  rewrites, and tailored interview questions, with rule-based fallbacks when the LLM is unavailable.
- Built a PDF report generator (ReportLab) producing a shareable, formatted analysis document.
- Containerized the full stack (FastAPI, Streamlit, PostgreSQL, Ollama) with Docker Compose and
  set up a GitHub Actions CI pipeline running automated pytest suites and Docker image builds
  on every push.
- Wrote unit and integration tests (pytest, FastAPI TestClient) covering NLP extraction, scoring
  logic, and end-to-end API behavior, keeping the suite fast and network-independent via mocking.
