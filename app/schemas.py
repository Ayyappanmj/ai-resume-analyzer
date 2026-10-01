from datetime import datetime

from pydantic import BaseModel, Field


class ScoreItem(BaseModel):
    name: str
    score: float
    max: float
    detail: str = ""


class AnalysisResult(BaseModel):
    id: int | None = None
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    ats_score: int
    grade: str
    breakdown: list[ScoreItem]

    similarity_score: float = Field(description="Raw cosine similarity resume vs JD, 0-100")
    similarity_method: str

    word_count: int
    pages: int
    contact: dict[str, str] = {}

    sections_detected: list[str]
    sections_missing: list[str]

    resume_skills: dict[str, list[str]]
    jd_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]

    keywords: list[str]
    jd_keywords: list[str]
    missing_keywords: list[str]

    summary: str
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    improved_bullets: list[str] = []
    interview_questions: list[str] = []
    llm_used: bool = False


class HistoryItem(BaseModel):
    id: int
    filename: str
    ats_score: int
    similarity_score: float
    created_at: datetime
