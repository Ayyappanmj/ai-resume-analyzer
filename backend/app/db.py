"""Optional PostgreSQL persistence. If DATABASE_URL is empty, everything is a no-op."""
import logging
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .config import get_settings
from .schemas import AnalysisResult, HistoryItem

log = logging.getLogger(__name__)
_engine = None


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    ats_score: Mapped[int] = mapped_column(Integer)
    similarity: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    payload: Mapped[dict] = mapped_column(JSON)


def init_db() -> None:
    global _engine
    url = get_settings().database_url
    if not url:
        return
    try:
        _engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
        log.info("Database ready")
    except Exception as exc:  # never break the API because of the optional DB
        log.warning("Database disabled: %s", exc)
        _engine = None


def enabled() -> bool:
    return _engine is not None


def save_analysis(result: AnalysisResult) -> int | None:
    if not _engine:
        return None
    try:
        with Session(_engine) as s:
            row = Analysis(
                filename=result.filename,
                ats_score=result.ats_score,
                similarity=result.similarity_score,
                created_at=result.created_at,
                payload=result.model_dump(mode="json"),
            )
            s.add(row)
            s.commit()
            return row.id
    except Exception as exc:
        log.warning("Could not save analysis: %s", exc)
        return None


def list_history(limit: int = 20) -> list[HistoryItem]:
    if not _engine:
        return []
    with Session(_engine) as s:
        rows = s.scalars(select(Analysis).order_by(Analysis.id.desc()).limit(limit)).all()
        return [
            HistoryItem(
                id=r.id,
                filename=r.filename,
                ats_score=r.ats_score,
                similarity_score=r.similarity,
                created_at=r.created_at,
            )
            for r in rows
        ]
