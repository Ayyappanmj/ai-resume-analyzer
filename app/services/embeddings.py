"""Resume <-> job description similarity with Sentence Transformers (lexical fallback)."""
import logging
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def _chunks(text: str, words: int = 180) -> list[str]:
    toks = text.split()
    return [" ".join(toks[i:i + words]) for i in range(0, len(toks), words)] or [text]


def _embed(text: str) -> np.ndarray:
    vecs = _model().encode(_chunks(text), normalize_embeddings=True, convert_to_numpy=True)
    v = vecs.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n else v


def _lexical(a: str, b: str) -> float:
    ta = Counter(re.findall(r"[a-z][a-z+#.]{1,}", a.lower()))
    tb = Counter(re.findall(r"[a-z][a-z+#.]{1,}", b.lower()))
    dot = sum(ta[k] * tb[k] for k in ta.keys() & tb.keys())
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity(resume: str, jd: str) -> tuple[float, str]:
    """Return (cosine in [0,1], method name)."""
    try:
        return max(0.0, float(np.dot(_embed(resume), _embed(jd)))), "sentence-transformers"
    except Exception as exc:
        log.warning("Embedding model unavailable (%s) - using lexical fallback", exc)
        return _lexical(resume, jd), "lexical-fallback"
