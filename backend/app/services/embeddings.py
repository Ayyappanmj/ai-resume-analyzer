"""Lightweight pure-Python TF-IDF similarity for resume and job-description text."""
import math
import re
from collections import Counter


def _tfidf_cosine(a: str, b: str) -> float:
    documents = [
        Counter(re.findall(r"[a-z][a-z+#.]{1,}", text.lower()))
        for text in (a, b)
    ]
    terms = documents[0].keys() | documents[1].keys()
    vectors: list[dict[str, float]] = [{}, {}]
    for term in terms:
        document_frequency = sum(term in document for document in documents)
        inverse_document_frequency = math.log(3 / (document_frequency + 1)) + 1
        for index, document in enumerate(documents):
            count = document.get(term, 0)
            if count:
                vectors[index][term] = count * inverse_document_frequency

    dot = sum(value * vectors[1].get(term, 0.0) for term, value in vectors[0].items())
    norm_a = math.sqrt(sum(value * value for value in vectors[0].values()))
    norm_b = math.sqrt(sum(value * value for value in vectors[1].values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def similarity(resume: str, jd: str) -> tuple[float, str]:
    """Return TF-IDF cosine similarity in [0, 1] and its method name."""
    return _tfidf_cosine(resume, jd), "tfidf"
