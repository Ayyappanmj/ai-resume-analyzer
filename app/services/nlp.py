"""spaCy-powered skill extraction, keyword extraction and contact parsing."""
import logging
import re
from collections import Counter
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher

from ..config import get_settings
from .skills import PATTERN_TO_SKILL, SKILL_TO_CATEGORY

log = logging.getLogger(__name__)

GENERIC_WORDS = {
    "experience", "work", "team", "year", "years", "ability", "skill", "skills", "role", "job",
    "company", "candidate", "requirement", "requirements", "responsibility", "responsibilities",
    "knowledge", "opportunity", "position", "level", "field", "world", "way", "time", "day",
    "environment", "plus", "etc", "well", "strong", "excellent", "good", "great", "new", "use",
    "working", "using", "related", "including", "preferred", "required", "minimum", "degree",
    "benefit", "salary", "location", "equal", "employer", "application", "people", "help",
}


@lru_cache
def get_nlp():
    name = get_settings().spacy_model
    try:
        return spacy.load(name, disable=["ner"])
    except OSError:
        log.warning("spaCy model '%s' missing (run: python -m spacy download %s). "
                    "Falling back to a blank English pipeline - keyword quality will be lower.", name, name)
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


@lru_cache
def _skill_matcher() -> PhraseMatcher:
    nlp = get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    by_skill: dict[str, list[str]] = {}
    for pattern, skill in PATTERN_TO_SKILL.items():
        by_skill.setdefault(skill, []).append(pattern)
    for skill, patterns in by_skill.items():
        matcher.add(skill, [nlp.make_doc(p) for p in patterns])
    return matcher


def extract_skills(text: str) -> dict[str, int]:
    """Return {canonical_skill: mention_count}."""
    nlp = get_nlp()
    doc = nlp.make_doc(text)  # tokenizer only - matcher does not need the full pipeline
    counts: Counter[str] = Counter()
    for match_id, _start, _end in _skill_matcher()(doc):
        counts[nlp.vocab.strings[match_id]] += 1
    return dict(counts)


def group_skills(skills: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for s in skills:
        grouped.setdefault(SKILL_TO_CATEGORY.get(s, "Other"), []).append(s)
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """Frequency-ranked noun phrases / nouns (lemmatised), excluding generic filler."""
    doc = get_nlp()(text[:20000])
    counter: Counter[str] = Counter()
    tagged = len(doc) > 0 and doc[0].pos_ != ""

    if tagged and doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            toks = [t for t in chunk if t.is_alpha and not t.is_stop]
            if not toks or len(toks) > 3:
                continue
            phrase = " ".join((t.lemma_ or t.text).lower() for t in toks)
            if len(phrase) > 2 and phrase not in GENERIC_WORDS:
                counter[phrase] += 2 if len(toks) > 1 else 1

    for t in doc:
        if not t.is_alpha or t.is_stop or len(t.text) < 3:
            continue
        if tagged and t.pos_ not in {"NOUN", "PROPN"}:
            continue
        word = (t.lemma_ or t.text).lower()
        if word not in GENERIC_WORDS:
            counter[word] += 1

    # Known skills always outrank generic words.
    for skill, n in extract_skills(text).items():
        counter[skill.lower()] += 3 * n

    # Drop unigrams already covered by a chosen multi-word phrase.
    ranked = [k for k, _ in counter.most_common(top_n * 3)]
    chosen: list[str] = []
    for k in ranked:
        if " " not in k and any(k in c.split() for c in chosen):
            continue
        chosen.append(k)
        if len(chosen) == top_n:
            break
    return chosen


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}")
_LINKEDIN = re.compile(r"linkedin\.com/[\w\-/%]+", re.I)
_GITHUB = re.compile(r"github\.com/[\w\-]+", re.I)


def extract_contact(text: str) -> dict[str, str]:
    head = "\n".join(text.splitlines()[:15])
    out: dict[str, str] = {}
    if m := _EMAIL.search(text):
        out["email"] = m.group(0)
    if m := _PHONE.search(head):
        if len(re.sub(r"\D", "", m.group(0))) >= 9:
            out["phone"] = m.group(0).strip()
    if m := _LINKEDIN.search(text):
        out["linkedin"] = m.group(0)
    if m := _GITHUB.search(text):
        out["github"] = m.group(0)
    return out
