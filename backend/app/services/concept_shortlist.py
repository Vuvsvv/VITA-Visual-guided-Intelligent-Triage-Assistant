"""Shortlist official concepts before AI semantic retrieval so its prompt stays small.

The knowledge base keeps growing, so the semantic-retrieval prompt carries only the
concepts closest to the patient's words: for each patient phrase, the top character
n-gram (TF-IDF) matches plus the top local-embedding matches. The embedding model and
the concept vectors live in ``backend/.model_cache`` (git-ignored), are prepared once at
backend startup, and only new concepts are embedded after a knowledge-base update.
Without a loaded model ``shortlist`` returns ``None`` and the caller sends the whole
catalog, exactly as before. Recall is measured by ``benchmark/concept_shortlist_recall.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"
LEXICAL_TOP = 20
EMBEDDING_TOP = 30
# A turn may carry many phrases; ranks are interleaved across them up to this cap.
SHORTLIST_CAP = 150

_model = None
_model_name = ""
_vectors: dict[str, object] = {}
_lexical: tuple[tuple[str, ...], "_Lexical"] | None = None


def _grams(text: str) -> Counter:
    text = "".join(text.split()).lower()
    grams = Counter(text)
    grams.update(text[i:i + 2] for i in range(len(text) - 1))
    return grams


class _Lexical:
    """Character uni/bi-gram TF-IDF; catches wording that shares characters."""

    def __init__(self, concepts: tuple[str, ...]):
        self.concepts = concepts
        docs = [_grams(concept) for concept in concepts]
        df = Counter(gram for doc in docs for gram in doc)
        self.idf = {gram: math.log((len(docs) + 1) / (count + 1)) + 1 for gram, count in df.items()}
        self.vectors = [self._weigh(doc) for doc in docs]

    def _weigh(self, counts: Counter) -> dict[str, float]:
        vector = {gram: count * self.idf.get(gram, 0.0) for gram, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {gram: value / norm for gram, value in vector.items()}

    def top(self, query: str, k: int) -> list[str]:
        weights = self._weigh(_grams(query))
        scored = sorted(
            ((sum(w * vector.get(gram, 0.0) for gram, w in weights.items()), index)
             for index, vector in enumerate(self.vectors)),
            reverse=True,
        )
        return [self.concepts[index] for score, index in scored[:k] if score > 0]


def _cache_file(model_name: str) -> Path:
    return MODEL_CACHE_DIR / "concept_vectors" / (re.sub(r"[^A-Za-z0-9._-]", "_", model_name) + ".npz")


def _embed(texts: list[str]):
    import numpy as np

    vectors = np.array(list(_model.embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def warm_up(concepts: list[str]) -> None:
    """Load (downloading on first use) the model and embed concepts not cached yet.

    Any failure leaves the model unloaded, so retrieval falls back to the whole catalog.
    """
    global _model, _model_name, _vectors
    name = get_settings().concept_embedding_model.strip()
    if not name:
        return
    try:
        import numpy as np
        from fastembed import TextEmbedding

        if _model is None or _model_name != name:
            _model, _model_name, _vectors = TextEmbedding(name, cache_dir=str(MODEL_CACHE_DIR)), name, {}
        path = _cache_file(name)
        if not _vectors and path.exists():
            stored = np.load(path, allow_pickle=False)
            _vectors = dict(zip(stored["concepts"].tolist(), stored["vectors"]))
        missing = sorted(set(concepts) - set(_vectors))
        if missing:
            _vectors.update(zip(missing, _embed(missing)))
            path.parent.mkdir(parents=True, exist_ok=True)
            names = sorted(_vectors)
            np.savez(path, concepts=np.array(names), vectors=np.stack([_vectors[c] for c in names]))
    except Exception as exc:
        _model, _model_name, _vectors = None, "", {}
        logger.warning(
            "找症狀詞的模型載入失敗（%s: %s），語意比對改送整份詞彙清單，回覆會比較慢。",
            type(exc).__name__, exc,
        )


def ready() -> bool:
    return _model is not None


def _interleave(ranked: list[list[str]], cap: int) -> list[str]:
    picked: dict[str, None] = {}
    for rank in range(max(map(len, ranked), default=0)):
        for items in ranked:
            if rank < len(items):
                picked.setdefault(items[rank])
                if len(picked) >= cap:
                    return list(picked)
    return list(picked)


def _shortlist(queries: list[str], catalog: list[str]) -> list[str] | None:
    global _lexical
    import numpy as np

    concepts = tuple(concept for concept in catalog if concept in _vectors)
    if len(concepts) != len(catalog):
        return None
    if _lexical is None or _lexical[0] != concepts:
        _lexical = (concepts, _Lexical(concepts))
    matrix = np.stack([_vectors[concept] for concept in concepts])
    ranked = []
    for query, vector in zip(queries, _embed(queries)):
        ranked.append(_lexical[1].top(query, LEXICAL_TOP))
        ranked.append([concepts[i] for i in np.argsort(-(matrix @ vector))[:EMBEDDING_TOP]])
    return sorted(_interleave(ranked, SHORTLIST_CAP))


async def shortlist(queries: list[str], catalog: list[str]) -> list[str] | None:
    """Concepts of ``catalog`` closest to ``queries``; ``None`` means send the whole catalog."""
    queries = sorted({query.strip() for query in queries if query.strip()})
    if not ready() or not queries or len(catalog) <= SHORTLIST_CAP:
        return None
    try:
        return await asyncio.to_thread(_shortlist, queries, catalog)
    except Exception as exc:
        logger.warning("concept shortlist unavailable error_type=%s", type(exc).__name__)
        return None
