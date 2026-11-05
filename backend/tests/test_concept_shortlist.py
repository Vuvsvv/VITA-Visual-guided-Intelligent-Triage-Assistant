"""Semantic retrieval sends only a shortlist of official concepts once the local model is ready."""
import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.services import concept_shortlist as shortlist
from app.services import department_reasoning_service as service
from tests.test_semantic_department_retrieval import ankle_case, data

# The fake model treats these wordings as synonyms of the official concept.
SYNONYMS = {"拉肚子": "腹瀉", "疼痛": "關節痛"}


class FakeModel:
    def __init__(self, *_args, **_kwargs):
        self.calls: list[list[str]] = []
        self.axes: dict[str, int] = {}

    def embed(self, texts):
        self.calls.append(list(texts))
        for text in texts:
            axis = self.axes.setdefault(SYNONYMS.get(text, text), len(self.axes))
            vector = np.zeros(4096, dtype=np.float32)
            vector[axis] = 1.0
            yield vector


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=lambda *a, **k: model))
    monkeypatch.setattr(shortlist, "MODEL_CACHE_DIR", tmp_path)
    for name, value in (("_model", None), ("_model_name", ""), ("_vectors", {}), ("_lexical", None)):
        monkeypatch.setattr(shortlist, name, value)
    return model


def catalog():
    records, _ = data()
    return sorted({record["concept"] for record in records})


def test_shortlist_keeps_paraphrases_and_shared_characters(fake_model):
    concepts = catalog()
    shortlist.warm_up(concepts)
    assert shortlist.ready()
    picked = asyncio.run(shortlist.shortlist(["拉肚子", "眼睛紅紅的很癢"], concepts))
    assert "腹瀉" in picked  # meaning only, no shared characters
    assert "眼睛癢" in picked  # shared characters
    assert set(picked) <= set(concepts)
    assert len(picked) <= 2 * (shortlist.LEXICAL_TOP + shortlist.EMBEDDING_TOP) < len(concepts)


def test_semantic_prompt_carries_only_the_shortlist(fake_model):
    concepts = catalog()
    shortlist.warm_up(concepts)
    # The ankle case's phrases: the symptom and the touching "腳踝疼痛" fragment.
    picked = asyncio.run(shortlist.shortlist(["疼痛", "腳踝疼痛"], concepts))
    outside = next(concept for concept in concepts if concept not in picked)
    records, resolutions = data()
    matches = [dict(patient_source_text="疼痛", knowledge_concept=concept, confidence=.98)
               for concept in ("關節痛", outside)]
    provider = AsyncMock(side_effect=[
        json.dumps(dict(matches=matches), ensure_ascii=False),
        json.dumps(dict(judgments=[dict(patient_source_text="疼痛", knowledge_concept="關節痛",
                                        supported=True, reason="語意相符")]), ensure_ascii=False),
    ])
    with patch.object(service, "complete_runtime_json", provider):
        result = asyncio.run(service.retrieve_semantic_official_evidence(ankle_case(), records, resolutions))
    sent = json.loads(provider.await_args_list[0].args[0].split("\n", 1)[1])["official_concepts"]
    assert "關節痛" in sent and outside not in sent
    assert len(sent) <= shortlist.SHORTLIST_CAP < len(concepts)
    # The AI may only pick concepts it was shown.
    assert {item["knowledge_concept"] for item in result} == {"關節痛"}


def test_without_model_the_whole_catalog_is_sent():
    assert not shortlist.ready()
    assert asyncio.run(shortlist.shortlist(["疼痛"], catalog())) is None
    records, resolutions = data()
    provider = AsyncMock(return_value=json.dumps(dict(matches=[])))
    with patch.object(service, "complete_runtime_json", provider):
        asyncio.run(service.retrieve_semantic_official_evidence(ankle_case(), records, resolutions))
    sent = json.loads(provider.await_args.args[0].split("\n", 1)[1])["official_concepts"]
    assert len(sent) == len({r["concept"] for r in records})


def test_model_failure_keeps_the_whole_catalog(fake_model, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("download blocked")
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=broken))
    shortlist.warm_up(catalog())
    assert not shortlist.ready()


def test_empty_model_setting_disables_the_shortlist(fake_model, monkeypatch):
    monkeypatch.setenv("CONCEPT_EMBEDDING_MODEL", "")
    shortlist.warm_up(catalog())
    assert not shortlist.ready() and fake_model.calls == []


def test_vectors_are_cached_and_only_new_concepts_are_embedded(fake_model, monkeypatch):
    shortlist.warm_up(["腹瀉", "頭痛"])
    monkeypatch.setattr(shortlist, "_model", None)
    monkeypatch.setattr(shortlist, "_vectors", {})
    shortlist.warm_up(["腹瀉", "頭痛", "耳鳴"])
    assert fake_model.calls == [["腹瀉", "頭痛"], ["耳鳴"]]


def test_concept_outside_the_cached_vectors_falls_back(fake_model):
    shortlist.warm_up(["腹瀉"])
    many = ["腹瀉"] + [f"概念{i}" for i in range(shortlist.SHORTLIST_CAP + 1)]
    assert asyncio.run(shortlist.shortlist(["拉肚子"], many)) is None
