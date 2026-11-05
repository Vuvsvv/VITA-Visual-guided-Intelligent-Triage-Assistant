"""Psychiatry, colorectal, orthopedic, dental and thoracic-surgery evidence (added 2026-11-03)."""
import json

from app.services.department_clues import clue_concepts
from app.services.department_knowledge import KNOWLEDGE_DIR

from test_common_department_knowledge import RECORDS, literal_departments

TAICHUNG = KNOWLEDGE_DIR / "supplemental_guidance_20261103_vghtc_symptom_query.json"


def test_taichung_table_is_a_marked_fallback_without_family_medicine():
    data = json.loads(TAICHUNG.read_text(encoding="utf-8"))
    assert [source["source_url"] for source in data["sources"]] == ["https://www.vghtc.gov.tw/SymptomQuery/605"]
    for record in data["records"]:
        assert record["fallback_for_vghtpe"] is True
        assert record["source_priority"] == 3
        assert record["evidence_type"] == "direct_routing_evidence"
        assert record["department_name"] != "家庭醫學科(一般門診/戒菸)"


def test_specialty_complaints_reach_their_department():
    assert literal_departments("最近很憂鬱") == {"精神科"}
    assert literal_departments("我很焦慮") == {"一般內科", "精神科"}
    # 鳳林分院 lists 痔瘡 under 一般外科 too (2026-11-04).
    assert literal_departments("痔瘡在痛") == {"直腸外科", "一般外科"}
    # 三軍總醫院 lists 便血 under 一般外科 and 義大醫院 under 一般醫學內科 too (2026-11-04).
    assert literal_departments("便血") == {"胃腸肝膽科", "直腸外科", "一般外科", "一般內科"}
    assert literal_departments("腳踝扭傷") == {"骨科"}
    assert literal_departments("手骨折") == {"骨科"}
    assert literal_departments("牙周病") == {"牙科"}
    assert literal_departments("氣胸") == {"胸腔外科"}
    assert literal_departments("心窩灼熱") == {"胃腸肝膽科", "胸腔外科"}


def test_psychiatry_concepts_can_now_be_asked():
    psychiatry, internal = 1251, 1232
    concepts = {clue.concept for clue in clue_concepts(RECORDS, {psychiatry, internal}, set(), [])}
    assert {"情緒低落", "過度擔心", "入睡困難"} <= concepts
    # Self-harm wording stays specialty-only, so it is never asked as a clue.
    assert not {"自殺意念", "自殺想法"} & concepts
