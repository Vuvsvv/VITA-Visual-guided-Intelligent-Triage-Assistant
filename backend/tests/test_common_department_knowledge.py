"""Taipei VGH symptom pages for common complaints (fid 6698-6746, added 2026-11-03)."""
import json

from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_reasoning_service as reasoning
from app.services.department_knowledge import KNOWLEDGE_DIR, load_department_knowledge, resolve_department_names

_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]
RESOLUTIONS = resolve_department_names(RECORDS, ACTIVE)
NAMES = {record["canonical_dept_id"]: record["department_name"] for record in RECORDS}
SUPPLEMENTS = [
    KNOWLEDGE_DIR / f"supplemental_guidance_20261103_{name}.json"
    for name in ("chest", "dermatology", "ent", "eye", "gastroenterology", "general", "neurology",
                 "orthopedics", "urology_gynecology")
]


def literal_departments(text):
    case = TriageCase(case_id="common-literal")
    case.history_records = [Message(role="user", content=text)]
    case.semantic_extractions = [SemanticExtraction(
        field="symptom", normalized_value=text, semantic_status="available", assertion="present",
        confidence=0.95, source_text=text, extractor="ai",
    )]
    return {NAMES[row["dept_id"]] for row in reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)}


def test_supplements_cite_symptom_pages_and_skip_family_medicine():
    for path in SUPPLEMENTS:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "mappings" not in data
        for source in data["sources"]:
            assert source["source_url"].startswith("https://www.vghtpe.gov.tw/vghtpe/Fpage.action?fid=")
        for record in data["records"]:
            assert record["evidence_type"] == "direct_routing_evidence"
            assert record["department_name"] != "家庭醫學科(一般門診/戒菸)"


def test_common_complaints_reach_the_listed_specialties():
    assert literal_departments("皮膚癢") >= {"皮膚科"}
    # 高雄榮總「該看哪一科?」lists 蕁麻疹 under 一般內科 too (2026-11-04).
    assert literal_departments("身上起蕁麻疹") == {"皮膚科", "過敏免疫風濕", "一般內科"}
    # 三軍總醫院 lists 腹脹 under 大腸直腸外科 and 義大醫院 under 一般醫學內科 too (2026-11-04).
    assert literal_departments("最近一直腹脹") == {"胃腸肝膽科", "直腸外科", "一般內科"}
    assert literal_departments("胃灼熱") == {"胃腸肝膽科"}
    assert literal_departments("喘不過氣") == {"胸腔內科", "心臟內科"}
    # 雙和醫院 lists 背痛 under 神經科 too (2026-11-04).
    assert literal_departments("背痛") == {"骨科", "復健醫學", "神經外科", "神經內科"}
    assert literal_departments("關節酸痛") == {"骨科", "復健醫學", "過敏免疫風濕"}
    # 義大醫院 lists 喉嚨痛 under 一般醫學內科, and 眩暈(天旋地轉) under 耳鼻喉科 (2026-11-04).
    assert literal_departments("喉嚨痛") == {"感染科", "喉科", "一般內科"}
    assert literal_departments("流鼻血") == {"鼻科"}
    assert literal_departments("耳朵疼痛") == {"耳科"}
    assert literal_departments("天旋地轉") == {"神經內科", "耳科"}
    # 臺北榮總復健醫學部 doctors list 手腳麻木 too (醫師及專長查詢, 2026-11-04).
    assert literal_departments("手腳麻木") == {"神經內科", "神經外科", "復健醫學"}
    # 屏東榮總「該看哪一科?」lists 解尿困難 under 婦產科 and 腎臟科 too (2026-11-04).
    assert literal_departments("解尿困難") == {"泌尿外科", "婦產科", "腎臟科"}
    assert literal_departments("眼屎很多") == {"眼科"}
    # 乳房特別門診 / 心臟血管外科 follow the Taichung VGH symptom table (breast pain → 一般外科;
    # plain numbness is not a 心臟外科 symptom there), mapped onto existing departments.
    assert literal_departments("乳房脹痛") == {"婦產科", "一般外科"}
    assert literal_departments("周邊血管阻塞") == {"心臟外科"}


def test_vghtpe_doctor_keywords_cover_buttock_and_elbow_pain_without_new_clues():
    # Taipei VGH 醫師及專長查詢, full department lists (2026-11-04): 復健醫學部 lists 臀部疼痛 and 手肘疼痛.
    assert literal_departments("臀部疼痛") == {"復健醫學"}
    assert literal_departments("手肘疼痛") == {"復健醫學"}
    path = KNOWLEDGE_DIR / "supplemental_guidance_20261104_vghtpe_docsearch_full.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for source in data["sources"]:
        assert source["source_url"].startswith("https://www.vghtpe.gov.tw/docsearch.action?more=")
    locked = {"頭痛", "頭暈", "胸痛", "胸悶", "心悸", "咳嗽", "發燒", "腹瀉", "腹痛", "拉肚子", "肚子痛"}
    for record in data["records"]:
        # Doctor specialties only score; they never become clue questions.
        assert record["evidence_type"] == "department_specialty_evidence"
        assert record["department_name"] != "家庭醫學科(一般門診/戒菸)"
        assert record["concept"] not in locked


def test_departments_awaiting_a_310_decision_are_not_mapped():
    names = {record["department_name"] for record in RECORDS}
    assert not {"乳房特別門診", "神經修復科", "心臟血管外科", "耳鼻喉科"} & names
