"""Official Taipei VGH pediatric symptom pages (fid 6747-6750) route to 兒童內科 / 兒童腸胃科."""
from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_reasoning_service as reasoning
from app.services.department_knowledge import load_department_knowledge, resolve_department_names

_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]
RESOLUTIONS = resolve_department_names(RECORDS, ACTIVE)
PEDIATRICS, PEDIATRIC_GI = 1320, 1322


def literal_departments(text):
    case = TriageCase(case_id="pediatric-literal")
    case.history_records = [Message(role="user", content=text)]
    case.semantic_extractions = [SemanticExtraction(
        field="symptom", normalized_value=text, semantic_status="available", assertion="present",
        confidence=0.95, source_text=text, extractor="ai",
    )]
    rows = reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    return {row["knowledge_concept"]: row["dept_id"] for row in rows}


def test_pediatric_departments_have_official_concepts():
    by_department = {}
    for record in RECORDS:
        by_department.setdefault(record["canonical_dept_id"], set()).add(record["concept"])
    # 2026-11-03 ihealth pages that send a child to 兒童醫學部 add child-only disease names.
    assert by_department[PEDIATRICS] >= {
        "小兒發燒", "小孩發燒", "熱痙攣", "川崎症", "兒童夜尿症", "兒童夜尿", "喉骨軟化症", "新生兒鎖骨骨折",
    }
    # 2026-11-04 兒童醫學部 doctor keywords: only the ones that name a child (「兒童發燒」 is locked, 4.6).
    docsearch = {record["concept"] for record in RECORDS if record["canonical_dept_id"] == PEDIATRICS
                 and record["source_id"].startswith("vghtpe_docsearch_more_")}
    assert {"兒童疝氣", "新生兒黃疸", "兒童癲癇", "嬰兒健檢"} <= docsearch
    assert all(any(mark in concept for mark in ("兒童", "小兒", "新生兒", "早產兒", "嬰", "幼"))
               for concept in docsearch)
    assert "兒童發燒" not in docsearch
    assert by_department[PEDIATRIC_GI] == {"牛奶過敏", "腸絞痛", "嬰兒哭鬧", "小兒便秘", "嬰兒便秘"}


def test_child_wording_matches_pediatric_departments_literally():
    assert literal_departments("小孩發燒")["小孩發燒"] == PEDIATRICS
    assert literal_departments("嬰兒便秘")["嬰兒便秘"] == PEDIATRIC_GI


def test_adult_wording_never_reaches_pediatric_departments():
    for text in ("腹瀉", "便秘", "發燒", "腹脹"):
        assert not {PEDIATRICS, PEDIATRIC_GI} & set(literal_departments(text).values()), text
