from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class VariantRng(Protocol):
    def choice(self, values: tuple[str, ...]) -> str: ...


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    state_field: str
    variants: tuple[str, ...]
    safety_level: str = "standard"

    def __post_init__(self) -> None:
        if len(self.variants) != 5:
            raise ValueError(f"QuestionSpec {self.question_id} must define exactly 5 variants")

    @property
    def canonical_text(self) -> str:
        return self.variants[0]


QUESTION_SPECS: tuple[QuestionSpec, ...] = (
    QuestionSpec(
        question_id="symptom",
        state_field="symptom",
        variants=(
            "請描述目前最主要的不舒服症狀。",
            "想先了解一下，你現在最主要是哪裡不舒服？",
            "請問目前最困擾你的症狀是什麼？",
            "方便說明一下，你這次最想處理的不舒服嗎？",
            "請告訴我，目前最主要的身體不適是什麼？",
        ),
    ),
    QuestionSpec(
        question_id="emergency_symptoms",
        state_field="red_flags",
        safety_level="critical",
        variants=(
            "請問是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛等急迫症狀？",
            "想確認安全狀況：目前是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
            "請確認一下，你現在有沒有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
            "為了確認是否需要立即處理，請問有無突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
            "接著確認急迫症狀：你是否出現突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
        ),
    ),
    QuestionSpec(
        question_id="body_part",
        state_field="body_part",
        variants=(
            "請問症狀主要發生在哪個部位？",
            "想了解一下，這個不舒服主要在身體哪裡？",
            "請問你主要是哪個部位感到不適？",
            "方便告訴我，症狀集中在哪個位置嗎？",
            "這個症狀主要影響身體的哪個部位呢？",
        ),
    ),
    QuestionSpec(
        question_id="duration",
        state_field="duration",
        variants=(
            "請問這個症狀大約持續多久了？",
            "想了解一下，這個不舒服大概持續多久了呢？",
            "這個症狀大約是從什麼時候開始的？",
            "方便告訴我，這個情況已經持續多久了嗎？",
            "請問你大概多久以前開始出現這個症狀？",
        ),
    ),
    QuestionSpec(
        question_id="severity",
        state_field="severity",
        variants=(
            "請問症狀程度是輕微、中等、明顯，還是很痛或影響生活？",
            "想了解一下，目前不舒服的程度大約是輕微、中等還是嚴重？",
            "請問這個症狀現在是輕微、普通，還是嚴重？",
            "這個不舒服目前是輕微、普通，還是嚴重到影響日常生活？",
            "方便描述一下，症狀目前是輕微、普通，還是相當嚴重嗎？",
        ),
    ),
    QuestionSpec(
        question_id="preferred_days",
        state_field="preferred_days",
        variants=(
            "請問你最近哪幾天有空就醫？",
            "想了解一下，你近期哪些日子方便看診？",
            "請問最近有哪一天比較方便就醫？",
            "方便告訴我，你接下來哪些日期可以看診嗎？",
            "你近期較方便安排就醫的日子是哪幾天呢？",
        ),
    ),
    QuestionSpec(
        question_id="preferred_sessions",
        state_field="preferred_sessions",
        variants=(
            "請問你偏好的看診時段是上午、下午還是夜間？",
            "想確認一下，你看診比較方便的時段是上午、下午或夜間？",
            "請問你偏好安排上午、下午，還是夜間門診？",
            "方便告訴我，你比較適合上午、下午或夜間看診嗎？",
            "你希望看診時間安排在上午、下午還是夜間呢？",
        ),
    ),
)

QUESTION_SPEC_BY_ID = {spec.question_id: spec for spec in QUESTION_SPECS}
QUESTION_SPEC_BY_FIELD = {spec.state_field: spec for spec in QUESTION_SPECS}


def question_spec_for_field(state_field: str) -> QuestionSpec:
    try:
        return QUESTION_SPEC_BY_FIELD[state_field]
    except KeyError as exc:
        raise ValueError(f"Unsupported checklist field: {state_field}") from exc


def select_question_variant(question_id: str, rng: VariantRng | None = None) -> str:
    try:
        spec = QUESTION_SPEC_BY_ID[question_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported question_id: {question_id}") from exc
    return (rng or random).choice(spec.variants)
