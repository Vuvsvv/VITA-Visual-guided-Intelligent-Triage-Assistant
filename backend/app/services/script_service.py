from __future__ import annotations

from app.schemas import RecommendationItem, ScriptStep

SCRIPT_ID = "vgh_booking_001"


def build_navigation_script(recommendation: RecommendationItem) -> list[ScriptStep]:
    doctor_target = f"{recommendation.slot}{recommendation.doctor}" if recommendation.slot else recommendation.doctor
    return [
        ScriptStep(
            action="open_app",
            target="台北榮民總醫院app",
            text="台北榮民總醫院app",
            description="開啟台北榮民總醫院掛號 App",
            delay_ms=1200,
            retry=1,
        ),
        _click_step("行動掛號", "點擊首頁的行動掛號入口"),
        _click_step("繼續掛號", "進入掛號流程"),
        _click_step("依門診科別", "選擇依門診科別掛號"),
        _click_step(recommendation.parentDept, "選擇上層科別"),
        _click_step(recommendation.childDept, "選擇建議科別"),
        _click_step("選擇看診時間/醫師", "開啟看診時間與醫師清單"),
        _click_step(recommendation.date, "選擇推薦看診日期"),
        _click_step(doctor_target, "選擇推薦診間與醫師"),
        _click_step("填寫個人資料", "進入個人資料填寫頁"),
    ]


def _click_step(target: str, description: str) -> ScriptStep:
    return ScriptStep(
        action="click",
        target=target,
        text=target,
        class_name="android.widget.TextView",
        description=description,
        delay_ms=500,
        retry=2,
    )
