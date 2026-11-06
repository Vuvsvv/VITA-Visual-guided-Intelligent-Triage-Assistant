package com.example.medicalaiguidance.util

import com.example.medicalaiguidance.R

/** Exact-match lookup for the REQUIRED pre-recorded triage prompts. */
object FixedTriageAudioResolver {
    private data class AudioResources(
        val chinese: Int,
        val taiwanese: Int
    )

    private val resourcesByText = mapOf(
        "請描述目前最主要的不舒服症狀。" to AudioResources(R.raw.triage_symptom_01_zh, R.raw.triage_symptom_01_taigi),
        "想先了解一下，你現在最主要是哪裡不舒服？" to AudioResources(R.raw.triage_symptom_02_zh, R.raw.triage_symptom_02_taigi),
        "請問目前最困擾你的症狀是什麼？" to AudioResources(R.raw.triage_symptom_03_zh, R.raw.triage_symptom_03_taigi),
        "方便說明一下，你這次最想處理的不舒服嗎？" to AudioResources(R.raw.triage_symptom_04_zh, R.raw.triage_symptom_04_taigi),
        "請告訴我，目前最主要的身體不適是什麼？" to AudioResources(R.raw.triage_symptom_05_zh, R.raw.triage_symptom_05_taigi),
        "請問是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛等急迫症狀？" to AudioResources(R.raw.triage_emergency_symptoms_01_zh, R.raw.triage_emergency_symptoms_01_taigi),
        "想確認安全狀況：目前是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？" to AudioResources(R.raw.triage_emergency_symptoms_02_zh, R.raw.triage_emergency_symptoms_02_taigi),
        "請確認一下，你現在有沒有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？" to AudioResources(R.raw.triage_emergency_symptoms_03_zh, R.raw.triage_emergency_symptoms_03_taigi),
        "為了確認是否需要立即處理，請問有無突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？" to AudioResources(R.raw.triage_emergency_symptoms_04_zh, R.raw.triage_emergency_symptoms_04_taigi),
        "接著確認急迫症狀：你是否出現突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？" to AudioResources(R.raw.triage_emergency_symptoms_05_zh, R.raw.triage_emergency_symptoms_05_taigi),
        "請問症狀主要發生在哪個部位？" to AudioResources(R.raw.triage_body_part_01_zh, R.raw.triage_body_part_01_taigi),
        "想了解一下，這個不舒服主要在身體哪裡？" to AudioResources(R.raw.triage_body_part_02_zh, R.raw.triage_body_part_02_taigi),
        "請問你主要是哪個部位感到不適？" to AudioResources(R.raw.triage_body_part_03_zh, R.raw.triage_body_part_03_taigi),
        "方便告訴我，症狀集中在哪個位置嗎？" to AudioResources(R.raw.triage_body_part_04_zh, R.raw.triage_body_part_04_taigi),
        "這個症狀主要影響身體的哪個部位呢？" to AudioResources(R.raw.triage_body_part_05_zh, R.raw.triage_body_part_05_taigi),
        "請問這個症狀大約持續多久了？" to AudioResources(R.raw.triage_duration_01_zh, R.raw.triage_duration_01_taigi),
        "想了解一下，這個不舒服大概持續多久了呢？" to AudioResources(R.raw.triage_duration_02_zh, R.raw.triage_duration_02_taigi),
        "這個症狀大約是從什麼時候開始的？" to AudioResources(R.raw.triage_duration_03_zh, R.raw.triage_duration_03_taigi),
        "方便告訴我，這個情況已經持續多久了嗎？" to AudioResources(R.raw.triage_duration_04_zh, R.raw.triage_duration_04_taigi),
        "請問你大概多久以前開始出現這個症狀？" to AudioResources(R.raw.triage_duration_05_zh, R.raw.triage_duration_05_taigi),
        "請問症狀程度是輕微、中等、明顯，還是很痛或影響生活？" to AudioResources(R.raw.triage_severity_01_zh, R.raw.triage_severity_01_taigi),
        "想了解一下，目前不舒服的程度大約是輕微、中等還是嚴重？" to AudioResources(R.raw.triage_severity_02_zh, R.raw.triage_severity_02_taigi),
        "請問這個症狀現在的嚴重程度如何？" to AudioResources(R.raw.triage_severity_03_zh, R.raw.triage_severity_03_taigi),
        "這個不舒服有沒有明顯到影響你的日常生活？" to AudioResources(R.raw.triage_severity_04_zh, R.raw.triage_severity_04_taigi),
        "方便描述一下，症狀目前是輕微、普通，還是相當嚴重嗎？" to AudioResources(R.raw.triage_severity_05_zh, R.raw.triage_severity_05_taigi),
        "請問你最近哪幾天有空就醫？" to AudioResources(R.raw.triage_preferred_days_01_zh, R.raw.triage_preferred_days_01_taigi),
        "想了解一下，你近期哪些日子方便看診？" to AudioResources(R.raw.triage_preferred_days_02_zh, R.raw.triage_preferred_days_02_taigi),
        "請問最近有哪一天比較方便就醫？" to AudioResources(R.raw.triage_preferred_days_03_zh, R.raw.triage_preferred_days_03_taigi),
        "方便告訴我，你接下來哪些日期可以看診嗎？" to AudioResources(R.raw.triage_preferred_days_04_zh, R.raw.triage_preferred_days_04_taigi),
        "你近期較方便安排就醫的日子是哪幾天呢？" to AudioResources(R.raw.triage_preferred_days_05_zh, R.raw.triage_preferred_days_05_taigi),
        "請問你偏好的看診時段是上午、下午還是夜間？" to AudioResources(R.raw.triage_preferred_sessions_01_zh, R.raw.triage_preferred_sessions_01_taigi),
        "想確認一下，你看診比較方便的時段是上午、下午或夜間？" to AudioResources(R.raw.triage_preferred_sessions_02_zh, R.raw.triage_preferred_sessions_02_taigi),
        "請問你偏好安排上午、下午，還是夜間門診？" to AudioResources(R.raw.triage_preferred_sessions_03_zh, R.raw.triage_preferred_sessions_03_taigi),
        "方便告訴我，你比較適合上午、下午或夜間看診嗎？" to AudioResources(R.raw.triage_preferred_sessions_04_zh, R.raw.triage_preferred_sessions_04_taigi),
        "你希望看診時間安排在上午、下午還是夜間呢？" to AudioResources(R.raw.triage_preferred_sessions_05_zh, R.raw.triage_preferred_sessions_05_taigi),
        "想確認一下，您最主要想處理的不舒服症狀是什麼？" to AudioResources(R.raw.triage_clarification_symptom_01_zh, R.raw.triage_clarification_symptom_01_taigi),
        "想再確認一下，你目前有沒有剛才提到的任何一項急迫症狀？如果有，請告訴我是胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛；如果都沒有，請回答『都沒有』。" to AudioResources(R.raw.triage_clarification_red_flags_01_zh, R.raw.triage_clarification_red_flags_01_taigi),
        "想確認一下，症狀主要在身體哪個部位？" to AudioResources(R.raw.triage_clarification_body_part_01_zh, R.raw.triage_clarification_body_part_01_taigi),
        "想確認一下，這個症狀大約持續多久了？例如幾天、幾週或幾個月。" to AudioResources(R.raw.triage_clarification_duration_01_zh, R.raw.triage_clarification_duration_01_taigi),
        "想確認一下，您的疼痛或不舒服是輕微、普通，還是嚴重到影響睡眠或日常活動？" to AudioResources(R.raw.triage_clarification_severity_01_zh, R.raw.triage_clarification_severity_01_taigi),
        "想確認一下，您可就醫的日期是平日、週末、每天都可以，還是目前不確定？" to AudioResources(R.raw.triage_clarification_preferred_days_01_zh, R.raw.triage_clarification_preferred_days_01_taigi),
        "想確認一下，您比較方便的時段是上午、下午、夜間，還是全天都可以？" to AudioResources(R.raw.triage_clarification_preferred_sessions_01_zh, R.raw.triage_clarification_preferred_sessions_01_taigi),
        "目前提供的資訊還不足以確定最合適的科別，我再確認一點：除了目前描述的不舒服之外，還有其他明顯症狀嗎？" to AudioResources(R.raw.triage_clarification_department_01_zh, R.raw.triage_clarification_department_01_taigi),
        "目前仍無法安全地自動判定唯一科別，系統沒有替你套用預設科別。請改用手動選科，或洽醫院掛號服務協助。" to AudioResources(R.raw.triage_system_department_unresolved_01_zh, R.raw.triage_system_department_unresolved_01_taigi),
        "好的，我會延續這次問診紀錄重新整理。請補充想修改的症狀、科別、嚴重程度、看診日期或時段。" to AudioResources(R.raw.triage_system_revision_01_zh, R.raw.triage_system_revision_01_taigi),
        // Backend fixed sentences listed in docs/固定句清單.md; each key must equal the backend text exactly.
        "先回到剛剛的問題：以下狀況如果都沒有，回答「沒有」就可以。" to AudioResources(R.raw.triage_safety_retry_01_zh, R.raw.triage_safety_retry_01_taigi),
        "這題需要您回答「有」或「沒有」，系統才能繼續幫您找科別。" to AudioResources(R.raw.triage_safety_last_attempt_01_zh, R.raw.triage_safety_last_attempt_01_taigi),
        "這題是想確認您有沒有需要馬上去急診的狀況；如果都沒有，回答「沒有」就可以。" to AudioResources(R.raw.triage_safety_explain_01_zh, R.raw.triage_safety_explain_01_taigi),
        "初步篩檢結果顯示您的狀況可能較危急，線上問診無法判斷嚴重程度，因此不會繼續掛號導引。請立即前往急診；情況危急請撥打 119。" to AudioResources(R.raw.triage_stop_urgent_01_zh, R.raw.triage_stop_urgent_01_taigi),
        "您回報的狀況可能需要立即處理，線上問診無法判斷嚴重程度，因此不會繼續掛號導引。請立即前往急診；情況危急請撥打 119。" to AudioResources(R.raw.triage_stop_safety_positive_01_zh, R.raw.triage_stop_safety_positive_01_taigi),
        "因為還沒能確認您是否有急迫症狀，系統不能繼續一般掛號導引。若身體不適，請洽醫療人員協助評估；若情況緊急，請立即尋求緊急醫療協助。" to AudioResources(R.raw.triage_stop_safety_unresolved_01_zh, R.raw.triage_stop_safety_unresolved_01_taigi),
        "目前仍無法可靠釐清症狀，系統不會替你猜測科別；請改用手動選科或洽醫院掛號服務。" to AudioResources(R.raw.triage_stop_unresolved_01_zh, R.raw.triage_stop_unresolved_01_taigi),
        "資訊不足，無法判斷科別。請改用手動選科或洽醫院掛號服務。" to AudioResources(R.raw.triage_stop_no_family_medicine_01_zh, R.raw.triage_stop_no_family_medicine_01_taigi),
        "目前還沒辦法確定要看哪一科，要繼續描述，還是先掛家庭醫學科？" to AudioResources(R.raw.triage_safety_net_01_zh, R.raw.triage_safety_net_01_taigi),
        "還有其他不舒服的地方嗎？沒有的話直接說「沒有」就可以。" to AudioResources(R.raw.triage_companion_02_zh, R.raw.triage_companion_02_taigi),
        "最近還有沒有其他不舒服的地方？" to AudioResources(R.raw.triage_companion_01_zh, R.raw.triage_companion_01_taigi),
        "請問就診者今年幾歲？" to AudioResources(R.raw.triage_age_01_zh, R.raw.triage_age_01_taigi),
        "可以再多說一點嗎？例如哪裡不舒服、從什麼時候開始、有多嚴重。" to AudioResources(R.raw.triage_open_question_01_zh, R.raw.triage_open_question_01_taigi),
        "已記錄目前症狀，但正式科別資料仍不足以確認建議科別。您可以補充新的症狀資訊，或返回就診類型選擇「快速查詢」自行選科。" to AudioResources(R.raw.triage_department_evidence_01_zh, R.raw.triage_department_evidence_01_taigi),
        "最近哪幾天、哪個時段比較方便看診？不方便的時間也可以告訴我；都可以的話直接說『都可以』。" to AudioResources(R.raw.triage_availability_01_zh, R.raw.triage_availability_01_taigi),
        "那一天中，上午、下午還是晚上比較方便？都可以的話直接說『都可以』。" to AudioResources(R.raw.triage_availability_session_01_zh, R.raw.triage_availability_session_01_taigi),
        "那最近哪幾天比較方便？都可以的話直接說『都可以』。" to AudioResources(R.raw.triage_availability_date_01_zh, R.raw.triage_availability_date_01_taigi),
        "目前的症狀無法明確區分科別，系統建議先由家庭醫學科初步評估。請確認後取得推薦掛號方案。" to AudioResources(R.raw.triage_confirm_family_medicine_01_zh, R.raw.triage_confirm_family_medicine_01_taigi),
        "請直接告訴我要改成什麼，例如「改成下週四下午」「改看皮膚科」，或補充新的症狀。" to AudioResources(R.raw.triage_revision_detail_01_zh, R.raw.triage_revision_detail_01_taigi),
        "系統剛剛沒有回應，請再傳一次剛才的訊息。" to AudioResources(R.raw.triage_system_no_response_01_zh, R.raw.triage_system_no_response_01_taigi)
    )

    internal val requiredTexts: Set<String>
        get() = resourcesByText.keys

    fun resolve(messageText: String, language: String): Int? {
        val resources = resourcesByText[messageText.trim()] ?: return null
        return when (language.trim().lowercase()) {
            "chinese", "zh" -> resources.chinese
            "taiwanese", "taigi" -> resources.taiwanese
            else -> null
        }
    }
}
