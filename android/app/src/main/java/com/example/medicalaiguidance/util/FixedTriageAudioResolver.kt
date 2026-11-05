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
        "好的，我會延續這次問診紀錄重新整理。請補充想修改的症狀、嚴重程度、看診日期或時段。" to AudioResources(R.raw.triage_system_revision_01_zh, R.raw.triage_system_revision_01_taigi)
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
