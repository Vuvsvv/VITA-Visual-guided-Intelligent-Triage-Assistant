package com.example.medicalaiguidance

import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.network.VoiceTtsResponseDto
import com.example.medicalaiguidance.repository.TtsSession
import com.example.medicalaiguidance.util.FixedTriageAudioResolver
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class FixedTriageAudioResolverTest {
    private val requiredPrompts = listOf(
        "triage_symptom_01" to "請描述目前最主要的不舒服症狀。",
        "triage_symptom_02" to "想先了解一下，你現在最主要是哪裡不舒服？",
        "triage_symptom_03" to "請問目前最困擾你的症狀是什麼？",
        "triage_symptom_04" to "方便說明一下，你這次最想處理的不舒服嗎？",
        "triage_symptom_05" to "請告訴我，目前最主要的身體不適是什麼？",
        "triage_emergency_symptoms_01" to "請問是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛等急迫症狀？",
        "triage_emergency_symptoms_02" to "想確認安全狀況：目前是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
        "triage_emergency_symptoms_03" to "請確認一下，你現在有沒有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
        "triage_emergency_symptoms_04" to "為了確認是否需要立即處理，請問有無突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
        "triage_emergency_symptoms_05" to "接著確認急迫症狀：你是否出現突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛？",
        "triage_body_part_01" to "請問症狀主要發生在哪個部位？",
        "triage_body_part_02" to "想了解一下，這個不舒服主要在身體哪裡？",
        "triage_body_part_03" to "請問你主要是哪個部位感到不適？",
        "triage_body_part_04" to "方便告訴我，症狀集中在哪個位置嗎？",
        "triage_body_part_05" to "這個症狀主要影響身體的哪個部位呢？",
        "triage_duration_01" to "請問這個症狀大約持續多久了？",
        "triage_duration_02" to "想了解一下，這個不舒服大概持續多久了呢？",
        "triage_duration_03" to "這個症狀大約是從什麼時候開始的？",
        "triage_duration_04" to "方便告訴我，這個情況已經持續多久了嗎？",
        "triage_duration_05" to "請問你大概多久以前開始出現這個症狀？",
        "triage_severity_01" to "請問症狀程度是輕微、中等、明顯，還是很痛或影響生活？",
        "triage_severity_02" to "想了解一下，目前不舒服的程度大約是輕微、中等還是嚴重？",
        "triage_severity_03" to "請問這個症狀現在的嚴重程度如何？",
        "triage_severity_04" to "這個不舒服有沒有明顯到影響你的日常生活？",
        "triage_severity_05" to "方便描述一下，症狀目前是輕微、普通，還是相當嚴重嗎？",
        "triage_preferred_days_01" to "請問你最近哪幾天有空就醫？",
        "triage_preferred_days_02" to "想了解一下，你近期哪些日子方便看診？",
        "triage_preferred_days_03" to "請問最近有哪一天比較方便就醫？",
        "triage_preferred_days_04" to "方便告訴我，你接下來哪些日期可以看診嗎？",
        "triage_preferred_days_05" to "你近期較方便安排就醫的日子是哪幾天呢？",
        "triage_preferred_sessions_01" to "請問你偏好的看診時段是上午、下午還是夜間？",
        "triage_preferred_sessions_02" to "想確認一下，你看診比較方便的時段是上午、下午或夜間？",
        "triage_preferred_sessions_03" to "請問你偏好安排上午、下午，還是夜間門診？",
        "triage_preferred_sessions_04" to "方便告訴我，你比較適合上午、下午或夜間看診嗎？",
        "triage_preferred_sessions_05" to "你希望看診時間安排在上午、下午還是夜間呢？",
        "triage_clarification_symptom_01" to "想確認一下，您最主要想處理的不舒服症狀是什麼？",
        "triage_clarification_red_flags_01" to "想再確認一下，你目前有沒有剛才提到的任何一項急迫症狀？如果有，請告訴我是胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛；如果都沒有，請回答『都沒有』。",
        "triage_clarification_body_part_01" to "想確認一下，症狀主要在身體哪個部位？",
        "triage_clarification_duration_01" to "想確認一下，這個症狀大約持續多久了？例如幾天、幾週或幾個月。",
        "triage_clarification_severity_01" to "想確認一下，您的疼痛或不舒服是輕微、普通，還是嚴重到影響睡眠或日常活動？",
        "triage_clarification_preferred_days_01" to "想確認一下，您可就醫的日期是平日、週末、每天都可以，還是目前不確定？",
        "triage_clarification_preferred_sessions_01" to "想確認一下，您比較方便的時段是上午、下午、夜間，還是全天都可以？",
        "triage_clarification_department_01" to "目前提供的資訊還不足以確定最合適的科別，我再確認一點：除了目前描述的不舒服之外，還有其他明顯症狀嗎？",
        "triage_system_department_unresolved_01" to "目前仍無法安全地自動判定唯一科別，系統沒有替你套用預設科別。請改用手動選科，或洽醫院掛號服務協助。",
        "triage_system_revision_01" to "好的，我會延續這次問診紀錄重新整理。請補充想修改的症狀、嚴重程度、看診日期或時段。"
    )

    @Test
    fun allRequiredPromptsResolveToTheirLanguageSpecificRawResources() {
        assertEquals(45, requiredPrompts.size)
        assertEquals(45, requiredPrompts.map { it.second }.toSet().size)
        assertEquals(requiredPrompts.map { it.second }.toSet(), FixedTriageAudioResolver.requiredTexts)

        requiredPrompts.forEach { (id, text) ->
            val zh = rawResource("${id}_zh")
            val taigi = rawResource("${id}_taigi")
            assertEquals(zh, FixedTriageAudioResolver.resolve(text, "chinese"))
            assertEquals(zh, FixedTriageAudioResolver.resolve("  $text  ", "zh"))
            assertEquals(taigi, FixedTriageAudioResolver.resolve(text, "taiwanese"))
            assertEquals(taigi, FixedTriageAudioResolver.resolve(text, "taigi"))
            assertNotEquals(zh, taigi)
        }
    }

    @Test
    fun dynamicUnknownAndPunctuationChangedTextUseNetworkFallback() {
        assertNull(
            FixedTriageAudioResolver.resolve(
                "目前建議科別為 一般骨科，請確認後取得推薦掛號方案。",
                "chinese"
            )
        )
        assertNull(FixedTriageAudioResolver.resolve("這是一句未知文字。", "taiwanese"))
        assertNull(FixedTriageAudioResolver.resolve("請描述目前最主要的不舒服症狀？", "chinese"))
        assertNull(FixedTriageAudioResolver.resolve(requiredPrompts.first().second, "unsupported"))
    }

    @Test
    fun fixedPromptSkipsNetworkPrefetchWhileDynamicPromptStillUsesIt() = runTest {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        val calls = mutableListOf<Pair<String, String>>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, text, lang ->
            calls += text to lang
            VoiceTtsResponseDto(audioBase64 = "audio")
        }
        val model = ChatViewModel(ttsSessionFactory = { session })
        try {
            model.startNewConversation()
            model.prepareSpeech(
                ChatMessage(content = requiredPrompts.first().second, sender = MessageSender.AI),
                "chinese"
            )
            runCurrent()
            assertEquals(emptyList<Pair<String, String>>(), calls)

            val dynamicText = "目前建議科別為 一般骨科，請確認後取得推薦掛號方案。"
            model.prepareSpeech(ChatMessage(content = dynamicText, sender = MessageSender.AI), "chinese")
            runCurrent()
            assertEquals(listOf(dynamicText to "chinese"), calls)
        } finally {
            session.close()
            model.viewModelScope.cancel()
            Dispatchers.resetMain()
        }
    }

    private fun rawResource(name: String): Int =
        R.raw::class.java.getField(name).getInt(null)
}
