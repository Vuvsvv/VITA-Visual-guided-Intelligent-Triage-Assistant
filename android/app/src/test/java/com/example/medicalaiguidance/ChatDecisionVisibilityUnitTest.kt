package com.example.medicalaiguidance

import com.example.medicalaiguidance.network.ConversationStateDto
import com.example.medicalaiguidance.network.DepartmentResultDto
import com.example.medicalaiguidance.network.QuestionItemDto
import com.example.medicalaiguidance.network.TriageResultDto
import com.example.medicalaiguidance.network.UrgencyResultDto
import com.example.medicalaiguidance.network.parseVoiceChatResponse
import com.example.medicalaiguidance.viewmodel.isReadyForRecommendation
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatDecisionVisibilityUnitTest {
    @Test
    fun buttonsAppearOnlyAfterSuggestedDepartmentAndNoPendingQuestion() {
        val ready = result()

        assertTrue(ready.isReadyForRecommendation())
        assertFalse(ready.copy(needMoreInfo = true).isReadyForRecommendation())
        assertFalse(ready.copy(nextQuestion = "症狀持續多久？").isReadyForRecommendation())
        assertFalse(
            ready.copy(
                triage = ready.triage.copy(nextQuestion = "症狀持續多久？")
            ).isReadyForRecommendation()
        )
        assertFalse(
            ready.copy(
                questionBatch = listOf(QuestionItemDto(key = "duration", question = "症狀持續多久？"))
            ).isReadyForRecommendation()
        )
        assertFalse(ready.copy(departmentResult = null).isReadyForRecommendation())
        assertFalse(
            ready.copy(
                conversationState = ready.conversationState.copy(stage = "collecting")
            ).isReadyForRecommendation()
        )
    }

    @Test
    fun voiceResponseParsesDepartmentBeforeShowingButtons() {
        val ready = parseVoiceChatResponse(
            """
            {
              "case_id":"case_voice_ready",
              "needMoreInfo":false,
              "stage":"waiting_confirmation",
              "department_result":{
                "parentDept":"內科系",
                "childDept":"一般內科",
                "confidence":0.9,
                "reason":[]
              }
            }
            """.trimIndent()
        )

        assertTrue(ready.isReadyForRecommendation())
        assertFalse(ready.copy(departmentResult = null).isReadyForRecommendation())
    }

    private fun result() = TriageResultDto(
        caseId = "case_ready",
        triageCase = null,
        conversationState = ConversationStateDto(stage = "waiting_confirmation"),
        triage = UrgencyResultDto(needMoreInfo = false, nextQuestion = null, isFinal = true),
        departmentResult = DepartmentResultDto(
            parentDept = "內科系",
            childDept = "一般內科",
            confidence = 0.9
        ),
        nextQuestion = null,
        needMoreInfo = false,
        questionBatch = emptyList()
    )
}
