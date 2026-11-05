package com.example.medicalaiguidance

import com.example.medicalaiguidance.network.parseRecommendationResult
import com.example.medicalaiguidance.network.parseScriptResponse
import com.example.medicalaiguidance.network.parseTriageResult
import org.junit.Test

import org.junit.Assert.*

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    @Test
    fun addition_isCorrect() {
        assertEquals(4, 2 + 2)
    }

    @Test
    fun parseBackendChatContract() {
        val result = parseTriageResult(
            """
            {
              "case_id": "case_1",
              "triage_case": {
                "case_id": "case_1",
                "patient_input": {
                  "symptom": "膝蓋痛",
                  "red_flags": [],
                  "red_flags_checked": true
                },
                "availability": {
                  "preferred_days": ["週一"],
                  "preferred_sessions": ["上午"],
                  "semantic_status": {"preferred_days": "partial"},
                  "confidence": {"preferred_days": 0.88}
                },
                "conversation_state": {
                  "stage": "waiting_confirmation",
                  "is_complete": true,
                  "awaiting_confirmation": true,
                  "confirmed": false,
                  "asked_fields": ["red_flags"],
                  "consumed_fields": ["red_flags", "preferred_days"],
                  "last_question_key": null,
                  "question_attempts": {"red_flags": 1},
                  "field_statuses": {"red_flags": "unavailable"},
                  "field_confidence": {"red_flags": 0.9},
                  "clarification_reasons": {}
                },
                "triage": {
                  "need_more_info": false,
                  "next_question": null
                }
              },
              "conversation_state": {
                "stage": "waiting_confirmation",
                "is_complete": true,
                "awaiting_confirmation": true,
                "confirmed": false,
                "consumed_fields": ["red_flags", "preferred_days"]
              },
              "triage": {
                "need_more_info": false,
                "next_question": null
              },
              "department_result": {
                "parentDept": "外科系",
                "childDept": "一般骨科",
                "confidence": 0.8,
                "reason": ["膝蓋疼痛"]
              },
              "next_question": null,
              "reply": "目前建議科別為 一般骨科，請確認後取得推薦掛號方案。",
              "needMoreInfo": false,
              "triage_reasons": ["問診必要資訊已完整"]
            }
            """.trimIndent()
        )

        assertEquals("case_1", result.caseId)
        assertFalse(result.needMoreInfo)
        assertEquals("waiting_confirmation", result.conversationState.stage)
        assertTrue(result.conversationState.consumedFields.contains("red_flags"))
        assertEquals("一般骨科", result.departmentResult?.childDept)
        assertEquals("partial", result.triageCase?.availability?.semanticStatus?.get("preferred_days"))
    }

    @Test
    fun parseRecommendationAndScriptContracts() {
        val recommendations = parseRecommendationResult(
            """
            {
              "case_id": "case_1",
              "recommendations": {
                "specialty_first": [
                  {
                    "recommendation_id": "rec_1",
                    "parentDept": "外科系",
                    "childDept": "一般骨科",
                    "doctor": "王醫師",
                    "date": "2026-05-26",
                    "session": "上午",
                    "slot": "3209診",
                    "score": 92.5,
                    "reasons": ["科別符合"],
                    "rank": 1,
                    "is_best_match": true
                  }
                ],
                "time_first": []
              },
              "fallback_departments": [],
              "total_count": 1
            }
            """.trimIndent()
        )

        assertEquals(1, recommendations.totalCount)
        assertEquals("rec_1", recommendations.recommendations.specialtyFirst.first().recommendationId)
        assertTrue(recommendations.recommendations.specialtyFirst.first().isBestMatch)

        val script = parseScriptResponse(
            """
            {
              "isSuccess": true,
              "script_id": "vgh_booking_001",
              "recommendation_id": "rec_1",
              "steps": [
                {
                  "action": "tap",
                  "target": "一般骨科",
                  "resource_id": null,
                  "text": "一般骨科",
                  "class_name": null,
                  "description": "選擇科別",
                  "delay_ms": 0,
                  "retry": 1
                }
              ],
              "message": "ok",
              "step_count": 1
            }
            """.trimIndent()
        )

        assertTrue(script.isSuccess)
        assertEquals("rec_1", script.recommendationId)
        assertEquals("一般骨科", script.steps.first().target)
    }
}
