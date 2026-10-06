package com.example.medicalaiguidance.network

import android.util.Log
import com.example.medicalaiguidance.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.net.URLEncoder

class MedicalApiClient(
    private val baseUrl: String = DEFAULT_BASE_URL
) {
    suspend fun chat(request: ChatRequest): TriageResultDto =
        post("/chat", request.toJson(), ::parseTriageResult)

    suspend fun recommend(request: RecommendRequest): RecommendationResultDto =
        post("/recommend", request.toJson(), ::parseRecommendationResult)

    suspend fun followupRecommend(request: FollowupRecommendRequest): FollowupRecommendationResultDto =
        post("/followup/recommend", request.toJson(), ::parseFollowupRecommendationResult)

    suspend fun referenceDepartments(): List<ReferenceDepartmentDto> =
        get("/reference/departments", ::parseReferenceDepartments)

    suspend fun quickSearch(request: QuickSearchRequest): QuickSearchResultDto {
        return get(quickSearchPath(request), ::parseQuickSearchResult)
    }

    suspend fun referenceDoctors(department: String): List<ReferenceDoctorDto> {
        val encodedDepartment = URLEncoder.encode(department, Charsets.UTF_8.name())
        return get("/reference/doctors?department=$encodedDepartment", ::parseReferenceDoctors)
    }

    suspend fun generateScript(request: ScriptRequest): ScriptResponseDto =
        post("/generate_script", request.toJson(), ::parseScriptResponse)

    suspend fun tts(request: TtsRequest): VoiceTtsResponseDto {
        Log.d(TAG, "POST /voice/tts lang=${request.lang}")
        return post("/voice/tts", request.toJson(), ::parseVoiceTtsResponse).also { result ->
            Log.d(
                TAG,
                "/voice/tts result lang=${request.lang} tts_failed=${result.ttsFailed} " +
                    "audio_format=${result.audioFormat} audio_len=${result.audioBase64.length} error=${result.error}"
            )
        }
    }

    suspend fun cleanupTts(sessionId: String) {
        post("/voice/tts/cleanup", JSONObject().put("session_id", sessionId).toString(),
            { body -> check(JSONObject(body).optBoolean("cleared")) }, timeoutMs = 1500)
    }

    suspend fun voiceChat(
        audioBytes: ByteArray,
        caseId: String?,
        lang: String,
        confirmed: Boolean = false,
        visitType: String? = null
    ): VoiceChatResponseDto = withContext(Dispatchers.IO) {
        val endpoint = baseUrl.trimEnd('/') + "/voice/chat"
        val boundary = "----MedicalBoundary${System.currentTimeMillis()}"
        Log.d(TAG, "POST $endpoint (multipart)")

        val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = VOICE_READ_TIMEOUT_MS
            doInput = true
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("Accept", "application/json")
        }

        try {
            connection.outputStream.use { output ->
                val writer = output.bufferedWriter(Charsets.UTF_8)

                fun writeField(name: String, value: String) {
                    writer.write("--$boundary\r\n")
                    writer.write("Content-Disposition: form-data; name=\"$name\"\r\n\r\n")
                    writer.write(value)
                    writer.write("\r\n")
                }

                caseId?.let { writeField("case_id", it) }
                writeField("lang", lang)
                writeField("confirmed", confirmed.toString())
                canonicalVoiceVisitType(visitType)?.let { writeField("visit_type", it) }

                writer.write("--$boundary\r\n")
                writer.write("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n")
                writer.write("Content-Type: audio/wav\r\n\r\n")
                writer.flush()

                output.write(audioBytes)
                output.flush()

                writer.write("\r\n--$boundary--\r\n")
                writer.flush()
            }

            val statusCode = connection.responseCode
            val responseBody = readBody(
                if (statusCode in 200..299) connection.inputStream else connection.errorStream
            )
            Log.d(TAG, "HTTP $statusCode /voice/chat")

            if (statusCode !in 200..299) {
                val detail = extractErrorMessage(responseBody)
                throw MedicalApiException(
                    message = medicalApiErrorMessage(statusCode, detail),
                    statusCode = statusCode,
                    detail = detail
                )
            }

            parseVoiceChatResponse(responseBody)
        } catch (error: MedicalApiException) {
            throw error
        } catch (error: SocketTimeoutException) {
            throw MedicalApiException("語音處理時間較長，請稍後再試。", error)
        } catch (error: IOException) {
            throw MedicalApiException("無法連線語音後端：$baseUrl", error)
        } finally {
            connection.disconnect()
        }
    }

    suspend fun voiceAsr(
        audioBytes: ByteArray,
        lang: String
    ): VoiceAsrResponseDto = withContext(Dispatchers.IO) {
        val endpoint = baseUrl.trimEnd('/') + "/voice/asr"
        val boundary = "----MedicalBoundary${System.currentTimeMillis()}"
        Log.d(TAG, "POST $endpoint (multipart)")

        val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = VOICE_READ_TIMEOUT_MS
            doInput = true
            doOutput = true
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("Accept", "application/json")
        }

        try {
            connection.outputStream.use { output ->
                val writer = output.bufferedWriter(Charsets.UTF_8)

                writer.write("--$boundary\r\n")
                writer.write("Content-Disposition: form-data; name=\"lang\"\r\n\r\n")
                writer.write(lang)
                writer.write("\r\n")

                writer.write("--$boundary\r\n")
                writer.write("Content-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\n")
                writer.write("Content-Type: audio/wav\r\n\r\n")
                writer.flush()

                output.write(audioBytes)
                output.flush()

                writer.write("\r\n--$boundary--\r\n")
                writer.flush()
            }

            val statusCode = connection.responseCode
            val responseBody = readBody(
                if (statusCode in 200..299) connection.inputStream else connection.errorStream
            )
            Log.d(TAG, "HTTP $statusCode /voice/asr")

            if (statusCode !in 200..299) {
                throw MedicalApiException("語音辨識後端錯誤 HTTP $statusCode：${extractErrorMessage(responseBody)}")
            }

            parseVoiceAsrResponse(responseBody)
        } catch (error: MedicalApiException) {
            throw error
        } catch (error: SocketTimeoutException) {
            throw MedicalApiException("語音辨識時間較長，請稍後再試。", error)
        } catch (error: IOException) {
            throw MedicalApiException("無法連線語音後端：$baseUrl", error)
        } finally {
            connection.disconnect()
        }
    }

    private suspend fun <T> post(
        path: String,
        body: String,
        parser: (String) -> T,
        timeoutMs: Int = READ_TIMEOUT_MS
    ): T = withContext(Dispatchers.IO) {
        val endpoint = baseUrl.trimEnd('/') + path
        Log.d(TAG, "POST $endpoint")

        val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = minOf(CONNECT_TIMEOUT_MS, timeoutMs)
            readTimeout = timeoutMs
            doInput = true
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }

        try {
            connection.outputStream.use { output ->
                output.write(body.toByteArray(Charsets.UTF_8))
            }

            val statusCode = connection.responseCode
            val responseBody = readBody(
                if (statusCode in 200..299) connection.inputStream else connection.errorStream
            )
            Log.d(TAG, "HTTP $statusCode $path")

            if (statusCode !in 200..299) {
                val detail = extractErrorMessage(responseBody)
                throw MedicalApiException(
                    message = medicalApiErrorMessage(statusCode, detail),
                    statusCode = statusCode,
                    detail = detail
                )
            }

            parser(responseBody)
        } catch (error: MedicalApiException) {
            Log.e(TAG, "API error on $path: ${error.message}")
            throw error
        } catch (error: SocketTimeoutException) {
            Log.e(TAG, "Request timeout on $path", error)
            throw MedicalApiException("目前無法連線至問診服務，請確認網路連線後稍後再試。", error)
        } catch (error: IOException) {
            Log.e(TAG, "Network error on $path", error)
            throw MedicalApiException("目前無法連線至問診服務，請確認網路連線後稍後再試。", error)
        } catch (error: Exception) {
            Log.e(TAG, "Request error on $path", error)
            throw MedicalApiException("請求後端時發生錯誤：${error.message ?: "未知錯誤"}", error)
        } finally {
            connection.disconnect()
        }
    }

    private suspend fun <T> get(
        path: String,
        parser: (String) -> T
    ): T = withContext(Dispatchers.IO) {
        val endpoint = baseUrl.trimEnd('/') + path
        Log.d(TAG, "GET $endpoint")
        val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doInput = true
            setRequestProperty("Accept", "application/json")
        }

        try {
            val statusCode = connection.responseCode
            val responseBody = readBody(
                if (statusCode in 200..299) connection.inputStream else connection.errorStream
            )
            Log.d(TAG, "HTTP $statusCode ${path.substringBefore('?')}")
            if (statusCode !in 200..299) {
                val detail = extractErrorMessage(responseBody)
                throw MedicalApiException(
                    message = "無法載入正式科別或醫師清單：$detail",
                    statusCode = statusCode,
                    detail = detail
                )
            }
            parser(responseBody)
        } catch (error: MedicalApiException) {
            Log.e(TAG, "API error on ${path.substringBefore('?')}: ${error.message}")
            throw error
        } catch (error: SocketTimeoutException) {
            throw MedicalApiException("載入正式科別或醫師清單逾時，請重新載入。", error)
        } catch (error: IOException) {
            throw MedicalApiException("無法連線後端以載入正式科別或醫師清單。", error)
        } catch (error: Exception) {
            throw MedicalApiException("載入正式科別或醫師清單時發生錯誤。", error)
        } finally {
            connection.disconnect()
        }
    }

    private fun readBody(stream: InputStream?): String {
        if (stream == null) return ""
        return BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
            reader.readText()
        }
    }

    private fun extractErrorMessage(responseBody: String): String {
        if (responseBody.isBlank()) return "沒有錯誤內容"
        return runCatching {
            JSONObject(responseBody).optString("detail", responseBody)
        }.getOrDefault(responseBody)
    }

    companion object {
        private const val TAG = "MedicalApiClient"
        private const val CONNECT_TIMEOUT_MS = 15_000
        private const val READ_TIMEOUT_MS = 60_000
        private const val VOICE_READ_TIMEOUT_MS = 60_000
        val DEFAULT_BASE_URL: String = BuildConfig.API_BASE_URL
    }
}

class MedicalApiException(
    message: String,
    cause: Throwable? = null,
    val statusCode: Int? = null,
    val detail: String? = null
) : Exception(message, cause)

internal fun medicalApiErrorMessage(statusCode: Int, detail: String): String = when (statusCode) {
    409 -> "此問診的就診類型與原案件不一致，請返回首頁重新開始問診。"
    400 -> "問診資料尚未完成或請求內容不正確：$detail"
    503 -> detail.ifBlank { "問診服務目前無法完成查詢，請稍後再試。" }
    else -> "後端回應錯誤 HTTP $statusCode：$detail"
}

internal fun canonicalVoiceVisitType(value: String?): String? =
    value?.takeIf { it == "initial" || it == "followup" || it == "return_visit" }

internal fun quickSearchPath(request: QuickSearchRequest): String {
    val deptId = URLEncoder.encode(request.deptId.toString(), Charsets.UTF_8.name())
    val date = URLEncoder.encode(request.date, Charsets.UTF_8.name())
    val period = URLEncoder.encode(request.period, Charsets.UTF_8.name())
    return "/schedules/search?dept_id=$deptId&date=$date&period=$period"
}
