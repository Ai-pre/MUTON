package com.example.myapplication

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

object OpenAiSummaryService {

    data class SummaryResult(
        val text: String? = null,
        val errorMessage: String? = null,
    )

    private val client = OkHttpClient()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    fun summarizeConversation(
        conversationText: String,
        onResult: (String?) -> Unit,
    ) {
        summarizeConversationDetailed(conversationText) { result ->
            onResult(result.text)
        }
    }

    fun summarizeConversationDetailed(
        conversationText: String,
        onResult: (SummaryResult) -> Unit,
    ) {
        val apiKey = BuildConfig.OPENAI_API_KEY.trim()
        if (apiKey.isBlank()) {
            onResult(SummaryResult(errorMessage = "OpenAI API 키가 설정되어 있지 않아요."))
            return
        }

        if (conversationText.isBlank()) {
            onResult(SummaryResult(errorMessage = "선택한 기록에 다시 요약할 대화 내용이 부족해요."))
            return
        }

        val input = JSONArray().apply {
            put(
                JSONObject().apply {
                    put("role", "user")
                    put(
                        "content",
                        JSONArray().put(
                            JSONObject().apply {
                                put("type", "input_text")
                                put("text", conversationText)
                            },
                        ),
                    )
                },
            )
        }

        val payload = JSONObject().apply {
            put("model", BuildConfig.OPENAI_SUMMARY_MODEL)
            put(
                "instructions",
                "You summarize conversations in Korean. Return exactly one concise Korean sentence that captures the overall conversation topic and intent. Do not add labels, quotes, bullets, or explanations.",
            )
            put("input", input)
            put("max_output_tokens", 60)
            put("truncation", "auto")
        }

        val request = Request.Builder()
            .url("https://api.openai.com/v1/responses")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(payload.toString().toRequestBody(jsonMediaType))
            .build()

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                onResult(
                    SummaryResult(
                        errorMessage = "OpenAI 요청에 실패했어요. 네트워크 상태를 확인해주세요.",
                    ),
                )
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                response.use {
                    val body = it.body?.string().orEmpty()

                    if (!it.isSuccessful) {
                        onResult(SummaryResult(errorMessage = parseErrorMessage(it.code, body)))
                        return
                    }

                    val text = parseOutputText(body)?.trim()?.takeIf(String::isNotBlank)
                    if (text.isNullOrBlank()) {
                        onResult(
                            SummaryResult(
                                errorMessage = "요약 결과가 비어 있어요. 잠시 후 다시 시도해주세요.",
                            ),
                        )
                    } else {
                        onResult(SummaryResult(text = text))
                    }
                }
            }
        })
    }

    private fun parseOutputText(body: String): String? {
        return runCatching {
            val json = JSONObject(body)

            json.optJSONArray("output")
                ?.let { output ->
                    for (i in 0 until output.length()) {
                        val item = output.optJSONObject(i) ?: continue
                        val content = item.optJSONArray("content") ?: continue
                        for (j in 0 until content.length()) {
                            val contentItem = content.optJSONObject(j) ?: continue
                            val text = contentItem.optString("text")
                            if (text.isNotBlank()) return@runCatching text
                        }
                    }
                    null
                }
                ?: json.optString("output_text").ifBlank { null }
        }.getOrNull()
    }

    private fun parseErrorMessage(code: Int, body: String): String {
        val apiMessage = runCatching {
            val json = JSONObject(body)
            json.optJSONObject("error")?.optString("message").orEmpty()
        }.getOrNull().orEmpty()

        val mappedMessage = when (code) {
            401 -> "OpenAI API 키가 올바르지 않거나 만료되었어요."
            402 -> "OpenAI 결제 또는 크레딧 상태를 확인해주세요."
            403 -> "현재 API 키로는 이 모델을 사용할 수 없어요."
            429 -> "요청 한도에 걸렸어요. 잠시 후 다시 시도해주세요."
            else -> "OpenAI 요약 요청에 실패했어요. (HTTP $code)"
        }

        return if (apiMessage.isBlank()) {
            mappedMessage
        } else {
            "$mappedMessage\n$apiMessage"
        }
    }
}
