package com.example.myapplication

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.myapplication.databinding.ActivityMainBinding
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class MainActivity : BaseActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null

    private val client = OkHttpClient.Builder().build()

    companion object {
        private const val PERMISSION_REQUEST_CODE = 10
        private const val REMOTE_CONFIG_URL =
            "https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/server/backend_url.json"
        private const val TAG = "BodyCamApp"
        private const val ENDPOINT_VIDEO = "/process_video_chunk"
        private const val ENDPOINT_FAST_AUDIO = "/process_audio_chunk"
        private const val ENDPOINT_SLOW_ANALYSIS = "/get_fusion_analysis"
    }

    @Volatile
    private var serverBaseUrl: String? = null
    @Volatile
    private var isConfigLoading = false

    private val mainHandler = Handler(Looper.getMainLooper())
    private val configRetryDelayMs = 3000L

    private var isVideoStreaming = false
    private var isAudioStreaming = false
    private var audioRecord: AudioRecord? = null
    private val sampleRate = 16000
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

    private var isFrameSending = false
    private var lastSentTime = 0L
    private var sessionStartedAt = 0L
    private var latestSpeechText = ""
    private var latestSummaryText = ""
    private var latestSpeaker = "self"
    private val conversationItems = mutableListOf<ConversationTurn>()

    private data class ConversationTurn(
        val speaker: String,
        var speech: String,
        var summary: String = "",
    )

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.backButton.setOnClickListener { finish() }
        binding.txtFaceResult.setText(R.string.live_status_ready)
        binding.txtPlaceholder.visibility = View.GONE

        if (!hasPermissions()) {
            requestPermissions()
        } else {
            loadServerBaseUrlWithRetry { beginStreaming() }
        }

        binding.btnStart.setOnClickListener {
            if (!isVideoStreaming && !isAudioStreaming) {
                if (serverBaseUrl.isNullOrBlank()) {
                    loadServerBaseUrlWithRetry { beginStreaming() }
                } else {
                    beginStreaming()
                }
            }
        }

        binding.btnStop.setOnClickListener {
            stopAndArchiveSession()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAllStreaming()
        mainHandler.removeCallbacksAndMessages(null)
        cameraExecutor.shutdown()
    }

    private fun beginStreaming() {
        sessionStartedAt = System.currentTimeMillis()
        latestSpeechText = ""
        latestSummaryText = ""
        latestSpeaker = "self"
        conversationItems.clear()
        binding.conversationList.removeAllViews()
        binding.txtFaceResult.setText(R.string.live_status_listening)
        binding.txtPlaceholder.visibility = View.GONE
        isVideoStreaming = true
        startCamera()
        startAudioStreaming()
    }

    private fun renderStoppedState() {
        binding.txtFaceResult.setText(R.string.live_status_saved)
        binding.txtPlaceholder.visibility = View.GONE
    }

    private fun loadServerBaseUrl(onReady: (() -> Unit)? = null) {
        if (isConfigLoading) return
        if (!serverBaseUrl.isNullOrBlank()) {
            onReady?.invoke()
            return
        }

        isConfigLoading = true
        val requestUrl = "${REMOTE_CONFIG_URL}?t=${System.currentTimeMillis()}"
        Log.d(TAG, "Fetching config: $requestUrl")

        val request = Request.Builder()
            .url(requestUrl)
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                isConfigLoading = false
                Log.e(TAG, "Config load fail: ${e.message}", e)
                runOnUiThread {
                    binding.btnStart.isEnabled = false
                    binding.txtFaceResult.setText(R.string.live_status_offline)
                    binding.txtPlaceholder.visibility = View.VISIBLE
                    binding.txtPlaceholder.setText(R.string.server_load_failed)
                }
            }

            override fun onResponse(call: Call, response: Response) {
                isConfigLoading = false
                response.use {
                    if (!it.isSuccessful) {
                        runOnUiThread {
                            binding.btnStart.isEnabled = false
                            binding.txtFaceResult.setText(R.string.live_status_offline)
                            binding.txtPlaceholder.visibility = View.VISIBLE
                            binding.txtPlaceholder.text =
                                getString(R.string.server_response_error, it.code)
                        }
                        return
                    }

                    val body = it.body?.string().orEmpty()
                    Log.d(TAG, "Config raw body = $body")

                    try {
                        val url = JSONObject(body).optString("base_url").trim().removeSuffix("/")

                        if (url.isBlank()) {
                            runOnUiThread {
                                binding.btnStart.isEnabled = false
                                binding.txtFaceResult.setText(R.string.live_status_offline)
                                binding.txtPlaceholder.visibility = View.VISIBLE
                                binding.txtPlaceholder.setText(R.string.server_empty)
                            }
                            return
                        }

                        serverBaseUrl = url
                        runOnUiThread {
                            binding.btnStart.isEnabled = true
                            if (onReady == null) {
                                binding.txtFaceResult.setText(R.string.live_status_ready)
                                binding.txtPlaceholder.visibility = View.GONE
                            }
                            onReady?.invoke()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Config parse fail: ${e.message}", e)
                        runOnUiThread {
                            binding.btnStart.isEnabled = false
                            binding.txtFaceResult.setText(R.string.live_status_offline)
                            binding.txtPlaceholder.visibility = View.VISIBLE
                            binding.txtPlaceholder.setText(R.string.server_parse_failed)
                        }
                    }
                }
            }
        })
    }

    private fun loadServerBaseUrlWithRetry(onReady: (() -> Unit)? = null) {
        binding.txtFaceResult.setText(R.string.live_status_ready)
        loadServerBaseUrl(onReady)

        if (!serverBaseUrl.isNullOrBlank()) return

        mainHandler.removeCallbacksAndMessages(null)
        mainHandler.postDelayed(
            object : Runnable {
                override fun run() {
                    if (!serverBaseUrl.isNullOrBlank() || isDestroyed || isFinishing) return

                    binding.txtPlaceholder.visibility = View.VISIBLE
                    binding.txtPlaceholder.text = getString(R.string.server_retrying)
                    loadServerBaseUrl(onReady)

                    if (serverBaseUrl.isNullOrBlank()) {
                        mainHandler.postDelayed(this, configRetryDelayMs)
                    }
                }
            },
            configRetryDelayMs,
        )
    }

    private fun buildUrl(path: String): String? {
        val baseUrl = serverBaseUrl
        if (baseUrl.isNullOrBlank()) {
            runOnUiThread {
                binding.txtPlaceholder.visibility = View.VISIBLE
                binding.txtPlaceholder.setText(R.string.server_not_loaded)
            }
            return null
        }
        return "$baseUrl$path"
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
            {
                try {
                    cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also { previewUseCase ->
                        previewUseCase.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                    }

                    imageAnalyzer = ImageAnalysis.Builder()
                        .setTargetResolution(Size(320, 240))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalyzer?.setAnalyzer(cameraExecutor) { image ->
                        if (!isVideoStreaming) {
                            image.close()
                            return@setAnalyzer
                        }

                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastSentTime < 400 || isFrameSending) {
                            image.close()
                            return@setAnalyzer
                        }

                        lastSentTime = currentTime
                        isFrameSending = true
                        try {
                            val bytes = imageToJpegBytes(image)
                            sendVideoFrame(bytes)
                        } catch (_: Exception) {
                            isFrameSending = false
                        } finally {
                            image.close()
                        }
                    }

                    cameraProvider?.unbindAll()
                    cameraProvider?.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalyzer,
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Camera fail: ${e.message}")
                    isVideoStreaming = false
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    private fun imageToJpegBytes(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + (ySize / 2))
        yBuffer.get(nv21, 0, ySize)

        val pixelStride = image.planes[1].pixelStride
        val rowStride = image.planes[1].rowStride
        val uBytes = ByteArray(uSize)
        val vBytes = ByteArray(vSize)
        uBuffer.get(uBytes)
        vBuffer.get(vBytes)

        var uvIndex = ySize
        val width = image.width
        val height = image.height
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val bufferIndex = (row * rowStride) + (col * pixelStride)
                if (bufferIndex < vSize) {
                    nv21[uvIndex++] = vBytes[bufferIndex]
                    nv21[uvIndex++] = uBytes[bufferIndex]
                }
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 30, out)
        return out.toByteArray()
    }

    private fun sendVideoFrame(bytes: ByteArray) {
        if (!isVideoStreaming) {
            isFrameSending = false
            return
        }

        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "frame",
                "frame.jpg",
                bytes.toRequestBody("application/octet-stream".toMediaTypeOrNull()),
            )
            .build()
        val url = buildUrl(ENDPOINT_VIDEO) ?: run {
            isFrameSending = false
            return
        }
        val request = Request.Builder().url(url).post(body).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                isFrameSending = false
            }

            override fun onResponse(call: Call, response: Response) {
                isFrameSending = false
                response.use {
                    if (!it.isSuccessful) return
                    try {
                        val json = JSONObject(it.body?.string() ?: "")
                        val status = json.optString("status")
                        runOnUiThread {
                            if (status == "ok" && isVideoStreaming) {
                                val emotion = json.optString("emotion", "Unknown")
                                binding.txtFaceResult.text =
                                    getString(R.string.face_visual_result, emotion)
                                binding.txtFaceResult.setTextColor(
                                    ContextCompat.getColor(
                                        this@MainActivity,
                                        if (emotion == "Angry" || emotion == "Surprise") {
                                            android.R.color.holo_red_light
                                        } else {
                                            R.color.muton_primary
                                        },
                                    ),
                                )
                            }
                        }
                    } catch (_: Exception) {
                    }
                }
            }
        })
    }

    @SuppressLint("MissingPermission")
    private fun startAudioStreaming() {
        if (isAudioStreaming) return
        isAudioStreaming = true

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize,
            )
            audioRecord?.startRecording()
        } catch (_: Exception) {
            isAudioStreaming = false
            return
        }

        thread {
            val chunkSize = 6400
            val pcmChunk = ByteArray(chunkSize)

            while (isAudioStreaming) {
                if (audioRecord == null || audioRecord?.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                    break
                }

                val readSize = audioRecord?.read(pcmChunk, 0, chunkSize) ?: 0
                if (readSize > 0 && isAudioStreaming) {
                    val chunk = if (readSize == chunkSize) pcmChunk else pcmChunk.copyOf(readSize)
                    sendAudioChunk(chunk)
                }
            }
        }
    }

    private fun stopAudioStreaming() {
        isAudioStreaming = false
        try {
            if (audioRecord?.state == AudioRecord.STATE_INITIALIZED) {
                audioRecord?.stop()
            }
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Audio stop fail: ${e.message}")
        } finally {
            audioRecord = null
        }
    }

    private fun sendAudioChunk(pcmBytes: ByteArray) {
        if (!isAudioStreaming) return

        val reqBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "audio",
                "chunk.pcm",
                pcmBytes.toRequestBody("application/octet-stream".toMediaTypeOrNull()),
            )
            .build()
        val url = buildUrl(ENDPOINT_FAST_AUDIO) ?: return
        val request = Request.Builder().url(url).post(reqBody).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {}

            override fun onResponse(call: Call, response: Response) {
                if (!isAudioStreaming) {
                    response.close()
                    return
                }

                val body = response.body?.string()
                response.close()
                if (body != null) {
                    try {
                        val json = JSONObject(body)
                        val text = json.optString("text")
                        val speaker = json.optString("speaker")

                        if (text.isNotBlank()) {
                            runOnUiThread {
                                latestSpeechText = text
                                latestSpeaker = speaker.ifBlank { latestSpeaker }
                                renderSpeechBubble(text, latestSpeaker)
                            }

                            val prosody = json.optString("prosody")
                            val content = json.optString("content")

                            if (prosody.isNotEmpty() && content.isNotEmpty()) {
                                requestFusionAnalysis(text, prosody, content, speaker)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Audio parse fail: ${e.message}")
                    }
                }
            }
        })
    }

    private fun requestFusionAnalysis(
        text: String,
        prosody: String,
        content: String,
        speaker: String,
    ) {
        if (!isAudioStreaming) return

        val reqBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("text", text)
            .addFormDataPart("prosody", prosody)
            .addFormDataPart("content", content)
            .addFormDataPart("speaker", speaker)
            .build()

        val url = buildUrl(ENDPOINT_SLOW_ANALYSIS) ?: return
        val request = Request.Builder().url(url).post(reqBody).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "Fusion Analysis Fail: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                if (!isAudioStreaming) {
                    response.close()
                    return
                }

                val body = response.body?.string()
                response.close()

                if (body != null) {
                    try {
                        val json = JSONObject(body)
                        val fusionEmotion = json.optString("fusion_emotion", "Waiting")
                        val fusionConfidence = json.optDouble("fusion_confidence", 0.0)
                        val summary = json.optString("summary")

                        runOnUiThread {
                            val nextSummary =
                                when {
                                    summary.isNotEmpty() -> summary
                                    fusionEmotion == "No Visual Input" -> getString(R.string.fusion_no_visual)
                                    fusionEmotion == "Low Confidence" -> getString(R.string.fusion_low_confidence)
                                    fusionEmotion == "Analyzing..." || fusionEmotion == "Waiting" -> getString(
                                        R.string.fusion_collecting,
                                    )

                                    fusionEmotion == "Model Error" || fusionEmotion == "Inference Fail" -> getString(
                                        R.string.fusion_failed,
                                    )

                                    else -> {
                                        val confPercent = (fusionConfidence * 100).toInt()
                                        getString(R.string.fusion_detected, fusionEmotion, confPercent)
                                    }
                            }
                            latestSummaryText = nextSummary
                            renderSummaryBubble(nextSummary, speaker)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Fusion parse fail: ${e.message}")
                    }
                }
            }
        })
    }

    private fun stopCamera() {
        isVideoStreaming = false
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
    }

    private fun stopAllStreaming() {
        stopCamera()
        stopAudioStreaming()
    }

    private fun stopAndArchiveSession() {
        val speech = latestSpeechText.trim()
        val summary = latestSummaryText.trim()
        val hasContent = speech.isNotBlank() || summary.isNotBlank()

        stopAllStreaming()

        ConversationRecordStore.saveTodayRecord(
            context = this,
            title = if (hasContent) summary.ifBlank { getString(R.string.calendar_fallback_title) } else getString(R.string.calendar_fallback_title),
            subtitle = if (hasContent) speech else "",
            startedAt = if (sessionStartedAt == 0L) System.currentTimeMillis() else sessionStartedAt,
            selfSpeech = conversationItems
                .filter { isSelfSpeaker(it.speaker) }
                .joinToString("\n") { it.speech },
            selfSummary = conversationItems
                .filter { isSelfSpeaker(it.speaker) }
                .mapNotNull { it.summary.takeIf(String::isNotBlank) }
                .joinToString("\n"),
            otherSpeech = conversationItems
                .filterNot { isSelfSpeaker(it.speaker) }
                .joinToString("\n") { it.speech },
            otherSummary = conversationItems
                .filterNot { isSelfSpeaker(it.speaker) }
                .mapNotNull { it.summary.takeIf(String::isNotBlank) }
                .joinToString("\n"),
        )

        renderStoppedState()
        startActivity(Intent(this, CalendarActivity::class.java))
        finish()
    }

    private fun renderSpeechBubble(text: String, speaker: String) {
        val normalizedText = text.trim()
        val currentTurn = conversationItems.lastOrNull()
        val shouldAppendNewTurn =
            currentTurn == null ||
                currentTurn.speaker != speaker ||
                currentTurn.speech.trim() != normalizedText

        if (shouldAppendNewTurn) {
            conversationItems.add(
                ConversationTurn(
                    speaker = speaker,
                    speech = normalizedText,
                ),
            )
        } else {
            return
        }

        renderConversationList()
        scrollConversationToBottom()
    }

    private fun renderSummaryBubble(summary: String, speaker: String) {
        val currentTurn = conversationItems.lastOrNull { it.speaker == speaker && it.summary.isBlank() }
            ?: conversationItems.lastOrNull { it.speaker == speaker }

        if (currentTurn == null) {
            conversationItems.add(
                ConversationTurn(
                    speaker = speaker,
                    speech = "",
                    summary = summary,
                ),
            )
        } else {
            currentTurn.summary = summary
        }

        renderConversationList()
        scrollConversationToBottom()
    }

    private fun scrollConversationToBottom() {
        binding.conversationScroll.post {
            binding.conversationScroll.smoothScrollTo(0, binding.conversationContent.bottom)
        }
    }

    private fun isSelfSpeaker(speaker: String): Boolean {
        val normalized = speaker.lowercase()
        return normalized.contains("self") ||
            normalized.contains("user") ||
            normalized.contains("me") ||
            normalized.contains("speaker_0") ||
            normalized.contains("local")
    }

    private fun renderConversationList() {
        binding.conversationList.removeAllViews()
        conversationItems.forEach { item ->
            val bubble = layoutInflater.inflate(
                R.layout.item_conversation_bubble,
                binding.conversationList,
                false,
            ) as LinearLayout

            bubble.background = ContextCompat.getDrawable(
                this,
                if (isSelfSpeaker(item.speaker)) {
                    R.drawable.bg_camera_bubble_light
                } else {
                    R.drawable.bg_camera_bubble_dark
                },
            )

            val speechView = bubble.findViewById<TextView>(R.id.txtBubbleSpeech)
            val summaryView = bubble.findViewById<TextView>(R.id.txtBubbleSummary)

            speechView.text = item.speech.trim()
            summaryView.text = item.summary.trim()
            summaryView.visibility = if (item.summary.isBlank()) View.GONE else View.VISIBLE

            binding.conversationList.addView(bubble)
        }
    }

    private fun hasPermissions(): Boolean {
        return arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
        ).all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO),
            PERMISSION_REQUEST_CODE,
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                loadServerBaseUrlWithRetry { beginStreaming() }
            } else {
                Toast.makeText(this, getString(R.string.permission_required), Toast.LENGTH_LONG)
                    .show()
            }
        }
    }
}
