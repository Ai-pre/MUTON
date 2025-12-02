package com.example.myapplication
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.*
import android.os.*
import androidx.appcompat.app.AppCompatActivity
import android.util.Size
import android.widget.Button
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.*
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnStop: Button
    private lateinit var txtResult: TextView

    private lateinit var cameraExecutor: ExecutorService

    // ---- Audio ----
    private var isAudioStreaming = false
    private var audioRecord: AudioRecord? = null
    private var lastFrameSentTime = 0L
    private val sampleRate = 16000
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val bufferSize =
        AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    private val audioIntervalMs = 500L // 0.5초마다 전송

    private val client = OkHttpClient()

    companion object {
        private const val PERMISSION_REQUEST_CODE = 10
        private const val SERVER_BASE_URL = "http://YOUR_SERVER_IP:5000" // ← 수정
    }

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.acti2)

        previewView = findViewById(R.id.cameraPreview)
        btnStop = findViewById(R.id.btnStop)
        txtResult = findViewById(R.id.txtResult)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // 권한 먼저 체크
        if (!hasPermissions()) {
            requestPermissions()
            return  // 🔥 반드시 필요함!
        }

        // 권한 허용된 상태면 바로 실행
        startCamera()
        startAudioStreaming()
    }


    private fun hasPermissions(): Boolean {
        val needed = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.POST_NOTIFICATIONS,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
        )

        return needed.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }



    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.POST_NOTIFICATIONS,
                Manifest.permission.MODIFY_AUDIO_SETTINGS
            ),
            PERMISSION_REQUEST_CODE
        )
    }


    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSION_REQUEST_CODE) {

            if (grantResults.isEmpty() || grantResults.any { it != PackageManager.PERMISSION_GRANTED }) {
                txtResult.text = "권한이 없어서 실행할 수 없습니다."
                return
            }

            startCamera()
            startAudioStreaming()
        }
    }



    // =======================
    // 🔥 CameraX 시작 (후면 카메라 고정)
    // =======================
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(cameraExecutor) { image ->
                val now = System.currentTimeMillis()

                // 0.5초(500ms) 지난 경우에만 전송
                if (now - lastFrameSentTime >= 500L) {
                    val jpegBytes = imageToBytes(image)
                    if (jpegBytes != null) {
                        sendVideoFrame(jpegBytes)
                        lastFrameSentTime = now
                    }
                }

                // 무조건 호출해서 다음 프레임이 들어오게 해야 함
                image.close()
            }


            // ★ 후면 카메라 고정
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // 간단히 plane[0]만 보내는 버전 (Y 채널, 서버에서 처리 방식에 맞춰 나중에 손봐도 됨)
    private fun imageToBytes(image: ImageProxy): ByteArray? {
        return try {
            if (image.format == ImageFormat.YUV_420_888) {
                yuv420ToJpeg(image, 80)   // quality 80 정도로 압축
            } else {
                // 혹시 다른 포맷이면, 일단 plane[0]만 보내는 fallback
                val buffer: ByteBuffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                bytes
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // YUV_420_888 → NV21(ByteArray) 변환
    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // NV21: Y + VU
        val nv21 = ByteArray(ySize + uSize + vSize)

        // Y 채널
        yBuffer.get(nv21, 0, ySize)

        // V + U 채널 (단순히 뒤에 붙이는 방식 - 대부분 기기에서 동작)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        return nv21
    }

    // NV21 → JPEG ByteArray
    private fun yuv420ToJpeg(image: ImageProxy, quality: Int = 80): ByteArray {
        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            quality,
            out
        )
        return out.toByteArray()
    }


    // =======================
    // 서버로 영상 chunk 전송
    // =======================
    private fun sendVideoFrame(bytes: ByteArray) {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "frame",
                "frame.jpg",
                RequestBody.create("image/jpeg".toMediaTypeOrNull(), bytes)
            )
            .build()

        val req = Request.Builder()
            .url("$SERVER_BASE_URL/process_video_chunk")  // 서버 쪽에서 upload_frame alias로 처리 중
            .post(body)
            .build()

        client.newCall(req).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                // 필요하면 응답 사용
                response.close()
            }

            override fun onFailure(call: Call, e: IOException) {
                // 무시하거나 로그만
            }
        })
    }


    // =======================
    // 🔥 AudioRecord streaming
    // =======================
    @SuppressLint("MissingPermission")
    private fun startAudioStreaming() {
        if (isAudioStreaming) return
        if (!hasPermissions()) return

        isAudioStreaming = true

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        audioRecord?.startRecording()

        thread {
            val chunkSize = sampleRate / 2 // 0.5초
            val pcmChunk = ByteArray(chunkSize * 2)

            while (isAudioStreaming) {
                val read = audioRecord?.read(pcmChunk, 0, pcmChunk.size) ?: 0
                if (read > 0) {
                    val wavFile = createWavFile(pcmChunk)
                    sendAudioChunk(wavFile)
                }
                Thread.sleep(audioIntervalMs)
            }
        }
    }

    private fun stopAudioStreaming() {
        isAudioStreaming = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    private fun createWavFile(pcm: ByteArray): File {
        val file = File(cacheDir, "chunk.wav")
        val out = FileOutputStream(file)

        val totalAudioLen = pcm.size
        val totalDataLen = totalAudioLen + 36
        val channels = 1
        val byteRate = sampleRate * 16 * channels / 8

        val header = ByteArray(44)

        // RIFF/WAVE header
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (channels * 16 / 8).toByte()
        header[33] = 0
        header[34] = 16
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        out.write(header)
        out.write(pcm)
        out.close()

        return file
    }

    private fun sendAudioChunk(wavFile: File) {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "audio",
                wavFile.name,
                RequestBody.create("audio/wav".toMediaTypeOrNull(), wavFile)
            )
            .build()

        val req = Request.Builder()
            .url("$SERVER_BASE_URL/process_audio_chunk")
            .post(body)
            .build()

        client.newCall(req).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                val res = response.body?.string()
                runOnUiThread {
                    txtResult.text = res ?: "응답 없음"
                }
            }

            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    txtResult.text = "오디오 전송 실패"
                }
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAudioStreaming()
        cameraExecutor.shutdown()
    }
}
