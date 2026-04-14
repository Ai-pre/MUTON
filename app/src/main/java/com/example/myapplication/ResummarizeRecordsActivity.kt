package com.example.myapplication

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.CheckBox
import android.widget.TextView
import android.widget.Toast
import com.example.myapplication.databinding.ActivityResummarizeRecordsBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ResummarizeRecordsActivity : BaseActivity() {

    private lateinit var binding: ActivityResummarizeRecordsBinding
    private val selectedRecordIds = linkedSetOf<String>()
    private var records: List<ConversationRecord> = emptyList()
    private var isProcessing = false
    private var lastFailureMessage: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResummarizeRecordsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backButton.setOnClickListener { finish() }
        binding.btnResummarize.setOnClickListener {
            if (!isProcessing) startResummarizeSelected()
        }

        renderRecords()
    }

    override fun onResume() {
        super.onResume()
        ConversationRecordStore.syncFromFirebase(this) {
            runOnUiThread {
                renderRecords()
            }
        }
    }

    private fun renderRecords() {
        records = ConversationRecordStore.getAllActiveRecords(this)
        val previousSelection = selectedRecordIds.toSet()
        selectedRecordIds.clear()
        selectedRecordIds.addAll(previousSelection.filter { selectedId ->
            records.any { record -> recordId(record) == selectedId }
        })
        binding.recordContainer.removeAllViews()

        if (records.isEmpty()) {
            binding.txtEmpty.visibility = View.VISIBLE
        } else {
            binding.txtEmpty.visibility = View.GONE
            records.forEach { record ->
                val row = LayoutInflater.from(this)
                    .inflate(R.layout.item_resummarize_record, binding.recordContainer, false)
                val checkbox = row.findViewById<CheckBox>(R.id.checkboxRecord)
                val titleView = row.findViewById<TextView>(R.id.txtRecordTitle)
                val dateView = row.findViewById<TextView>(R.id.txtRecordDate)
                val id = recordId(record)

                checkbox.isChecked = selectedRecordIds.contains(id)
                titleView.text = record.title
                dateView.text = formatRecordDate(record.createdAt)

                row.setOnClickListener {
                    if (isProcessing) return@setOnClickListener
                    checkbox.isChecked = !checkbox.isChecked
                    syncSelection(id, checkbox.isChecked)
                }
                checkbox.setOnClickListener {
                    if (isProcessing) return@setOnClickListener
                    syncSelection(id, checkbox.isChecked)
                }

                binding.recordContainer.addView(row)
            }
        }

        updateActionState()
    }

    private fun startResummarizeSelected() {
        val selectedRecords = records.filter { selectedRecordIds.contains(recordId(it)) }
        if (selectedRecords.isEmpty()) {
            Toast.makeText(this, R.string.settings_resummary_select_required, Toast.LENGTH_SHORT).show()
            return
        }

        if (BuildConfig.OPENAI_API_KEY.trim().isBlank()) {
            Toast.makeText(this, R.string.settings_resummary_api_missing, Toast.LENGTH_SHORT).show()
            return
        }

        isProcessing = true
        lastFailureMessage = null
        updateActionState()
        processResummaryQueue(selectedRecords, 0, 0)
    }

    private fun processResummaryQueue(
        selectedRecords: List<ConversationRecord>,
        index: Int,
        successCount: Int,
    ) {
        if (index >= selectedRecords.size) {
            isProcessing = false
            val completionMessage = if (successCount == 0) {
                lastFailureMessage ?: getString(R.string.settings_resummary_failed_default)
            } else {
                getString(R.string.settings_resummary_done, successCount, selectedRecords.size)
            }
            Toast.makeText(this, completionMessage, Toast.LENGTH_LONG).show()
            ConversationRecordStore.syncFromFirebase(this) {
                runOnUiThread {
                    renderRecords()
                }
            }
            return
        }

        val record = selectedRecords[index]
        binding.txtGuide.text = getString(
            R.string.settings_resummary_progress,
            index + 1,
            selectedRecords.size,
        )

        OpenAiSummaryService.summarizeConversationDetailed(buildConversationForSummary(record)) { result ->
            val summary = result.text?.trim().orEmpty()
            runOnUiThread {
                if (summary.isBlank()) {
                    if (!result.errorMessage.isNullOrBlank()) {
                        lastFailureMessage = result.errorMessage
                    }
                    processResummaryQueue(selectedRecords, index + 1, successCount)
                } else {
                    ConversationRecordStore.updateRecordTitle(
                        context = this,
                        dateKey = record.dateKey,
                        createdAt = record.createdAt,
                        title = summary,
                    ) {
                        runOnUiThread {
                            processResummaryQueue(selectedRecords, index + 1, successCount + 1)
                        }
                    }
                }
            }
        }
    }

    private fun buildConversationForSummary(record: ConversationRecord): String {
        val lines = mutableListOf<String>()
        appendConversationLines(lines, "me", record.selfSpeech, record.selfSummary)
        appendConversationLines(lines, "other", record.otherSpeech, record.otherSummary)
        return lines.joinToString(separator = "\n").trim()
    }

    private fun appendConversationLines(
        target: MutableList<String>,
        speaker: String,
        speech: String,
        summary: String,
    ) {
        speech.lines()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .forEach { target.add("$speaker: $it") }

        summary.lines()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .forEach { target.add("$speaker summary: $it") }
    }

    private fun syncSelection(recordId: String, isChecked: Boolean) {
        if (isChecked) {
            selectedRecordIds.add(recordId)
        } else {
            selectedRecordIds.remove(recordId)
        }
        updateActionState()
    }

    private fun updateActionState() {
        binding.btnResummarize.isEnabled = selectedRecordIds.isNotEmpty() && !isProcessing
        binding.btnResummarize.alpha = if (binding.btnResummarize.isEnabled) 1f else 0.55f
        binding.progressBar.visibility = if (isProcessing) View.VISIBLE else View.GONE
        binding.recordScroll.alpha = if (isProcessing) 0.76f else 1f

        if (!isProcessing) {
            binding.txtGuide.text = if (selectedRecordIds.isEmpty()) {
                getString(R.string.settings_resummary_guide_default)
            } else {
                getString(R.string.settings_resummary_guide, selectedRecordIds.size)
            }
        }
    }

    private fun formatRecordDate(createdAt: Long): String {
        return SimpleDateFormat("yy년 M월 d일", Locale.KOREA).format(Date(createdAt))
    }

    private fun recordId(record: ConversationRecord): String = "${record.dateKey}_${record.createdAt}"
}
