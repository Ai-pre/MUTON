package com.example.myapplication

import android.os.Bundle
import android.view.LayoutInflater
import android.widget.TextView
import com.example.myapplication.databinding.ActivityTrashBinding

class TrashActivity : BaseActivity() {

    private lateinit var binding: ActivityTrashBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTrashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backButton.setOnClickListener { finish() }
        renderTrash()
    }

    override fun onResume() {
        super.onResume()
        ConversationRecordStore.syncFromFirebase(this) {
            runOnUiThread {
                renderTrash()
            }
        }
    }

    private fun renderTrash() {
        val records = ConversationRecordStore.getTrashedRecords(this)
        binding.trashContainer.removeAllViews()

        if (records.isEmpty()) {
            binding.trashContainer.addView(
                TextView(this).apply {
                    text = getString(R.string.trash_empty)
                    textSize = 14f
                    setTextColor(getColor(R.color.muton_muted))
                },
            )
            return
        }

        records.forEach { record ->
            val row = LayoutInflater.from(this)
                .inflate(R.layout.item_trash_record, binding.trashContainer, false)
            row.findViewById<TextView>(R.id.txtTrashItemTitle).text = record.title
            row.findViewById<TextView>(R.id.txtTrashItemTime).text = record.timeRange
            row.findViewById<android.widget.ImageView>(R.id.btnPermanentDelete).setOnClickListener {
                ConversationRecordStore.permanentlyDelete(this, record.dateKey, record.createdAt)
                renderTrash()
            }
            binding.trashContainer.addView(row)
        }
    }
}
