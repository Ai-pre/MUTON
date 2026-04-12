package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.widget.SeekBar
import com.google.firebase.auth.FirebaseAuth
import com.example.myapplication.databinding.ActivitySettingsBinding

class SettingsActivity : BaseActivity() {

    private lateinit var binding: ActivitySettingsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backButton.setOnClickListener { finish() }
        binding.rowEditProfile.setOnClickListener {
            startActivity(Intent(this, EditProfileActivity::class.java))
        }
        binding.rowChangePassword.setOnClickListener {
            startActivity(Intent(this, ChangePasswordActivity::class.java))
        }
        binding.txtLogout.setOnClickListener {
            FirebaseAuth.getInstance().signOut()
            startActivity(Intent(this, LoginActivity::class.java))
            finishAffinity()
        }

        val currentOption = AppTextScaleManager.getTextSizeOption(this)
        binding.seekTextSize.progress = AppTextScaleManager.TextSizeOption.entries.indexOf(currentOption)
        binding.seekTextSize.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser) return

                val selectedOption = AppTextScaleManager.TextSizeOption.entries[progress]
                if (selectedOption == AppTextScaleManager.getTextSizeOption(this@SettingsActivity)) {
                    return
                }

                AppTextScaleManager.saveTextSizeOption(this@SettingsActivity, selectedOption)
                recreate()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
    }

    override fun onResume() {
        super.onResume()
        loadProfile()
    }

    private fun loadProfile() {
        FirebaseUserStore.loadProfile(this) { profile ->
            runOnUiThread {
                binding.txtUserName.text = profile.displayName
            }
        }
    }
}
