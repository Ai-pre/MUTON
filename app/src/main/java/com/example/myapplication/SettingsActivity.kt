package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
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

        val sizeOptions = listOf("small", "medium", "large")
        binding.spinnerTextSize.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            sizeOptions,
        )

        val currentOption = AppTextScaleManager.getTextSizeOption(this)
        binding.spinnerTextSize.setSelection(
            AppTextScaleManager.TextSizeOption.entries.indexOf(currentOption),
            false,
        )
        binding.spinnerTextSize.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedOption = AppTextScaleManager.TextSizeOption.entries[position]
                if (selectedOption == AppTextScaleManager.getTextSizeOption(this@SettingsActivity)) {
                    return
                }

                AppTextScaleManager.saveTextSizeOption(this@SettingsActivity, selectedOption)
                recreate()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
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
