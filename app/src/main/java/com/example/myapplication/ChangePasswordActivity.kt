package com.example.myapplication

import android.os.Bundle
import android.text.method.HideReturnsTransformationMethod
import android.text.method.PasswordTransformationMethod
import android.widget.Toast
import com.example.myapplication.databinding.ActivityChangePasswordBinding
import com.google.firebase.auth.EmailAuthProvider
import com.google.firebase.auth.FirebaseAuth

class ChangePasswordActivity : BaseActivity() {

    private lateinit var binding: ActivityChangePasswordBinding
    private var isConfirmVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChangePasswordBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backButton.setOnClickListener { finish() }
        binding.inputConfirmPassword.transformationMethod = PasswordTransformationMethod.getInstance()
        binding.btnConfirmVisibility.alpha = 0.55f

        binding.btnConfirmVisibility.setOnClickListener {
            isConfirmVisible = !isConfirmVisible
            binding.inputConfirmPassword.transformationMethod =
                if (isConfirmVisible) HideReturnsTransformationMethod.getInstance()
                else PasswordTransformationMethod.getInstance()
            binding.btnConfirmVisibility.alpha = if (isConfirmVisible) 1f else 0.55f
            binding.inputConfirmPassword.setSelection(binding.inputConfirmPassword.text?.length ?: 0)
        }

        binding.btnChange.setOnClickListener {
            changePassword()
        }
    }

    private fun changePassword() {
        val currentPassword = binding.inputCurrentPassword.text.toString()
        val newPassword = binding.inputNewPassword.text.toString()
        val confirmPassword = binding.inputConfirmPassword.text.toString()
        val user = FirebaseAuth.getInstance().currentUser
        val email = user?.email

        when {
            currentPassword.isBlank() || newPassword.isBlank() || confirmPassword.isBlank() -> {
                Toast.makeText(this, R.string.login_empty_fields, Toast.LENGTH_SHORT).show()
                return
            }
            newPassword.length < 6 -> {
                Toast.makeText(this, R.string.signup_password_short, Toast.LENGTH_SHORT).show()
                return
            }
            newPassword != confirmPassword -> {
                Toast.makeText(this, R.string.signup_password_mismatch, Toast.LENGTH_SHORT).show()
                return
            }
            user == null || email.isNullOrBlank() -> {
                Toast.makeText(this, R.string.settings_password_user_missing, Toast.LENGTH_SHORT).show()
                return
            }
        }

        val credential = EmailAuthProvider.getCredential(email, currentPassword)
        user.reauthenticate(credential)
            .addOnSuccessListener {
                user.updatePassword(newPassword)
                    .addOnSuccessListener {
                        Toast.makeText(this, R.string.settings_password_saved, Toast.LENGTH_SHORT).show()
                        finish()
                    }
                    .addOnFailureListener { error ->
                        Toast.makeText(
                            this,
                            getString(R.string.settings_password_failed, error.localizedMessage ?: error.message.orEmpty()),
                            Toast.LENGTH_LONG,
                        ).show()
                    }
            }
            .addOnFailureListener {
                Toast.makeText(this, R.string.settings_current_password_wrong, Toast.LENGTH_SHORT).show()
            }
    }
}
