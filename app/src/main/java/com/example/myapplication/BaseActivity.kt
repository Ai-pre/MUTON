package com.example.myapplication

import android.content.Context
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

abstract class BaseActivity : AppCompatActivity() {

    private var appliedTextSizeOption: AppTextScaleManager.TextSizeOption? = null

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(AppTextScaleManager.wrapContext(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        appliedTextSizeOption = AppTextScaleManager.getTextSizeOption(this)
        super.onCreate(savedInstanceState)
    }

    override fun onResume() {
        super.onResume()
        val currentOption = AppTextScaleManager.getTextSizeOption(this)
        if (appliedTextSizeOption != null && appliedTextSizeOption != currentOption) {
            appliedTextSizeOption = currentOption
            recreate()
        }
    }
}
