package com.example.myapplication

import android.animation.AnimatorSet
import android.animation.ArgbEvaluator
import android.animation.ObjectAnimator
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.animation.AccelerateDecelerateInterpolator
import com.example.myapplication.databinding.ActivitySplashBinding

class SplashActivity : BaseActivity() {

    private lateinit var binding: ActivitySplashBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        playSplashAnimation()
    }

    private fun playSplashAnimation() {
        val logoFade = ObjectAnimator.ofFloat(binding.imgSplashLogo, "alpha", 0f, 1f).apply {
            duration = 520L
            startDelay = 180L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val logoScaleX = ObjectAnimator.ofFloat(binding.imgSplashLogo, "scaleX", 0.86f, 1f).apply {
            duration = 520L
            startDelay = 180L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val logoScaleY = ObjectAnimator.ofFloat(binding.imgSplashLogo, "scaleY", 0.86f, 1f).apply {
            duration = 520L
            startDelay = 180L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val muFade = ObjectAnimator.ofFloat(binding.txtSplashMu, "alpha", 0f, 1f).apply {
            duration = 520L
            startDelay = 420L
            interpolator = AccelerateDecelerateInterpolator()
        }
        val brandColor = ObjectAnimator.ofObject(
            binding.txtSplashBrand,
            "textColor",
            ArgbEvaluator(),
            Color.parseColor("#111827"),
            Color.parseColor("#FFFFFF"),
        ).apply {
            duration = 620L
            startDelay = 420L
            interpolator = AccelerateDecelerateInterpolator()
        }

        AnimatorSet().apply {
            playTogether(logoFade, logoScaleX, logoScaleY, muFade, brandColor)
            start()
        }

        binding.brandStage.postDelayed({
            startActivity(Intent(this, LoginActivity::class.java))
            finish()
        }, 1420L)
    }
}
