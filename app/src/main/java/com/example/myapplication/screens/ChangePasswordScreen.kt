package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun ChangePasswordScreen(onBackClick: () -> Unit) {
    var current by remember { mutableStateOf("") }
    var next by remember { mutableStateOf("") }
    var confirm by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(horizontal = 36.dp, vertical = 24.dp)
    ) {
        BackLabel(onClick = onBackClick)
        Spacer(modifier = Modifier.height(64.dp))
        Text(
            text = "새로운 비밀번호를\n설정해주세요.",
            fontSize = 28.sp,
            fontWeight = FontWeight.SemiBold,
            lineHeight = 36.sp
        )
        Spacer(modifier = Modifier.height(52.dp))
        MutonInputField(
            value = current,
            onValueChange = { current = it },
            placeholder = "Current password"
        )
        Spacer(modifier = Modifier.height(14.dp))
        MutonInputField(
            value = next,
            onValueChange = { next = it },
            placeholder = "New password"
        )
        Spacer(modifier = Modifier.height(14.dp))
        MutonInputField(
            value = confirm,
            onValueChange = { confirm = it },
            placeholder = "Confirm password",
            password = true,
            trailing = "eye"
        )
        Spacer(modifier = Modifier.height(42.dp))
        FilledPillButton(text = "change", onClick = {})
    }
}