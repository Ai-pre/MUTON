package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MutonInk
import com.example.myapplication.ui.theme.MutonPanel
import com.example.myapplication.ui.theme.MutonSoftText

@Composable
fun ProfileScreen(
    onBackClick: () -> Unit,
    onChangePasswordClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(horizontal = 26.dp, vertical = 24.dp)
    ) {
        BackLabel(onClick = onBackClick)
        Spacer(modifier = Modifier.padding(top = 36.dp))
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(CircleShape)
                .background(MutonPanel)
                .align(Alignment.CenterHorizontally)
        )
        Spacer(modifier = Modifier.padding(top = 14.dp))
        Text(
            text = "user",
            modifier = Modifier.align(Alignment.CenterHorizontally),
            fontWeight = FontWeight.SemiBold
        )
        Spacer(modifier = Modifier.padding(top = 34.dp))
        ProfileLine("Birthday date", "08 / 24")
        ProfileLine("Change password", ">", onChangePasswordClick)
        ProfileLine("Email", "hi@muton.ai")
        ProfileLine("Nickname", "hirin")
        Spacer(modifier = Modifier.weight(1f))
        Text(text = "log out", color = MutonSoftText)
    }
}

@Composable
private fun ProfileLine(
    label: String,
    value: String,
    onClick: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 14.dp)
            .clickable(enabled = onClick != null) { onClick?.invoke() },
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, color = MutonSoftText)
        Text(text = value, color = MutonInk)
    }
}