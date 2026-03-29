package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MutonOutline
import com.example.myapplication.ui.theme.MutonPanel

@Composable
fun MenuScreen(
    onBackClick: () -> Unit,
    onProfileClick: () -> Unit,
    onPasswordClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        HomeScreen(
            onMenuClick = onBackClick,
            onRecordClick = {},
            onChatClick = {}
        )

        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 76.dp, end = 31.dp)
                .width(196.dp)
                .clip(RoundedCornerShape(18.dp))
                .background(MutonPanel)
                .border(1.dp, MutonOutline, RoundedCornerShape(18.dp))
                .padding(horizontal = 18.dp, vertical = 16.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                MenuLine("profile", onProfileClick)
                MenuLine("change password", onPasswordClick)
                MenuLine("close", onBackClick)
            }
        }
    }
}

@Composable
private fun MenuLine(text: String, onClick: () -> Unit) {
    androidx.compose.foundation.layout.Row(
        modifier = Modifier
            .fillMaxSize()
            .clickable(onClick = onClick),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = text)
        Text(text = ">")
    }
}