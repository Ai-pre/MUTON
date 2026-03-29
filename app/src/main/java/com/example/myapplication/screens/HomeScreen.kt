package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MutonBlue
import com.example.myapplication.ui.theme.MutonInk
import com.example.myapplication.ui.theme.MutonOutline
import com.example.myapplication.ui.theme.MutonPanel
import com.example.myapplication.ui.theme.MutonSoft
import com.example.myapplication.ui.theme.MutonSoftText

@Composable
fun HomeScreen(
    onMenuClick: () -> Unit,
    onRecordClick: () -> Unit,
    onChatClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(horizontal = 24.dp, vertical = 26.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            SmallBadge()
            Text(
                text = "menu",
                color = MutonSoftText,
                modifier = Modifier.clickable(onClick = onMenuClick)
            )
        }
        Spacer(modifier = Modifier.height(48.dp))
        Text(
            text = "hello, user!",
            fontSize = 34.sp,
            fontWeight = FontWeight.Bold,
            color = MutonInk
        )
        Spacer(modifier = Modifier.height(22.dp))
        RoundedCard(height = 77.dp, onClick = onChatClick) {
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 20.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = "camera on", fontSize = 20.sp, color = MutonInk)
                Spacer(modifier = Modifier.weight(1f))
                CircleChip(">")
            }
        }
        Spacer(modifier = Modifier.height(22.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(362.dp)
                .clip(RoundedCornerShape(28.dp))
                .background(MutonPanel)
                .padding(horizontal = 18.dp, vertical = 20.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
                Text(text = "sample", color = MutonSoftText)
                ThinDivider()
                RecordRow("sample", "오늘의 대화 기록", onRecordClick)
                RecordRow("sample", "기분이 풀린 날", onRecordClick)
                RecordRow("sample", "다시 듣고 싶은 말", onRecordClick)
            }
        }
        Spacer(modifier = Modifier.height(18.dp))
        PagerDots(selected = 1)
    }
}

@Composable
fun RoundedCard(
    height: Dp,
    onClick: (() -> Unit)? = null,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(height)
            .clip(RoundedCornerShape(26.dp))
            .background(MutonSoft)
            .border(1.dp, MutonOutline, RoundedCornerShape(26.dp))
            .clickable(enabled = onClick != null) { onClick?.invoke() },
        content = content
    )
}

@Composable
private fun SmallBadge() {
    Box(
        modifier = Modifier
            .size(width = 26.dp, height = 23.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(MutonBlue)
    )
}

@Composable
fun CircleChip(text: String) {
    Box(
        modifier = Modifier
            .size(38.dp)
            .clip(CircleShape)
            .background(MutonPanel),
        contentAlignment = Alignment.Center
    ) {
        Text(text = text, color = MutonSoftText)
    }
}

@Composable
private fun RecordRow(title: String, subtitle: String, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = title, color = MutonInk)
        Spacer(modifier = Modifier.width(16.dp))
        Text(text = subtitle, color = MutonSoftText, modifier = Modifier.weight(1f))
        Text(text = ">", color = MutonSoftText)
    }
}

@Composable
private fun PagerDots(selected: Int) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center
    ) {
        repeat(3) { index ->
            Box(
                modifier = Modifier
                    .padding(horizontal = 6.dp)
                    .size(10.dp)
                    .clip(CircleShape)
                    .background(if (index == selected) MutonBlue else MutonPanel)
            )
        }
    }
}

@Composable
fun ThinDivider() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(MutonOutline)
    )
}