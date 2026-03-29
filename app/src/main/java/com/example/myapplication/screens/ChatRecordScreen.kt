package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MutonInk
import com.example.myapplication.ui.theme.MutonPanel
import com.example.myapplication.ui.theme.MutonSoftText

@Composable
fun ChatRecordScreen(onBackClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(horizontal = 24.dp, vertical = 24.dp)
    ) {
        BackLabel(onClick = onBackClick)
        Spacer(modifier = Modifier.height(28.dp))
        Text(
            text = "26년 3월 25일\n17:55 대화기록입니다.",
            fontSize = 28.sp,
            fontWeight = FontWeight.SemiBold,
            lineHeight = 36.sp
        )
        Spacer(modifier = Modifier.height(22.dp))
        RoundedCard(height = 113.dp) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp, vertical = 22.dp),
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    "친구와 생일파티 장소를 의논함.",
                    fontSize = 23.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
        Spacer(modifier = Modifier.height(26.dp))
        ThinDivider()
        Spacer(modifier = Modifier.height(18.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp))
                .background(MutonPanel)
                .padding(horizontal = 18.dp, vertical = 22.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                ChatBubble("너 오늘 생일이었나? 미안 까먹었다.", "미안해서 안절부절함")
                ChatBubble(
                    "응 생일이야. 우리집에서 생일파티할건데 놀러올래? 엄마가 맛있는거 해준대!",
                    "따뜻하고 편안한 어조로 달램"
                )
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "delete",
                    color = MutonSoftText,
                    modifier = Modifier.align(Alignment.End)
                )
            }
        }
    }
}

@Composable
private fun ChatBubble(message: String, mood: String) {
    Column(
        modifier = Modifier
            .widthIn(max = 313.dp)
            .clip(RoundedCornerShape(22.dp))
            .background(Color.White)
            .padding(horizontal = 20.dp, vertical = 18.dp)
    ) {
        Text(text = message, color = MutonInk)
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = mood, color = MutonSoftText)
    }
}