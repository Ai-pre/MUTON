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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
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
import com.example.myapplication.ui.theme.MutonBlue
import com.example.myapplication.ui.theme.MutonInk
import com.example.myapplication.ui.theme.MutonSoftText

@Composable
fun CalendarScreen(
    onBackClick: () -> Unit,
    onRecordClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(horizontal = 24.dp, vertical = 24.dp)
    ) {
        BackLabel(onClick = onBackClick)
        Spacer(modifier = Modifier.height(24.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("March 2026", fontSize = 28.sp, fontWeight = FontWeight.SemiBold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CircleChip("<")
                CircleChip("...")
            }
        }
        Spacer(modifier = Modifier.height(24.dp))
        CalendarGrid()
        Spacer(modifier = Modifier.height(28.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("Wednesday 25", fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
            Text("2 records", color = MutonSoftText)
        }
        Spacer(modifier = Modifier.height(18.dp))
        ScheduleRow("수강 정정 문의", "09:00 - 09:30 am", onRecordClick)
        Spacer(modifier = Modifier.height(16.dp))
        ScheduleRow("생일파티 의논", "10:00 - 11:00 am", onRecordClick)
    }
}

@Composable
fun BackLabel(onClick: () -> Unit) {
    Text(
        text = "< back",
        color = MutonSoftText,
        modifier = Modifier.clickable(onClick = onClick)
    )
}

@Composable
private fun CalendarGrid() {
    val rows = listOf(
        listOf("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"),
        listOf("22", "23", "24", "25", "26", "27", "28"),
        listOf("29", "30", "31", "1", "2", "3", "4")
    )

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(Color(0xFFF8F8F8))
            .padding(horizontal = 14.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(22.dp)
    ) {
        rows.forEachIndexed { rowIndex, row ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                row.forEach { label ->
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(
                                if (rowIndex == 1 && label == "25") MutonBlue else Color.Transparent
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = label,
                            color = if (rowIndex == 1 && label == "25") Color.White else MutonInk,
                            fontSize = if (rowIndex == 0) 12.sp else 14.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ScheduleRow(title: String, time: String, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(Color(0xFFF8F8F8))
            .clickable(onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(title, fontWeight = FontWeight.Medium)
            Spacer(modifier = Modifier.height(6.dp))
            Text(time, color = MutonSoftText)
        }
        Text("*", color = Color(0xFFFFC83D), fontSize = 26.sp)
    }
}