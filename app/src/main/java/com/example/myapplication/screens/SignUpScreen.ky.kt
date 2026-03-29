package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MutonBlue
import com.example.myapplication.ui.theme.MutonSoftText

@Composable
fun SignUpScreen(
    onCompleteClick: () -> Unit,
    onBackClick: () -> Unit
) {
    var name by remember { mutableStateOf("Name") }
    var password by remember { mutableStateOf("1234567890") }
    var passwordCheck by remember { mutableStateOf("1234567890") }
    var agreeAll by remember { mutableStateOf(false) }
    var agreeTerms by remember { mutableStateOf(false) }
    var agreePrivacy by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .background(Color.White)
            .padding(horizontal = 42.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(143.dp))
        MutonLogoMark(modifier = Modifier.size(width = 91.dp, height = 78.dp))
        Spacer(modifier = Modifier.height(73.dp))
        MutonInputField(value = name, onValueChange = { name = it }, placeholder = "Name")
        Spacer(modifier = Modifier.height(14.dp))
        MutonInputField(
            value = password,
            onValueChange = { password = it },
            placeholder = "Password",
            password = true
        )
        Spacer(modifier = Modifier.height(14.dp))
        MutonInputField(
            value = passwordCheck,
            onValueChange = { passwordCheck = it },
            placeholder = "Password Check",
            password = true,
            trailing = "eye"
        )
        Spacer(modifier = Modifier.height(28.dp))
        ConsentBox(
            agreeAll = agreeAll,
            agreeTerms = agreeTerms,
            agreePrivacy = agreePrivacy,
            onAgreeAll = {
                val next = !agreeAll
                agreeAll = next
                agreeTerms = next
                agreePrivacy = next
            },
            onAgreeTerms = {
                val next = !agreeTerms
                agreeTerms = next
                agreeAll = next && agreePrivacy
            },
            onAgreePrivacy = {
                val next = !agreePrivacy
                agreePrivacy = next
                agreeAll = agreeTerms && next
            }
        )
        Spacer(modifier = Modifier.height(42.dp))
        FilledPillButton(text = "complete", onClick = onCompleteClick)
        Spacer(modifier = Modifier.height(20.dp))
        Text(
            text = "back",
            fontSize = 12.sp,
            color = MutonSoftText,
            modifier = Modifier.clickable(onClick = onBackClick)
        )
        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
private fun ConsentBox(
    agreeAll: Boolean,
    agreeTerms: Boolean,
    agreePrivacy: Boolean,
    onAgreeAll: () -> Unit,
    onAgreeTerms: () -> Unit,
    onAgreePrivacy: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .border(1.dp, MutonBlue, RoundedCornerShape(24.dp))
            .padding(horizontal = 18.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        ConsentRow("전체 동의", agreeAll, onAgreeAll, true)
        ConsentRow("이용약관 동의", agreeTerms, onAgreeTerms, false)
        ConsentRow("개인정보 수집 동의", agreePrivacy, onAgreePrivacy, false)
    }
}

@Composable
private fun ConsentRow(
    text: String,
    checked: Boolean,
    onClick: () -> Unit,
    bold: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = text, fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal)
        Box(
            modifier = Modifier
                .size(18.dp)
                .clip(CircleShape)
                .background(if (checked) MutonBlue else Color.Transparent)
                .border(1.dp, MutonBlue, CircleShape)
        )
    }
}