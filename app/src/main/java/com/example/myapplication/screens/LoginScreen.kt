package com.example.myapplication.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
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
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MutonBlue
import com.example.myapplication.ui.theme.MutonInk
import com.example.myapplication.ui.theme.MutonSoft
import com.example.myapplication.ui.theme.MutonSoftText

@Composable
fun LoginScreen(
    onLoginClick: () -> Unit,
    onSignUpClick: () -> Unit
) {
    var id by remember { mutableStateOf("hirin") }
    var password by remember { mutableStateOf("1234567890") }
    var passwordCheck by remember { mutableStateOf("1234567890") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .padding(horizontal = 42.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(218.dp))
        MutonLogoMark(modifier = Modifier.size(width = 91.dp, height = 78.dp))
        Spacer(modifier = Modifier.height(68.dp))
        MutonInputField(value = id, onValueChange = { id = it }, placeholder = "ID")
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
        Spacer(modifier = Modifier.height(15.dp))
        OutlinePillButton(text = "sign up", onClick = onSignUpClick)
        Spacer(modifier = Modifier.height(46.dp))
        FilledPillButton(text = "log in", onClick = onLoginClick)
    }
}

@Composable
fun MutonLogoMark(modifier: Modifier = Modifier) {
    Box(modifier = modifier.width(91.dp).height(78.dp)) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(MutonBlue)
        )
        Box(
            modifier = Modifier
                .size(40.dp)
                .offset(x = 28.dp)
                .clip(CircleShape)
                .background(MutonBlue)
        )
        Box(
            modifier = Modifier
                .width(68.dp)
                .height(42.dp)
                .offset(y = 24.dp)
                .background(MutonBlue)
        )
        Text(
            text = "mu",
            color = Color.White,
            fontSize = 34.sp,
            fontWeight = FontWeight.ExtraBold,
            modifier = Modifier.offset(x = 10.dp, y = 24.dp)
        )
    }
}

@Composable
fun MutonInputField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    password: Boolean = false,
    trailing: String? = null
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(45.dp)
            .clip(RoundedCornerShape(40.dp))
            .background(MutonSoft)
            .border(1.dp, MutonBlue, RoundedCornerShape(40.dp))
            .padding(horizontal = 26.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            singleLine = true,
            textStyle = TextStyle(fontSize = 16.sp, color = MutonInk),
            visualTransformation = if (password) PasswordVisualTransformation() else VisualTransformation.None,
            modifier = Modifier.weight(1f),
            decorationBox = { inner ->
                Box(contentAlignment = Alignment.CenterStart) {
                    if (value.isEmpty()) {
                        Text(text = placeholder, color = MutonSoftText)
                    }
                    inner()
                }
            }
        )
        if (trailing != null) {
            Text(text = trailing, color = MutonSoftText)
        }
    }
}

@Composable
fun OutlinePillButton(text: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(40.dp))
            .border(1.dp, MutonBlue, RoundedCornerShape(40.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 22.dp, vertical = 4.dp)
    ) {
        Text(text = text, color = MutonBlue, fontSize = 15.sp)
    }
}

@Composable
fun FilledPillButton(text: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(40.dp))
            .background(MutonSoft)
            .clickable(onClick = onClick)
            .padding(horizontal = 38.dp, vertical = 10.dp)
    ) {
        Text(text = text, color = MutonSoftText, fontSize = 18.sp)
    }
}