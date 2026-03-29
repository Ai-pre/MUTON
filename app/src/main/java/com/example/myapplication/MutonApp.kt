package com.example.myapplication

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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MutonBlue
import com.example.myapplication.ui.theme.MutonInk
import com.example.myapplication.ui.theme.MutonOutline
import com.example.myapplication.ui.theme.MutonPanel
import com.example.myapplication.ui.theme.MutonSoft
import com.example.myapplication.ui.theme.MutonSoftText
import com.example.myapplication.ui.theme.MyApplicationTheme
import kotlinx.coroutines.delay
import com.example.myapplication.screens.LoginScreen

private enum class MutonScreen {
    Splash,
    Login,
    SignUp,
    Home,
    Menu,
    Calendar,
    ChatRecord,
    Profile,
    ChangePassword
}

@Composable
fun MutonApp() {
    var screen by remember { mutableStateOf(MutonScreen.Splash) }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color.White
    ) {
        when (screen) {
            MutonScreen.Splash -> SplashScreen { screen = MutonScreen.Login }
            MutonScreen.Login -> LoginScreen(
                onLoginClick = { screen = MutonScreen.Home },
                onSignUpClick = { screen = MutonScreen.SignUp }
            )
            MutonScreen.SignUp -> SignUpScreen(
                onCompleteClick = { screen = MutonScreen.Home },
                onBackClick = { screen = MutonScreen.Login }
            )
            MutonScreen.Home -> HomeScreen(
                onMenuClick = { screen = MutonScreen.Menu },
                onRecordClick = { screen = MutonScreen.Calendar },
                onChatClick = { screen = MutonScreen.ChatRecord }
            )
            MutonScreen.Menu -> MenuScreen(
                onBackClick = { screen = MutonScreen.Home },
                onProfileClick = { screen = MutonScreen.Profile },
                onPasswordClick = { screen = MutonScreen.ChangePassword }
            )
            MutonScreen.Calendar -> CalendarScreen(
                onBackClick = { screen = MutonScreen.Home },
                onRecordClick = { screen = MutonScreen.ChatRecord }
            )
            MutonScreen.ChatRecord -> ChatRecordScreen(
                onBackClick = { screen = MutonScreen.Calendar }
            )
            MutonScreen.Profile -> ProfileScreen(
                onBackClick = { screen = MutonScreen.Menu },
                onChangePasswordClick = { screen = MutonScreen.ChangePassword }
            )
            MutonScreen.ChangePassword -> ChangePasswordScreen(
                onBackClick = { screen = MutonScreen.Menu }
            )
        }
    }
}

@Composable
private fun SplashScreen(onFinished: () -> Unit) {
    LaunchedEffect(Unit) {
        delay(1400)
        onFinished()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White),
        contentAlignment = Alignment.Center
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            MutonLogoMark()
            Text(
                text = "ton",
                fontSize = 60.sp,
                fontWeight = FontWeight.ExtraBold,
                color = Color.Black
            )
        }
    }
}

@Composable
private fun LoginScreen(
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
private fun SignUpScreen(
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
private fun HomeScreen(
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
private fun MenuScreen(
    onBackClick: () -> Unit,
    onProfileClick: () -> Unit,
    onPasswordClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        HomeScreen(onMenuClick = onBackClick, onRecordClick = {}, onChatClick = {})
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
private fun CalendarScreen(
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
private fun ChatRecordScreen(onBackClick: () -> Unit) {
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
                Text("친구와 생일파티 장소를 의논함.", fontSize = 23.sp, fontWeight = FontWeight.Medium)
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
                ChatBubble("응 생일이야. 우리집에서 생일파티할건데 놀러올래? 엄마가 맛있는거 해준대!", "따뜻하고 편안한 어조로 달램")
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
private fun ProfileScreen(
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
        Spacer(modifier = Modifier.height(36.dp))
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(CircleShape)
                .background(MutonPanel)
                .align(Alignment.CenterHorizontally)
        )
        Spacer(modifier = Modifier.height(14.dp))
        Text(
            text = "user",
            modifier = Modifier.align(Alignment.CenterHorizontally),
            fontWeight = FontWeight.SemiBold
        )
        Spacer(modifier = Modifier.height(34.dp))
        ProfileLine("Birthday date", "08 / 24")
        ProfileLine("Change password", ">", onChangePasswordClick)
        ProfileLine("Email", "hi@muton.ai")
        ProfileLine("Nickname", "hirin")
        Spacer(modifier = Modifier.weight(1f))
        Text(text = "log out", color = MutonSoftText)
    }
}

@Composable
private fun ChangePasswordScreen(onBackClick: () -> Unit) {
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
        MutonInputField(value = current, onValueChange = { current = it }, placeholder = "Current password")
        Spacer(modifier = Modifier.height(14.dp))
        MutonInputField(value = next, onValueChange = { next = it }, placeholder = "New password")
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

@Composable
private fun MutonLogoMark(modifier: Modifier = Modifier) {
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
private fun MutonInputField(
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
private fun OutlinePillButton(text: String, onClick: () -> Unit) {
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
private fun FilledPillButton(text: String, onClick: () -> Unit) {
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
private fun ConsentRow(text: String, checked: Boolean, onClick: () -> Unit, bold: Boolean) {
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

@Composable
private fun RoundedCard(
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
private fun CircleChip(text: String) {
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
private fun MenuLine(text: String, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = text)
        Text(text = ">")
    }
}

@Composable
private fun BackLabel(onClick: () -> Unit) {
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

@Composable
private fun ProfileLine(label: String, value: String, onClick: (() -> Unit)? = null) {
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

@Composable
private fun ThinDivider() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(MutonOutline)
    )
}

@Preview(showBackground = true, widthDp = 412, heightDp = 917)
@Composable
private fun MutonPreview() {
    MyApplicationTheme {
        MutonApp()
    }
}