package com.example.medicalaiguidance.screen

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.widget.Toast
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.painter.Painter
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.medicalaiguidance.R
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.service.MyAccessibilityService
import com.example.medicalaiguidance.service.toAccessibilityAppointmentTypeOrNull
import com.example.medicalaiguidance.viewmodel.ConfirmViewModel
import java.util.Calendar
import java.util.Locale

@Composable
fun ConfirmNeedScreen(
    navController: NavController,
    viewModel: ConfirmViewModel = viewModel()
) {
    val context = LocalContext.current
    val appointment by viewModel.appointmentInfo.collectAsState()
    val visitPlan by viewModel.visitPlan.collectAsState()

    // 控制提醒彈窗的顯示
    var showPermissionDialog by remember { mutableStateOf(false) }
    // 控制再次確認彈窗
    var showConfirmDialog by remember { mutableStateOf(false) }
    // 是否開啟智慧視覺導引功能 (預設為開啟)
    var isGuidanceEnabled by remember { mutableStateOf(true) }

    val primaryDark = Color(0xFF376F72)
    val badgeBg = Color(0xFFD9EAE7)
    val bgGradient = Brush.verticalGradient(colors = listOf(Color(0xFFF2FAF8), Color(0xFFF2FAF8)))

    val fontScale = LocalDensity.current.fontScale
    val cardMinHeight = when {
        fontScale >= 1.3f -> 480.dp
        fontScale >= 1.15f -> 440.dp
        else -> 400.dp
    }
    val cardMaxHeight = when {
        fontScale >= 1.3f -> 600.dp
        fontScale >= 1.15f -> 540.dp
        else -> 480.dp
    }

    // 權限檢查邏輯
    fun isServiceEnabled(): Boolean {
        val expectedService = "${context.packageName}/${MyAccessibilityService::class.java.name}"
        val enabled = try {
            Settings.Secure.getInt(context.contentResolver, Settings.Secure.ACCESSIBILITY_ENABLED)
        } catch (e: Exception) { 0 }

        if (enabled == 1) {
            val enabledServices = Settings.Secure.getString(context.contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES)
            return enabledServices?.contains(expectedService) == true
        }
        return false
    }

    if (appointment == null) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator(color = primaryDark)
        }
        return
    }

    val currentApt = appointment!!
    val targetDept = currentApt.department.name
    val targetClinic = currentApt.department.clinicName
    val targetDoctor = currentApt.doctor.name
    val targetTime = currentApt.timeSlot
    val targetDisplayTime = formatConfirmVisitTime(
        date = currentApt.date,
        dayOfWeek = currentApt.dayOfWeek,
        timeSlot = currentApt.timeSlot
    )

    // 提醒小視窗 (AlertDialog)
    if (showPermissionDialog) {
        AlertDialog(
            onDismissRequest = { showPermissionDialog = false },
            shape = RoundedCornerShape(24.dp),
            containerColor = Color.White,
            title = {
                Text(
                    text = "需要開啟導引權限",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF1A2E2E)
                )
            },
            text = {
                Text(
                    text = "為協助您完成掛號，智慧視覺導引需要使用協助工具權限。開啟後，系統會在台北榮總 App 中以紅框提示下一步操作的位置。",
                    fontSize = 16.sp,
                    lineHeight = 22.sp,
                    color = Color(0xFF556666)
                )
            },
            confirmButton = {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            //.background(Color(0xFFD5E5E5).copy(alpha = 0.35f), RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable { showPermissionDialog = false },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "取消", color = Color(0xFF7A8B8B), fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .background(color = primaryDark, RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable {
                                showPermissionDialog = false
                                context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "前往開啟", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }
            },
            dismissButton = null
        )
    }

    // 再次確認視窗
    if (showConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showConfirmDialog = false },
            shape = RoundedCornerShape(24.dp),
            containerColor = Color.White,
            title = {
                Text(
                    text = "再次確認",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF1A2E2E)
                )
            },
            text = {
                Text(
                    text = if (isGuidanceEnabled) "即將前往台北榮總 App，並啟動「智慧視覺導引」，協助您依照提示完成掛號。" else "即將前往台北榮總 App，您將自行操作完成掛號。",
                    fontSize = 16.sp,
                    lineHeight = 24.sp,
                    color = Color(0xFF556666)
                )
            },
            confirmButton = {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            //.background(Color(0xFFD5E5E5).copy(alpha = 0.35f), RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable { showConfirmDialog = false },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "取消", color = Color(0xFF7A8B8B), fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .background(color = primaryDark, RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable {
                                showConfirmDialog = false
                                val appointmentType = visitPlan.toAccessibilityAppointmentTypeOrNull()
                                if (isGuidanceEnabled && appointmentType == null) {
                                    Toast.makeText(
                                        context,
                                        "無法確認就診類型，請返回重新選擇掛號流程。",
                                        Toast.LENGTH_LONG
                                    ).show()
                                } else {
                                    if (isGuidanceEnabled) {
                                        MyAccessibilityService.updateTarget(
                                            department = targetDept,
                                            clinic = targetClinic,
                                            doctor = targetDoctor,
                                            date = currentApt.date,
                                            timeSlot = targetTime,
                                            appointmentType = requireNotNull(appointmentType)
                                        )
                                    }
                                    val packageName = "tw.com.bicom.VGHTPE"
                                    viewModel.launchHospitalAfterVoiceCleanup {
                                        val launchIntent = context.packageManager.getLaunchIntentForPackage(packageName)
                                        if (launchIntent != null) {
                                            context.startActivity(launchIntent)
                                        } else {
                                            val webIntent = Intent(Intent.ACTION_VIEW, Uri.parse("https://www.vghtpe.gov.tw/Index.action"))
                                            context.startActivity(webIntent)
                                        }
                                    }
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "確認前往", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }
            },
            dismissButton = null
        )
    }

    Column(
        modifier = Modifier.fillMaxSize().background(brush = bgGradient).statusBarsPadding()
    ) {
        // 1. 頂部導航列
        Box(modifier = Modifier.fillMaxWidth().height(80.dp).padding(horizontal = 20.dp), contentAlignment = Alignment.Center) {
            Box(
                modifier = Modifier.align(Alignment.CenterStart).size(48.dp).shadow(6.dp, RoundedCornerShape(16.dp))
                    .background(Color.White, RoundedCornerShape(16.dp)).clickable { navController.popBackStack() },
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.ArrowBack, "返回", tint = primaryDark, modifier = Modifier.size(22.dp))
            }
            Text(text = "準備好了嗎", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = primaryDark)
        }

        // 2. 核心大卡片與智慧導引狀態區塊
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f, fill = false)
                .padding(horizontal = 24.dp, vertical = 4.dp)
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .offset(x = 12.dp, y = (-55).dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.sheep_2),
                    contentDescription = "安心陪伴者",
                    modifier = Modifier
                        .size(110.dp)
                        .clip(CircleShape)
                )
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 28.dp)
                    .heightIn(min = cardMinHeight, max = cardMaxHeight)
                    .shadow(2.dp, RoundedCornerShape(30.dp))
                    .border(width = 10.dp, color = badgeBg.copy(alpha = 0.8f), shape = RoundedCornerShape(30.dp)),
                shape = RoundedCornerShape(30.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(modifier = Modifier.fillMaxSize()) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 36.dp, bottom = 12.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .background(badgeBg, RoundedCornerShape(20.dp))
                                .padding(horizontal = 16.dp, vertical = 8.dp)
                        ) {
                            Text(
                                text = "掛號資訊確認",
                                fontSize = 20.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = primaryDark
                            )
                        }
                    }

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .verticalScroll(rememberScrollState())
                            .padding(horizontal = 28.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(20.dp)
                    ) {
                        InfoRowItem(
                            icon = Icons.Default.LocalHospital,
                            iconPainter = painterResource(id = R.drawable.ic_hospital),
                            label = "科別",
                            title = "$targetDept $targetClinic"
                        )
                        HorizontalDivider(color = Color(0xFFEBF2F2), thickness = 1.dp)
                        InfoRowItem(
                            icon = Icons.Default.Person,
                            iconPainter = painterResource(id = R.drawable.ic_doctor),
                            label = "看診醫師",
                            title = "$targetDoctor 醫師"
                        )
                        HorizontalDivider(color = Color(0xFFEBF2F2), thickness = 1.dp)
                        InfoRowItem(
                            icon = Icons.Default.AccessTime,
                            iconPainter = painterResource(id = R.drawable.ic_time),
                            label = "預約時間",
                            title = targetDisplayTime
                        )
                    }

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFF9FCFC))
                    ) {
                        HorizontalDivider(color = Color(0xFFEBF2F2), thickness = 1.dp)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 24.dp, vertical = 16.dp)
                                .padding(bottom = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.TipsAndUpdates,
                                    contentDescription = null,
                                    tint = primaryDark,
                                    modifier = Modifier.size(26.dp)
                                )
                                Spacer(modifier = Modifier.width(10.dp))
                                Column {
                                    Text(
                                        text = "啟用智慧視覺導引",
                                        fontSize = 15.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = primaryDark
                                    )
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text(
                                        text = if (isGuidanceEnabled) "在榮總 App 顯示紅框提示步驟" else "直接開啟榮總 App 自行操作",
                                        fontSize = 12.sp,
                                        color = Color(0xFF556666)
                                    )
                                }
                            }
                            Switch(
                                checked = isGuidanceEnabled,
                                onCheckedChange = { isGuidanceEnabled = it },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = Color.White,
                                    checkedTrackColor = primaryDark,
                                    uncheckedThumbColor = Color.White,
                                    uncheckedTrackColor = Color(0xFFB0C4C4),
                                    uncheckedBorderColor = Color.Transparent
                                )
                            )
                        }
                    }
                }
            }

            Image(
                painter = painterResource(id = R.drawable.ic_clip),
                contentDescription = null,
                colorFilter = ColorFilter.tint(Color(0xFF2C4E4E)),
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .offset(y = (-3).dp)
                    .size(width = 130.dp, height = 84.dp)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 3. 底部按鈕列
        Column(
            modifier = Modifier.fillMaxWidth().navigationBarsPadding().padding(start = 24.dp, end = 24.dp, bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // 前往掛號
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .background(color = primaryDark, RoundedCornerShape(20.dp))
                    .clickable {
                        if (isGuidanceEnabled) {
                            if (isServiceEnabled()) {
                                showConfirmDialog = true
                            } else {
                                showPermissionDialog = true
                            }
                        } else {
                            showConfirmDialog = true
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Launch, null, tint = Color.White, modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("前往掛號", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }
            }

            // 重新詢問
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .border(
                        width = 1.dp,
                        color = primaryDark.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(20.dp)
                    )
                    .background(
                        color = Color(0xFFD5E5E5).copy(alpha = 0.7f),
                        shape = RoundedCornerShape(20.dp)
                    )
                    .clickable {
                        navController.navigate(Route.CHAT) { popUpTo(Route.HOME) }
                    },
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "重新詢問",
                        tint = primaryDark,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "重新詢問",
                        color = primaryDark,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }
    }
}

private fun formatConfirmVisitTime(
    date: String,
    dayOfWeek: String,
    timeSlot: String
): String {
    val calendar = parseConfirmDate(date) ?: parseConfirmDate(dayOfWeek)
    val dateText = calendar?.let {
        String.format(
            Locale.TAIWAN,
            "%04d/%02d/%02d",
            it.get(Calendar.YEAR),
            it.get(Calendar.MONTH) + 1,
            it.get(Calendar.DAY_OF_MONTH)
        )
    } ?: date.trim()

    val weekdayText = extractConfirmWeekday(date)
        ?: extractConfirmWeekday(dayOfWeek)
        ?: calendar?.let { "(${confirmWeekdayName(it)})" }

    return listOfNotNull(
        dateText.takeIf { it.isNotBlank() },
        weekdayText,
        normalizeConfirmSession(timeSlot).takeIf { it.isNotBlank() }
    ).joinToString(" ")
}

private fun parseConfirmDate(raw: String): Calendar? {
    val match = Regex("""(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})""").find(raw) ?: return null
    val year = match.groupValues[1].toIntOrNull() ?: return null
    val month = match.groupValues[2].toIntOrNull() ?: return null
    val day = match.groupValues[3].toIntOrNull() ?: return null
    return try {
        Calendar.getInstance(Locale.TAIWAN).apply {
            isLenient = false
            set(Calendar.YEAR, year)
            set(Calendar.MONTH, month - 1)
            set(Calendar.DAY_OF_MONTH, day)
            getTime()
        }
    } catch (e: IllegalArgumentException) {
        null
    }
}

private fun confirmWeekdayName(calendar: Calendar): String =
    when (calendar.get(Calendar.DAY_OF_WEEK)) {
        Calendar.MONDAY -> "一"
        Calendar.TUESDAY -> "二"
        Calendar.WEDNESDAY -> "三"
        Calendar.THURSDAY -> "四"
        Calendar.FRIDAY -> "五"
        Calendar.SATURDAY -> "六"
        else -> "日"
    }

private fun extractConfirmWeekday(raw: String): String? =
    Regex("""\(([日一二三四五六])\)""").find(raw)?.value

private fun normalizeConfirmSession(timeSlot: String): String {
    val raw = timeSlot.trim()
    if (raw.isBlank()) return ""
    return when {
        raw.contains("上午") || raw.contains("早") -> "上午診"
        raw.contains("下午") || raw.contains("午") -> "下午診"
        raw.contains("夜") || raw.contains("晚") -> "夜診"
        else -> inferConfirmSessionFromTime(raw) ?: raw
    }
}

private fun inferConfirmSessionFromTime(raw: String): String? {
    val hour = Regex("""(\d{1,2})\s*:""").find(raw)?.groupValues?.getOrNull(1)?.toIntOrNull()
        ?: return null
    return when {
        hour < 12 -> "上午診"
        hour < 18 -> "下午診"
        else -> "夜診"
    }
}

@Composable
fun InfoRowItem(
    icon: ImageVector,
    label: String,
    title: String,
    subLabel: String? = null,
    iconPainter: Painter? = null
) {
    Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(modifier = Modifier.weight(1f)) {
            Text(label, fontSize = 14.sp, color = Color(0xFF7A8B8B), fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(4.dp))
            Text(title, fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1A2E2E))
            subLabel?.let { Text(it, fontSize = 15.sp, color = Color(0xFF4A5959)) }
        }
        Spacer(Modifier.width(12.dp))
        if (iconPainter != null) {
            Icon(iconPainter, null, tint = Color(0xFF2C4E4E), modifier = Modifier.size(46.dp))
        } else {
            Icon(icon, null, tint = Color(0xFF2C4E4E), modifier = Modifier.size(46.dp))
        }
    }
}
