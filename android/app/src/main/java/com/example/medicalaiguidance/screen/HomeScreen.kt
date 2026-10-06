package com.example.medicalaiguidance.screen

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.HelpOutline
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import com.example.medicalaiguidance.R
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.service.MyAccessibilityService
import com.example.medicalaiguidance.viewmodel.HomeViewModel
import androidx.compose.ui.draw.alpha
@Composable
fun HomeScreen(
    navController: NavHostController,
    homeViewModel: HomeViewModel = viewModel(),
    onCancelRegistrationClick: () -> Unit = {}
) {
    val context = LocalContext.current
    val recentHistory by homeViewModel.recentHistory.collectAsState()
    var showPermissionDialog by remember { mutableStateOf(false) }
    var showCancelDialog by remember { mutableStateOf(false) }

    val textDarkColor = Color(0xFF1A2E2E)
    val primaryDark = Color(0xFF376F72)

    val buttonGradient = Brush.verticalGradient(
        colors = listOf(Color(0xFF6E9999), Color(0xFF036A6D))
    )
    val circleDarkColor = Color(0xFFC2D9D5).copy(alpha = 0.6f)

    fun isServiceEnabled(): Boolean {
        val expectedService = "${context.packageName}/${MyAccessibilityService::class.java.name}"
        val enabled = try {
            Settings.Secure.getInt(context.contentResolver, Settings.Secure.ACCESSIBILITY_ENABLED)
        } catch (e: Exception) {
            0
        }

        if (enabled == 1) {
            val enabledServices = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
            )
            return enabledServices?.contains(expectedService) == true
        }
        return false
    }

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
                    text = "為協助您取消掛號，智慧視覺導引需要使用協助工具權限。開啟後，系統會在台北榮總 App 中以紅框提示下一步操作的位置。",
                    fontSize = 15.sp,
                    lineHeight = 22.sp,
                    color = Color(0xFF556666)
                )
            },
            confirmButton = {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .clip(RoundedCornerShape(24.dp))
                            .clickable { showPermissionDialog = false },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "返回",
                            color = Color(0xFF7A8B8B),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold
                        )
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

    if (showCancelDialog) {
        AlertDialog(
            onDismissRequest = { showCancelDialog = false },
            shape = RoundedCornerShape(24.dp),
            containerColor = Color.White,
            title = {
                Text(
                    text = "取消掛號導引",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF1A2E2E)
                )
            },
            text = {
                Text(
                    text = "即將前往台北榮總 App 並啟動智慧視覺導引，確定要繼續嗎？",
                    fontSize = 15.sp,
                    lineHeight = 22.sp,
                    color = Color(0xFF556666)
                )
            },
            confirmButton = {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .clip(RoundedCornerShape(24.dp))
                            .clickable { showCancelDialog = false },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "返回", color = Color(0xFF7A8B8B), fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .background(color = primaryDark, RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable {
                                showCancelDialog = false
                                onCancelRegistrationClick()
                                MyAccessibilityService.startCancellationGuidance()

                                val packageName = "tw.com.bicom.VGHTPE"
                                val launchIntent = context.packageManager.getLaunchIntentForPackage(packageName)
                                if (launchIntent != null) {
                                    context.startActivity(launchIntent)
                                } else {
                                    val webIntent = Intent(
                                        Intent.ACTION_VIEW,
                                        Uri.parse("https://www.vghtpe.gov.tw/Index.action")
                                    )
                                    context.startActivity(webIntent)
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(text = "確定前往", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }
            },
            dismissButton = null
        )
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // --- 1. 背景層 (Canvas) ---
        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height

            val baseBackgroundGradient = Brush.linearGradient(
                colors = listOf(
                    Color(0xFFE8F2F0),
                    Color(0xFFFBFCFC),
                    Color(0xFFE4EFED)
                ),
                start = Offset(0f, 0f),
                end = Offset(canvasWidth, canvasHeight)
            )
            drawRect(brush = baseBackgroundGradient)

            val topCircleRadius = canvasWidth * 0.85f
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(circleDarkColor, Color.Transparent),
                    center = Offset(canvasWidth, 0f),
                    radius = topCircleRadius
                ),
                radius = topCircleRadius,
                center = Offset(canvasWidth, 0f)
            )

            val bottomCircleRadius = canvasWidth * 0.85f
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(circleDarkColor, Color.Transparent),
                    center = Offset(0f, canvasHeight), // 圓心左下角
                    radius = bottomCircleRadius
                ),
                radius = bottomCircleRadius,
                center = Offset(0f, canvasHeight)
            )
        }

        // --- 2. 內容層 (LazyColumn) ---
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding(),
            contentPadding = PaddingValues(start = 28.dp, end = 28.dp, bottom = 24.dp)
        ) {
            item {
                Spacer(modifier = Modifier.height(24.dp))

                // 問候與大按鈕用 Box 包起來，跨元件定位與疊加
                Box(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.fillMaxWidth()) {

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 12.dp, bottom = 12.dp)
                        ) {
                            Column {
                                Text(
                                    text = "Hello",
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF4A7373)
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    text = "今天有什麼可以幫您？",
                                    fontSize = 24.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = textDarkColor
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(15.dp))

                        // 2. 醫療指引大按鈕
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp)
                                .shadow(
                                    elevation = 12.dp,
                                    shape = RoundedCornerShape(50.dp),
                                    clip = false,
                                    ambientColor = Color(0xFF385E5E).copy(alpha = 0.3f),
                                    spotColor = Color(0xFF385E5E).copy(alpha = 0.5f)
                                )
                                .background(brush = buttonGradient, shape = RoundedCornerShape(50.dp))
                                .clickable {
                                    navController.navigate(Route.VISIT_TYPE_SELECTION)
                                },
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.Center
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(80.dp)
                                        .background(Color.White.copy(alpha = 0.15f), shape = CircleShape),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.MedicalServices,
                                        contentDescription = null,
                                        tint = Color.White,
                                        modifier = Modifier.size(38.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.height(20.dp))
                                Text(
                                    text = "開始醫療指引",
                                    fontSize = 26.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color.White,
                                    letterSpacing = 2.sp
                                )
                            }
                        }
                    }

                    Image(
                        painter = painterResource(id = R.drawable.sheep),
                        contentDescription = null,
                        modifier = Modifier
                            .size(135.dp)
                            .align(Alignment.TopEnd)
                            .offset(x = (20).dp, y = (1).dp),
                        contentScale = ContentScale.Fit
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    QuickActionButton(
                        title = "使用說明",
                        subtitle = "就醫變得更簡單",
                        icon = Icons.Outlined.HelpOutline,
                        modifier = Modifier.weight(1f),
                        onClick = {
                            navController.navigate(Route.GETTING_STARTED)
                        }
                    )

                    QuickActionButton(
                        title = "取消掛號",
                        subtitle = "帶您一步步取消",
                        icon = Icons.Outlined.Close,
                        modifier = Modifier.weight(1f),
                        onClick = {
                            if (isServiceEnabled()) {
                                showCancelDialog = true
                            } else {
                                showPermissionDialog = true
                            }
                        }
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Spacer(modifier = Modifier.height(40.dp))

                // 標題與「查看全部」
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "近期紀錄",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = textDarkColor
                    )

                    Box(
                        modifier = Modifier
                            .border(
                                width = 1.dp,
                                color = Color(0xFF7A8B8B).copy(alpha = 0.5f),
                                shape = RoundedCornerShape(20.dp)
                            )
                            .background(Color.White, shape = RoundedCornerShape(20.dp))
                            .clickable {
                                navController.navigate(Route.HISTORY)
                            }
                            .padding(horizontal = 14.dp, vertical = 6.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "查看全部",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color(0xFF4A7373)
                        )
                    }
                }
                Spacer(modifier = Modifier.height(18.dp))
            }

            // 近期紀錄：有資料就渲染清單，沒資料就顯示空狀態
            if (recentHistory.isEmpty()) {
                item {
                    EmptyHistoryState()
                }
            } else {
                items(
                    items = recentHistory,
                    key = { it.id }
                ) { history ->
                    RecordItem(
                        date = history.date,
                        summary = if (history.status == HistoryStatus.UNCOMPLETED) "症狀評估中" else history.typeTitle,
                        icon = if (history.status == HistoryStatus.COMPLETED) Icons.Filled.MedicalServices else Icons.Default.Psychology,
                        status = history.status,
                        onClick = { navController.navigate(Route.chatHistory(history.id)) }
                    )
                    Spacer(modifier = Modifier.height(14.dp))
                }
            }
        }
    }
}

@Composable
fun QuickActionButton(
    title: String,
    subtitle: String,
    icon: ImageVector,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val themeColor = Color(0xFF336063)
    Surface(
        modifier = modifier
            .shadow(
                elevation = 8.dp,
                shape = RoundedCornerShape(24.dp),
                ambientColor = Color.Black.copy(alpha = 0.05f),
                spotColor = Color(0xFF385E5E).copy(alpha = 0.3f)
            )
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(24.dp),
        color = Color.White
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .defaultMinSize(minHeight = 88.dp)
                .padding(horizontal = 14.dp, vertical = 20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = themeColor,
                modifier = Modifier.size(34.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = title,
                        fontSize = 17.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF1F2B2E),
                        letterSpacing = 0.8.sp,
                        maxLines = 1
                    )
                    Spacer(modifier = Modifier.width(3.dp))
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                        contentDescription = null,
                        tint = Color(0xFF8C9B9D),
                        modifier = Modifier.size(18.dp)
                    )
                }
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = subtitle,
                    fontSize = 12.sp,
                    color = Color(0xFF526365),
                    lineHeight = 16.sp,
                    letterSpacing = 0.4.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun EmptyHistoryState() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxHeight()
            .height(280.dp)
            .padding(horizontal = 16.dp, vertical = 24.dp) // 避免在平板上直接貼齊邊緣
            .border(
                width = 2.dp,
                color = Color(0xFFD9EAE7),
                shape = RoundedCornerShape(36.dp)
            ),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Image(
                painter = painterResource(id = R.drawable.sheep_3),
                contentDescription = null,
                modifier = Modifier.size(120.dp)
                    .alpha(0.5f),
                contentScale = ContentScale.Fit
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "暫無紀錄",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Color(0xFF7A8B8B)
            )
        }
    }
}

@Composable
fun RecordItem(
    date: String,
    summary: String,
    icon: ImageVector,
    status: HistoryStatus,
    onClick: () -> Unit
) {
    val isUncompleted = status == HistoryStatus.UNCOMPLETED

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(
                elevation = 4.dp,
                shape = RoundedCornerShape(45.dp),
                ambientColor = Color.Black.copy(alpha = 0.05f),
                spotColor = Color.Black.copy(alpha = 0.05f)
            )
            .background(Color.White, shape = RoundedCornerShape(45.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 28.dp, vertical = 28.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // 圖標圓圈
        Box(
            modifier = Modifier
                .size(52.dp)
                .background(Color(0xFFF0F5F5), shape = CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = Color(0xFF036A6D),
                modifier = Modifier.size(26.dp)
            )
        }

        Spacer(modifier = Modifier.width(16.dp))

        // 文字資訊
        Column(
            modifier = Modifier.weight(1f)
        ) {
            Text(
                text = date,
                fontSize = 14.sp,
                color = Color(0xFF7A8B8B),
                fontWeight = FontWeight.Medium
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = summary,
                fontSize = 18.sp,
                color = Color(0xFF1A2E2E),
                fontWeight = FontWeight.Bold
            )
        }

        // 狀態 Badge (未完成/已完成)
        Surface(
            color = if (isUncompleted) Color(0xFFFDF6EC) else Color(0xFFD1E9E3),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text(
                text = if (isUncompleted) "未完成" else "已完成",
                color = if (isUncompleted) Color(0xFFE6A23C) else Color(0xFF385E5E),
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
            )
        }
    }
}
