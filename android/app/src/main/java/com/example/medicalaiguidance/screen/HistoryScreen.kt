package com.example.medicalaiguidance.screen

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MenuDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.viewmodel.HistoryUiState
import com.example.medicalaiguidance.viewmodel.HistoryViewModel

@Composable
fun HistoryScreen(
    navController: NavHostController,
    viewModel: HistoryViewModel
) {
    val uiState by viewModel.uiState.collectAsState()
    val selectedTab by viewModel.selectedTab.collectAsState()
    val tabs = listOf("全部", "已完成", "未完成")
    val primaryDark = Color(0xFF036A6D)
    val lightBg = Color(0xFFF2FAF8)

    Scaffold(
        containerColor = lightBg,
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { navController.navigate(Route.VISIT_TYPE_SELECTION) },
                containerColor = primaryDark,
                contentColor = Color.White,
                shape = RoundedCornerShape(20.dp),
                icon = { Icon(Icons.Default.Add, contentDescription = null) },
                text = { Text("新增問診", fontWeight = FontWeight.Bold, fontSize = 16.sp) },
                modifier = Modifier.padding(bottom = 16.dp, end = 8.dp)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(lightBg)
        ) {
            Header(navController = navController, primaryDark = primaryDark)
            TabSelector(
                tabs = tabs,
                selectedTab = selectedTab,
                primaryDark = primaryDark,
                onSelect = viewModel::onTabSelected
            )
            Spacer(modifier = Modifier.height(12.dp))

            when (val state = uiState) {
                HistoryUiState.Loading -> {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = primaryDark)
                    }
                }

                is HistoryUiState.Success -> {
                    if (state.filteredHistory.isEmpty()) {
                        EmptyHistory()
                    } else {
                        LazyColumn(
                            modifier = Modifier.fillMaxSize(),
                            contentPadding = PaddingValues(start = 24.dp, end = 24.dp, bottom = 100.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            items(state.filteredHistory, key = { it.id }) { history ->
                                HistoryCardItem(
                                    history = history,
                                    onActionClick = {
                                        navController.navigate(Route.chatHistory(history.id))
                                    },
                                    onDeleteClick = {
                                        viewModel.deleteHistory(history.id)
                                    }
                                )
                            }
                        }
                    }
                }

                is HistoryUiState.Error -> {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text(state.message, color = Color.Red)
                    }
                }
            }
        }
    }
}

@Composable
private fun Header(navController: NavHostController, primaryDark: Color) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 20.dp),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.CenterStart)
                .size(48.dp)
                .shadow(elevation = 4.dp, shape = RoundedCornerShape(16.dp))
                .background(Color.White, shape = RoundedCornerShape(16.dp))
                .clickable { navController.popBackStack() },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "返回",
                tint = primaryDark,
                modifier = Modifier.size(22.dp)
            )
        }

        Text(
            text = "歷史紀錄",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = primaryDark
        )
    }
}

@Composable
private fun TabSelector(
    tabs: List<String>,
    selectedTab: Int,
    primaryDark: Color,
    onSelect: (Int) -> Unit
) {
    Box(
        modifier = Modifier
            .padding(horizontal = 24.dp, vertical = 8.dp)
            .fillMaxWidth()
            .height(54.dp)
            .background(Color(0xFFEBEFEF), shape = RoundedCornerShape(28.dp))
            .padding(4.dp)
    ) {
        Row(modifier = Modifier.fillMaxSize()) {
            tabs.forEachIndexed { index, title ->
                val isSelected = selectedTab == index
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .background(
                            color = if (isSelected) primaryDark else Color.Transparent,
                            shape = RoundedCornerShape(24.dp)
                        )
                        .clickable { onSelect(index) },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = title,
                        color = if (isSelected) Color.White else Color(0xFF556666),
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyHistory() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(bottom = 100.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Default.Description,
            contentDescription = null,
            modifier = Modifier.size(80.dp),
            tint = Color(0xFFC0CCCC)
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "尚無歷史紀錄",
            fontSize = 16.sp,
            color = Color(0xFFB0C4C4),
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun HistoryCardItem(
    history: History,
    onActionClick: () -> Unit,
    onDeleteClick: () -> Unit
) {
    val isCompleted = history.status == HistoryStatus.COMPLETED
    val primaryDark = Color(0xFF036A6D)
    // 控制下拉選單與確認彈窗State
    var showMenu by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }

    // 1. 動態計算卡片背景色：選單開啟時變暗灰色 (0xFFE2EDED)，關閉時恢復白色
    val cardBgColor by animateColorAsState(
        targetValue = if (showMenu) Color(0xFFEBEFEF) else Color.White,
        animationSpec = tween(durationMillis = 200), // 200ms 的平滑過渡
        label = "cardBgColorAnimation"
    )
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(28.dp),
        colors = CardDefaults.cardColors(containerColor = cardBgColor),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(history.date, fontSize = 14.sp, color = Color(0xFF7A8B8B))

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Surface(
                        color = if (isCompleted) Color(0xFFD1E9E3) else Color(0xFFFDF6EC),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = if (isCompleted) "已完成" else "未完成",
                            color = if (isCompleted) primaryDark else Color(0xFFE6A23C),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    // 三點點 Menu 區域
                    Box {
                        IconButton(
                            onClick = { showMenu = true },
                            modifier = Modifier.size(34.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.MoreVert,
                                contentDescription = "更多選項",
                                tint = Color(0xFF7A8B8B),
                                modifier = Modifier.size(20.dp)
                            )
                        }

                        DropdownMenu(
                            expanded = showMenu,
                            onDismissRequest = { showMenu = false },
                            containerColor = Color.White,
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            // 選項一：刪除對話 (警告紅)
                            DropdownMenuItem(
                                text = {
                                    Text("刪除對話", fontWeight = FontWeight.Bold)
                                },
                                leadingIcon = {
                                    Icon(
                                        imageVector = Icons.Default.Delete,
                                        contentDescription = null
                                    )
                                },
                                colors = MenuDefaults.itemColors(
                                    textColor = Color(0xFFB44A4A),
                                    leadingIconColor = Color(0xFFB44A4A)
                                ),
                                onClick = {
                                    showMenu = false
                                    showDeleteDialog = true
                                }
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .background(Color(0xFFF0F5F5), shape = CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isCompleted) Icons.Default.MedicalServices else Icons.Default.Psychology,
                        contentDescription = null,
                        tint = primaryDark,
                        modifier = Modifier.size(24.dp)
                    )
                }

                Spacer(modifier = Modifier.width(16.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (isCompleted) history.typeTitle else "症狀評估中",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF1A2E2E),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        history.summaryText,
                        fontSize = 15.sp,
                        color = Color(0xFF556666),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            HorizontalDivider(color = Color(0xFFF0F4F4), thickness = 1.dp)
            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    modifier = Modifier.clickable { onActionClick() },
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (isCompleted) "查看" else "繼續",
                        color = primaryDark,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Icon(
                        Icons.Default.ChevronRight,
                        contentDescription = null,
                        tint = primaryDark,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
    }
    // 刪除確認彈窗
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            shape = RoundedCornerShape(24.dp),
            containerColor = Color.White,
            title = {
                Text(
                    text = "確認刪除紀錄",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF1A2E2E)
                )
            },
            text = {
                Text(
                    text = "刪除後該筆紀錄將無法恢復，確定要刪除嗎？",
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
                    // 左按鈕：取消（灰色中性按鈕）
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            //.background(Color(0xFFEBEFEF), RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable { showDeleteDialog = false },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "取消",
                            color = Color(0xFF7A8B8B),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }

                    // 右按鈕：確認刪除（警示紅）
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .background(color = Color(0xFFB44A4A), shape = RoundedCornerShape(24.dp))
                            .clip(RoundedCornerShape(24.dp))
                            .clickable {
                                showDeleteDialog = false
                                onDeleteClick()
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "刪除",
                            color = Color.White,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            },
            dismissButton = null
        )
    }
}

