package com.example.medicalaiguidance.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.EventRepeat
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.navigation.Route
import androidx.compose.material.icons.filled.PersonSearch
@Composable
fun VisitTypeSelectionScreen(navController: NavHostController) {
    val primary = Color(0xFF2F6F73)
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF2FAF8))
            .statusBarsPadding()
            .padding(horizontal = 24.dp, vertical = 20.dp)
    ) {
        Box(
            modifier = Modifier.fillMaxWidth(),
            contentAlignment = Alignment.Center
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.CenterStart)
                    .size(48.dp)
                    .shadow(elevation = 4.dp, shape = RoundedCornerShape(16.dp))
                    .background(Color.White, RoundedCornerShape(16.dp))
                    .clickable { navController.popBackStack() },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "返回",
                    tint = primary,
                    modifier = Modifier.size(22.dp)
                )
            }
            Text(
                text = "選擇就診類型",
                color = primary,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.height(42.dp))
        Text("請問您想如何安排就醫？", color = Color(0xFF1A2E2E), fontSize = 20.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(12.dp))
        Text("選擇初/複診進行問診，或直接查詢門診班表", color = Color(0xFF607575), fontSize = 16.sp,fontWeight = FontWeight.Medium)
        Spacer(modifier = Modifier.height(34.dp))

        Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
            VisitTypeCard(
                title = "初診",
                description = "第一次來，從沒在本院看過病。",
                icon = Icons.Default.MedicalServices,
                primary = primary,
                onClick = { navController.navigate(Route.chat(VisitPlan.INITIAL)) }
            )
            VisitTypeCard(
                title = "複診",
                description = "曾經來看過，在本院有過病歷。",
                icon = Icons.Default.EventRepeat,
                primary = primary,
                onClick = { navController.navigate(Route.chat(VisitPlan.FOLLOW_UP)) }
            )
            VisitTypeCard(
                title = "快速查詢",
                description = "已知道就診科別，直接依條件尋找班表。",
                icon = Icons.Default.PersonSearch,
                primary = primary,
                onClick = { navController.navigate(Route.QUICK_SEARCH) }
            )
        }
    }
}

@Composable
private fun VisitTypeCard(
    title: String,
    description: String,
    icon: ImageVector,
    primary: Color,
    onClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color.White, RoundedCornerShape(22.dp))
            .clickable(onClick = onClick)
            .padding(20.dp)
    ) {
        Icon(icon, contentDescription = null, tint = primary, modifier = Modifier.size(34.dp))
        Spacer(modifier = Modifier.height(12.dp))
        Text(title, color = primary, fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(6.dp))
        Text(description, color = Color(0xFF607575), fontSize = 15.sp, lineHeight = 21.sp)
    }
}
