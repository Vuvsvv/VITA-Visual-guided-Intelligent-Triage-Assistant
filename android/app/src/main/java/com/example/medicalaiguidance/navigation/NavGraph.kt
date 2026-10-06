package com.example.medicalaiguidance.navigation

import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.example.medicalaiguidance.screen.ChatScreen
import com.example.medicalaiguidance.screen.ConfirmNeedScreen
import com.example.medicalaiguidance.screen.DoctorSelectionScreen
import com.example.medicalaiguidance.screen.HistoryScreen
import com.example.medicalaiguidance.screen.HomeScreen
import com.example.medicalaiguidance.screen.GettingStartedScreen
import com.example.medicalaiguidance.screen.QuickSearchScreen
import com.example.medicalaiguidance.screen.VisitTypeSelectionScreen
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.viewmodel.HistoryViewModel

object Route {
    const val HOME = "home"
    const val CHAT = "chat"
    const val CHAT_START = "chat/start/{visitPlan}"
    const val CHAT_HISTORY = "chat/history/{historyId}"
    const val VISIT_TYPE_SELECTION = "visit_type_selection"
    const val RETURN_VISIT = "return_visit"
    const val QUICK_SEARCH = "quick_search"
    const val SELECT_DOCTOR = "select_doctor/{visitPlan}"
    const val CONFIRM_NEED = "confirm_need"
    const val HISTORY = "history"
    const val GETTING_STARTED = "getting_started"

    fun chat(visitPlan: VisitPlan): String = "chat/start/${visitPlan.routeValue}"
    fun chatHistory(historyId: String): String = "chat/history/$historyId"
    fun selectDoctor(visitPlan: VisitPlan): String = "select_doctor/${visitPlan.routeValue}"
}

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Route.HOME
    ) {
        composable(Route.HOME) {
            HomeScreen(navController = navController)
        }

        composable(Route.GETTING_STARTED) {
            GettingStartedScreen(navController = navController)
        }

        composable(Route.VISIT_TYPE_SELECTION) {
            VisitTypeSelectionScreen(navController = navController)
        }

        composable(Route.CHAT) {
            ChatScreen(navController = navController, startNew = true, visitPlan = VisitPlan.UNKNOWN)
        }

        composable(
            route = Route.CHAT_START,
            arguments = listOf(navArgument("visitPlan") { type = NavType.StringType })
        ) { backStackEntry ->
            ChatScreen(
                navController = navController,
                startNew = true,
                visitPlan = VisitPlan.fromRoute(backStackEntry.arguments?.getString("visitPlan"))
            )
        }

        composable(
            route = Route.CHAT_HISTORY,
            arguments = listOf(navArgument("historyId") { type = NavType.StringType })
        ) { backStackEntry ->
            ChatScreen(
                navController = navController,
                historyId = backStackEntry.arguments?.getString("historyId")
            )
        }

        composable(
            route = Route.SELECT_DOCTOR,
            arguments = listOf(navArgument("visitPlan") { type = NavType.StringType })
        ) { backStackEntry ->
            DoctorSelectionScreen(
                navController = navController,
                visitPlan = VisitPlan.fromRoute(backStackEntry.arguments?.getString("visitPlan"))
            )
        }

        composable(Route.CONFIRM_NEED) {
            ConfirmNeedScreen(navController)
        }

        composable(Route.QUICK_SEARCH) {
            QuickSearchScreen(navController = navController)
        }

        // Legacy route alias retained so old navigation state does not crash.
        composable(Route.RETURN_VISIT) {
            QuickSearchScreen(navController = navController)
        }

        composable(Route.HISTORY) {
            val historyViewModel: HistoryViewModel = viewModel()
            HistoryScreen(navController = navController, viewModel = historyViewModel)
        }
    }
}
