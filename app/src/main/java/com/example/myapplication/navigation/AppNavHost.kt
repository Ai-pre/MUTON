package com.example.myapplication.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.myapplication.screens.CalendarScreen
import com.example.myapplication.screens.ChangePasswordScreen
import com.example.myapplication.screens.ChatRecordScreen
import com.example.myapplication.screens.HomeScreen
import com.example.myapplication.screens.LoginScreen
import com.example.myapplication.screens.MenuScreen
import com.example.myapplication.screens.ProfileScreen
import com.example.myapplication.screens.SignUpScreen
import com.example.myapplication.screens.SplashScreen

@Composable
fun AppNavHost(
    navController: NavHostController = rememberNavController()
) {
    NavHost(
        navController = navController,
        startDestination = NavRoutes.Splash
    ) {
        composable(NavRoutes.Splash) {
            SplashScreen(
                onFinished = {
                    navController.navigate(NavRoutes.Login) {
                        popUpTo(NavRoutes.Splash) { inclusive = true }
                    }
                }
            )
        }

        composable(NavRoutes.Login) {
            LoginScreen(
                onLoginClick = {
                    navController.navigate(NavRoutes.Home) {
                        popUpTo(NavRoutes.Login) { inclusive = true }
                    }
                },
                onSignUpClick = {
                    navController.navigate(NavRoutes.SignUp)
                }
            )
        }

        composable(NavRoutes.SignUp) {
            SignUpScreen(
                onCompleteClick = {
                    navController.navigate(NavRoutes.Home) {
                        popUpTo(NavRoutes.Login) { inclusive = true }
                    }
                },
                onBackClick = {
                    navController.popBackStack()
                }
            )
        }

        composable(NavRoutes.Home) {
            HomeScreen(
                onMenuClick = { navController.navigate(NavRoutes.Menu) },
                onRecordClick = { navController.navigate(NavRoutes.Calendar) },
                onChatClick = { navController.navigate(NavRoutes.ChatRecord) }
            )
        }

        composable(NavRoutes.Menu) {
            MenuScreen(
                onBackClick = { navController.popBackStack() },
                onProfileClick = { navController.navigate(NavRoutes.Profile) },
                onPasswordClick = { navController.navigate(NavRoutes.ChangePassword) }
            )
        }

        composable(NavRoutes.Calendar) {
            CalendarScreen(
                onBackClick = { navController.popBackStack() },
                onRecordClick = { navController.navigate(NavRoutes.ChatRecord) }
            )
        }

        composable(NavRoutes.ChatRecord) {
            ChatRecordScreen(
                onBackClick = { navController.popBackStack() }
            )
        }

        composable(NavRoutes.Profile) {
            ProfileScreen(
                onBackClick = { navController.popBackStack() },
                onChangePasswordClick = { navController.navigate(NavRoutes.ChangePassword) }
            )
        }

        composable(NavRoutes.ChangePassword) {
            ChangePasswordScreen(
                onBackClick = { navController.popBackStack() }
            )
        }
    }
}