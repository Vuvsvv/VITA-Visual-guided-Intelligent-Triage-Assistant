package com.example.medicalaiguidance.model

enum class MessageSender { USER, AI }

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val content: String,
    val sender: MessageSender,
    val timestamp: Long = System.currentTimeMillis()
)