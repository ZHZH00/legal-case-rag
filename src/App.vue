<script setup>
import Sidebar from './components/Sidebar.vue'
import ChatHeader from './components/ChatHeader.vue'
import WelcomePanel from './components/WelcomePanel.vue'
import ChatMessage from './components/ChatMessage.vue'
import ChatInput from './components/ChatInput.vue'
import { useChatStore } from './stores/chat'

// App 只负责组合页面，并根据当前消息决定显示欢迎页还是聊天内容。
const chatStore = useChatStore()
</script>

<template>
  <div class="app-shell">
    <Sidebar />

    <main class="chat-workspace">
      <ChatHeader />

      <section class="conversation-area">
        <WelcomePanel v-if="chatStore.messages.length === 0" />

        <div v-else class="message-list">
          <ChatMessage v-for="message in chatStore.messages" :key="message.id" :message="message" />
        </div>
      </section>

      <ChatInput />
    </main>
  </div>
</template>
