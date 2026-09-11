<script setup>
import Sidebar from './components/Sidebar.vue'
import ChatHeader from './components/ChatHeader.vue'
import WelcomePanel from './components/WelcomePanel.vue'
import ChatMessage from './components/ChatMessage.vue'
import ChatInput from './components/ChatInput.vue'
import { useChatStore } from './stores/chat'

// 根组件读取共享聊天状态。
const chatStore = useChatStore()
</script>

<template>
  <div class="app-shell">
    <!-- 左侧导航与历史记录。 -->
    <Sidebar />

    <main class="chat-workspace">
      <!-- 顶部系统状态。 -->
      <ChatHeader />

      <!-- 空会话显示欢迎页，其余显示消息流。 -->
      <section class="conversation-area">
        <WelcomePanel v-if="chatStore.messages.length === 0" />

        <div v-else class="message-list">
          <ChatMessage v-for="message in chatStore.messages" :key="message.id" :message="message" />
        </div>
      </section>

      <!-- 底部固定输入区。 -->
      <ChatInput />
    </main>
  </div>
</template>
