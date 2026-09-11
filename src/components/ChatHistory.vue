<script setup>
import { useChatStore } from '../stores/chat'

// 历史列表和右侧聊天区共享同一个 Store，切换后无需再向 App 手动传递消息。
const chatStore = useChatStore()
</script>

<template>
  <section class="history-section" aria-labelledby="history-title">
    <h2 id="history-title" class="section-label">聊天历史</h2>
    <div class="history-list">
      <!-- 每一行分别提供会话切换按钮和删除按钮，避免两个按钮互相嵌套。 -->
      <div
        v-for="chat in chatStore.chatHistory"
        :key="chat.id"
        class="history-item"
        :class="{ active: chatStore.currentChatId === chat.id }"
      >
        <button class="history-select" type="button" @click="chatStore.switchChat(chat.id)">
          <span class="history-icon" aria-hidden="true">◇</span>
          <span>{{ chat.title }}</span>
        </button>
        <button
          class="history-delete"
          type="button"
          :aria-label="`删除聊天：${chat.title}`"
          title="删除聊天"
          @click="chatStore.deleteChat(chat.id)"
        >
          ×
        </button>
      </div>
    </div>
  </section>
</template>
