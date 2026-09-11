<script setup>
import { useChatStore } from '../stores/chat'

// 读取共享会话状态。
const chatStore = useChatStore()
</script>

<template>
  <section class="history-section" aria-labelledby="history-title">
    <h2 id="history-title" class="section-label">聊天历史</h2>
    <div class="history-list">
      <!-- 每项包含切换与删除操作。 -->
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
