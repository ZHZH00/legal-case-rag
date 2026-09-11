<script setup>
import ChatHistory from './ChatHistory.vue'
import KnowledgePanel from './KnowledgePanel.vue'
import legalCaseLogo from '../assets/legal-case-logo.svg'
import { useChatStore } from '../stores/chat'

const chatStore = useChatStore()
</script>

<template>
  <!-- 侧栏纵向布局：品牌和按钮固定，历史区伸缩滚动，知识库保持在底部。 -->
  <aside class="sidebar">
    <div class="brand">
      <img class="brand-mark" :src="legalCaseLogo" alt="中文刑事类案检索 Logo" />
      <div>
        <h1>中文刑事类案检索</h1>
        <p>Legal Case RAG</p>
      </div>
    </div>

    <!-- 新建会话直接调用共享 Store，右侧 App 会同步观察到 messages 被清空。 -->
    <button class="new-chat-button" type="button" @click="chatStore.createChat">
      <span aria-hidden="true">＋</span>
      新建检索
    </button>

    <ChatHistory />
    <KnowledgePanel />
  </aside>
</template>
