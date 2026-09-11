<script setup>
import ChatHistory from './ChatHistory.vue'
import KnowledgePanel from './KnowledgePanel.vue'
import legalCaseLogo from '../assets/legal-case-logo.svg'
import { useChatStore } from '../stores/chat'

const chatStore = useChatStore()
</script>

<template>
  <!-- 侧栏包含品牌、会话和知识库状态。 -->
  <aside class="sidebar">
    <div class="brand">
      <img class="brand-mark" :src="legalCaseLogo" alt="中文刑事类案检索 Logo" />
      <div>
        <h1>中文刑事类案检索</h1>
        <p>Legal Case RAG</p>
      </div>
    </div>

    <!-- 新建并切换到空会话。 -->
    <button class="new-chat-button" type="button" @click="chatStore.createChat">
      <span aria-hidden="true">＋</span>
      新建检索
    </button>

    <ChatHistory />
    <KnowledgePanel />
  </aside>
</template>
