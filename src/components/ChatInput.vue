<script setup>
import { ref } from 'vue'
import { useChatStore } from '../stores/chat'

const chatStore = useChatStore()
const question = ref('')

// 发送按钮和 Enter 都调用这个函数：交给 Store 请求后端，然后清空输入框。
function submit() {
  if (!question.value.trim() || chatStore.loading) return
  chatStore.sendMessage(question.value)
  question.value = ''
}

// Enter 提交，Shift+Enter 保留原生换行行为。
function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <footer class="input-area">
    <div class="input-box">
      <textarea
        v-model="question"
        rows="2"
        placeholder="请描述案件事实，例如行为、金额、后果、自首或谅解等情节……"
        aria-label="输入案件事实"
        @keydown="handleKeydown"
      />
      <button
        type="button"
        class="send-button"
        :disabled="!question.trim() || chatStore.loading"
        @click="submit"
      >
        <span>{{ chatStore.loading ? '发送中' : '发送' }}</span>
        <b aria-hidden="true">↑</b>
      </button>
    </div>
    <p>结果来自历史案件检索，仅供学习和研究参考，不构成法律意见。</p>
  </footer>
</template>
