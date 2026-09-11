<script setup>
import SourceCard from './SourceCard.vue'

// 接收并展示 chatStore 中的一条用户消息或 AI 消息。
defineProps({
  message: { type: Object, required: true },
})
</script>

<template>
  <!-- role 生成不同样式类，决定用户消息靠右、AI 消息靠左。 -->
  <article class="message-row" :class="`message-${message.role}`">
    <div class="message-inner">
      <div class="message-meta">
        <span class="message-avatar">{{ message.role === 'user' ? '你' : 'AI' }}</span>
        <span>{{ message.role === 'user' ? '你的案情' : '类案助手' }}</span>
      </div>
      <div class="message-content">
        {{ message.content }}
      </div>

      <!-- 只有带有检索来源的 AI 消息才显示来源区，多条来源通过 v-for 逐个传给 SourceCard。 -->
      <section
        v-if="message.role === 'assistant' && message.sources?.length"
        class="message-sources"
      >
        <h3>回答引用的相似案件</h3>
        <div class="source-list">
          <SourceCard
            v-for="source in message.sources"
            :key="source.case_id"
            :source="source"
          />
        </div>
      </section>
    </div>
  </article>
</template>
