<script setup>
import SourceCard from './SourceCard.vue'

// 接收单条聊天消息。
defineProps({
  message: { type: Object, required: true },
})
</script>

<template>
  <!-- 按角色应用消息布局。 -->
  <article class="message-row" :class="`message-${message.role}`">
    <div class="message-inner">
      <div class="message-meta">
        <span class="message-avatar">{{ message.role === 'user' ? '你' : 'AI' }}</span>
        <span>{{ message.role === 'user' ? '你的案情' : '类案助手' }}</span>
      </div>
      <div class="message-content">
        {{ message.content }}
      </div>

      <!-- 仅为带来源的助手消息渲染案件卡片。 -->
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
