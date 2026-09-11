<script setup>
// 接收单个引用案件。
defineProps({
  source: { type: Object, required: true },
})
</script>

<template>
  <article class="source-card">
    <!-- 展示案件标识与排序分数。 -->
    <div class="source-heading">
      <span class="source-report">案件 {{ source.case_id }}</span>
      <span class="source-page">检索分数 {{ source.score.toFixed(4) }}</span>
    </div>

    <div class="source-tags">
      <span>{{ source.charge || '罪名未提供' }}</span>
      <span>法条 {{ source.article || '未提供' }}</span>
    </div>

    <!-- 核心事实默认展开。 -->
    <p class="source-fact">{{ source.fact }}</p>

    <!-- 长说理与判决按需展开。 -->
    <details v-if="source.reason">
      <summary>查看法院说理</summary>
      <p>{{ source.reason }}</p>
    </details>
    <details v-if="source.result">
      <summary>查看判决结果</summary>
      <p>{{ source.result }}</p>
    </details>
  </article>
</template>
