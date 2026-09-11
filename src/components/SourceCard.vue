<script setup>
// 每张卡片展示一个被回答实际引用的历史案件及其检索资料。
defineProps({
  source: { type: Object, required: true },
})
</script>

<template>
  <article class="source-card">
    <div class="source-heading">
      <span class="source-report">案件 {{ source.case_id }}</span>
      <span class="source-page">检索分数 {{ source.score.toFixed(4) }}</span>
    </div>

    <div class="source-tags">
      <span>{{ source.charge || '罪名未提供' }}</span>
      <span>法条 {{ source.article || '未提供' }}</span>
    </div>

    <p class="source-fact">{{ source.fact }}</p>

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
