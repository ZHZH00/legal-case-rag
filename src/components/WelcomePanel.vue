<script setup>
// 欢迎页使用本地示例数据。
const examples = [
  { number: '01', text: '驾驶机动车发生事故，造成一人死亡，负主要责任并主动报警。', tag: '交通肇事' },
  { number: '02', text: '在商场盗窃价值三千元商品，系初犯，到案后如实供述。', tag: '盗窃案件' },
  { number: '03', text: '与他人争执后持刀伤人，造成被害人轻伤二级。', tag: '故意伤害' },
  { number: '04', text: '明知朋友饮酒仍将车辆交给其驾驶，途中发生交通事故。', tag: '争议情节' },
]
</script>

<template>
  <!-- 空会话显示欢迎说明。 -->
  <div class="welcome-panel">
    <div class="welcome-kicker"><span aria-hidden="true" /> LEGAL CASE RETRIEVAL</div>
    <h2>你好，我是中文刑事<br />类案检索助手</h2>
    <p class="welcome-description">
      请描述案件事实，我会检索相似的历史案件，<br />并对比案件事实、法院说理和判决结果。
    </p>

    <!-- 示例卡片仅展示支持场景。 -->
    <div class="example-grid">
      <div v-for="example in examples" :key="example.number" class="example-card">
        <span class="example-number">{{ example.number }}</span>
        <span class="example-text">{{ example.text }}</span>
        <span class="example-tag">{{ example.tag }}</span>
      </div>
    </div>
  </div>
</template>
