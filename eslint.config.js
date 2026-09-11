import js from '@eslint/js'
import globals from 'globals'
import pluginVue from 'eslint-plugin-vue'

// ESLint 使用 flat config：先加载 JavaScript 与 Vue 推荐规则，再补充项目自己的约定。
export default [
  // 构建产物和第三方依赖不属于本项目源码，不参与静态检查。
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    files: ['**/*.{js,vue}'],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    rules: {
      // 当前组件名包含 App、Sidebar 等单词，不强制使用多单词组件名。
      'vue/multi-word-component-names': 'off',
      // 与 Prettier 的模板排版保持一致，避免两套工具对换行规则产生冲突。
      'vue/max-attributes-per-line': 'off',
      'vue/singleline-html-element-content-newline': 'off',
      'vue/html-self-closing': 'off',
    },
  },
]
