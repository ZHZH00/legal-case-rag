import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { useChatStore } from './stores/chat'
import './assets/main.css'

// 应用只创建一次：App 是根组件，其他页面组件都从 App 的组件树向下展开。
const app = createApp(App)
const pinia = createPinia()

// 在 mount 之前注册 Pinia，使组件树中的任意组件都能访问聊天和知识库 Store。
app.use(pinia)

// Pinia 状态变化时统一保存聊天记录，页面重启后由聊天 Store 自动读取。
const chatStore = useChatStore(pinia)
chatStore.$subscribe((_mutation, state) => {
  localStorage.setItem(
    'legal-case-chat',
    JSON.stringify({
      ...state,
      loading: false,
    }),
  )
})

// 将整个 Vue 组件树挂载到 index.html 的 <div id="app">。
app.mount('#app')
