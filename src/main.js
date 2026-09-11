import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { useChatStore } from './stores/chat'
import './assets/main.css'

// 创建Vue应用与Pinia实例。
const app = createApp(App)
const pinia = createPinia()

// 挂载前注册全局状态。
app.use(pinia)

// 状态变更后持久化聊天记录。
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

// 挂载根组件。
app.mount('#app')
