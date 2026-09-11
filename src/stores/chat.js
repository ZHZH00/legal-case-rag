import { defineStore } from 'pinia'
import { sendQuestion } from '../api/chat'

const CHAT_STORAGE_KEY = 'legal-case-chat'
// 从localStorage恢复会话快照。
const savedChat = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) || 'null')
const defaultChatHistory = [{ id: 1, title: '当前聊天', messages: [] }]
const chatHistory = savedChat?.chatHistory || defaultChatHistory
const currentChatId = savedChat?.currentChatId || chatHistory[0].id
const currentChat = chatHistory.find((chat) => chat.id === currentChatId) || chatHistory[0]

export const useChatStore = defineStore('chat', {
  // 统一维护会话列表与当前消息。
  state: () => ({
    messages: savedChat?.messages || [...currentChat.messages],
    chatHistory,
    currentChatId,
    loading: false,
  }),

  actions: {
    // 新建并切换到空会话。
    createChat() {
      const id = Date.now()
      this.chatHistory.unshift({ id, title: '新对话', messages: [] })
      this.currentChatId = id
      this.messages = []
    },

    // 切换并恢复目标会话。
    switchChat(chatId) {
      const chat = this.chatHistory.find((item) => item.id === chatId)
      if (!chat) return
      this.currentChatId = chatId
      this.messages = [...chat.messages]
    },

    // 删除会话并选择相邻会话。
    deleteChat(chatId) {
      const deleteIndex = this.chatHistory.findIndex((item) => item.id === chatId)
      if (deleteIndex === -1) return

      const isCurrentChat = this.currentChatId === chatId
      this.chatHistory.splice(deleteIndex, 1)

      // 非当前会话无需刷新消息区。
      if (!isCurrentChat) return

      // 优先选择删除位置后的相邻会话。
      const nextChat = this.chatHistory[deleteIndex] || this.chatHistory[deleteIndex - 1]
      if (nextChat) {
        this.currentChatId = nextChat.id
        this.messages = [...nextChat.messages]
      } else {
        this.createChat()
      }
    },

    // 发送问题并消费流式回答。
    async sendMessage(content) {
      const question = content.trim()
      if (!question || this.loading) return

      const chatId = this.currentChatId
      const chat = this.chatHistory.find((item) => item.id === chatId)
      if (!chat) return

      // 保存用户消息并生成会话标题。
      this.messages.push({ id: Date.now(), role: 'user', content: question })
      if (chat.title === '当前聊天' || chat.title === '新对话') {
        chat.title = question.slice(0, 16)
      }
      chat.messages = [...this.messages]

      // 创建可持续更新的助手消息。
      chat.messages.push({
        id: Date.now() + 1,
        role: 'assistant',
        content: '正在思考……',
        sources: [],
      })
      const assistantMessage = chat.messages[chat.messages.length - 1]
      if (this.currentChatId === chatId) {
        this.messages = [...chat.messages]
      }

      // token到达前允许状态文本覆盖占位内容。
      this.loading = true
      let hasReceivedToken = false

      try {
        // 当前会话ID直接用作thread_id。
        await sendQuestion(question, chatId, (event) => {
          // 正文开始前显示后端阶段状态。
          if (event.type === 'status' && !hasReceivedToken) {
            assistantMessage.content = event.message
          }

          if (event.type === 'token') {
            // 首个token清空状态文本。
            if (!hasReceivedToken) {
              assistantMessage.content = ''
              hasReceivedToken = true
            }
            assistantMessage.content += event.content
          }

          // 来源事件写入当前助手消息。
          if (event.type === 'sources') {
            assistantMessage.sources = event.sources || []
          }

          // 仅同步当前仍可见的会话。
          if (this.currentChatId === chatId) {
            this.messages = [...chat.messages]
          }
        })
      } catch (error) {
        console.error(error)
        // 保留已生成正文并追加中断提示。
        assistantMessage.content = hasReceivedToken
          ? `${assistantMessage.content}\n\n回答生成中断，请稍后重试。`
          : '无法连接类案检索后端，请确认 FastAPI 已启动。'

        if (this.currentChatId === chatId) {
          this.messages = [...chat.messages]
        }
      } finally {
        // 请求结束后恢复输入状态。
        this.loading = false
      }
    },
  },
})
