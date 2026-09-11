import { defineStore } from 'pinia'
import { sendQuestion } from '../api/chat'

const CHAT_STORAGE_KEY = 'legal-case-chat'
const savedChat = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) || 'null')
const defaultChatHistory = [{ id: 1, title: '当前聊天', messages: [] }]
const chatHistory = savedChat?.chatHistory || defaultChatHistory
const currentChatId = savedChat?.currentChatId || chatHistory[0].id
const currentChat = chatHistory.find((chat) => chat.id === currentChatId) || chatHistory[0]

export const useChatStore = defineStore('chat', {
  // 当前消息和会话列表由 Pinia 统一保存，侧栏与聊天区读取的是同一份状态。
  state: () => ({
    messages: savedChat?.messages || [...currentChat.messages],
    chatHistory,
    currentChatId,
    loading: false,
  }),

  actions: {
    // 新建聊天时保存一个空会话，并清空右侧当前消息。
    createChat() {
      const id = Date.now()
      this.chatHistory.unshift({ id, title: '新对话', messages: [] })
      this.currentChatId = id
      this.messages = []
    },

    // 点击历史记录时，恢复该会话保存的消息。
    switchChat(chatId) {
      const chat = this.chatHistory.find((item) => item.id === chatId)
      if (!chat) return
      this.currentChatId = chatId
      this.messages = [...chat.messages]
    },

    // 删除当前聊天后，优先切换到原本排在它下面的聊天；列表为空时创建一个新聊天。
    deleteChat(chatId) {
      const deleteIndex = this.chatHistory.findIndex((item) => item.id === chatId)
      if (deleteIndex === -1) return

      const isCurrentChat = this.currentChatId === chatId
      this.chatHistory.splice(deleteIndex, 1)

      // 删除的不是当前聊天时，右侧正在显示的内容不需要改变。
      if (!isCurrentChat) return

      // 删除后相同下标的位置，就是删除前排在它下面的聊天。
      const nextChat = this.chatHistory[deleteIndex] || this.chatHistory[deleteIndex - 1]
      if (nextChat) {
        this.currentChatId = nextChat.id
        this.messages = [...nextChat.messages]
      } else {
        this.createChat()
      }
    },

    // 完整流程：保存用户案情 -> 请求流式接口 -> 逐段显示案件对比 -> 保存引用类案。
    async sendMessage(content) {
      const question = content.trim()
      if (!question || this.loading) return

      const chatId = this.currentChatId
      const chat = this.chatHistory.find((item) => item.id === chatId)
      if (!chat) return

      this.messages.push({ id: Date.now(), role: 'user', content: question })
      if (chat.title === '当前聊天' || chat.title === '新对话') {
        chat.title = question.slice(0, 16)
      }
      chat.messages = [...this.messages]

      // 先创建 AI 占位消息，检索结束后模型生成的文字会持续追加到这里。
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

      this.loading = true
      let hasReceivedToken = false

      try {
        // 当前聊天 ID 同时作为后端 Agent 的 thread_id。
        await sendQuestion(question, chatId, (event) => {
          if (event.type === 'status' && !hasReceivedToken) {
            assistantMessage.content = event.message
          }

          if (event.type === 'token') {
            // 第一个正文片段到达时移除检索提示，后续片段直接追加。
            if (!hasReceivedToken) {
              assistantMessage.content = ''
              hasReceivedToken = true
            }
            assistantMessage.content += event.content
          }

          if (event.type === 'sources') {
            assistantMessage.sources = event.sources || []
          }

          // 用户等待期间即使切换了会话，流式内容也只更新它原本所属的聊天。
          if (this.currentChatId === chatId) {
            this.messages = [...chat.messages]
          }
        })
      } catch (error) {
        console.error(error)
        // 如果已经显示了部分正文就保留它，否则用统一错误提示替换检索占位文字。
        assistantMessage.content = hasReceivedToken
          ? `${assistantMessage.content}\n\n回答生成中断，请稍后重试。`
          : '无法连接类案检索后端，请确认 FastAPI 已启动。'

        if (this.currentChatId === chatId) {
          this.messages = [...chat.messages]
        }
      } finally {
        this.loading = false
      }
    },
  },
})
