const CHAT_API_URL = '/api/chat'

// 请求聊天接口并解析NDJSON事件流。
export async function sendQuestion(question, threadId, onEvent) {
  // thread_id关联后端持久化状态。
  const response = await fetch(CHAT_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, thread_id: String(threadId) }),
  })

  // 优先返回后端错误详情。
  if (!response.ok) {
    const errorData = await response.json().catch(() => null)
    throw new Error(errorData?.detail || '聊天接口请求失败')
  }

  // 流式响应必须提供可读正文。
  if (!response.body) {
    throw new Error('当前浏览器无法读取流式回答')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  // 缓冲跨网络分块的未完整JSON行。
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    // 逐行解析并分发完整事件。
    for (const line of lines) {
      if (!line.trim()) continue
      const event = JSON.parse(line)
      if (event.type === 'error') {
        throw new Error(event.message)
      }
      onEvent(event)
    }

    if (done) break
  }

  // 兼容末尾缺少换行符的事件。
  if (buffer.trim()) {
    const event = JSON.parse(buffer)
    if (event.type === 'error') {
      throw new Error(event.message)
    }
    onEvent(event)
  }
}
