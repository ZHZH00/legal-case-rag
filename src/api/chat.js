const CHAT_API_URL = '/api/chat'

// 这个文件负责读取后端数据流，并把每个回答片段或来源事件交给 Store。
export async function sendQuestion(question, threadId, onEvent) {
  const response = await fetch(CHAT_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    // thread_id 让后端能把同一个前端聊天的多次请求关联起来。
    body: JSON.stringify({ question, thread_id: String(threadId) }),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => null)
    throw new Error(errorData?.detail || '聊天接口请求失败')
  }

  if (!response.body) {
    throw new Error('当前浏览器无法读取流式回答')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  // 网络数据块不一定刚好对应一行，所以先放入 buffer，再按换行符拆分完整事件。
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

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

  // 正常情况下事件都以换行结尾，这里兼容最后一行没有换行符的情况。
  if (buffer.trim()) {
    const event = JSON.parse(buffer)
    if (event.type === 'error') {
      throw new Error(event.message)
    }
    onEvent(event)
  }
}
