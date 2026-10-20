/**
 * SSE 封装：基于原生 EventSource 订阅 /api/chat/stream。
 *
 * 事件协议（后端定义）：
 *   meta        会话信息（首次返回，含 conversation_id / title）
 *   status      阶段状态（routing / ...）
 *   tool_call   模型决定调用工具
 *   tool_result 工具返回结果（含 references）
 *   delta       回答文本增量
 *   done        结束（含 message_id / references / tool_calls）
 *   error       业务错误
 */
const EVENT_NAMES = ['meta', 'status', 'tool_call', 'tool_result', 'delta', 'done', 'error']

export function openChatStream({ message, conversationId, onEvent }) {
  const params = new URLSearchParams({ message })
  if (conversationId) params.set('conversation_id', conversationId)

  const source = new EventSource(`/api/chat/stream?${params.toString()}`)
  let closed = false

  const close = () => {
    if (!closed) {
      closed = true
      source.close()
    }
  }

  EVENT_NAMES.forEach((name) => {
    source.addEventListener(name, (event) => {
      let data = {}
      try {
        data = JSON.parse(event.data)
      } catch {
        data = { raw: event.data }
      }
      // 终态事件后主动断开，避免 EventSource 自动重连导致重复提问
      if (name === 'done' || name === 'error') close()
      onEvent(name, data)
    })
  })

  source.onerror = () => {
    if (closed) return
    close()
    onEvent('transport_error', { message: '连接中断，请确认后端服务已启动' })
  }

  return { close }
}