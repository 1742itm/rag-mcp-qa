<template>
  <div class="chat-panel">
    <div ref="scrollRef" class="chat-messages">
      <el-empty
        v-if="!messages.length"
        description="上传文档后，向 AI 提问吧"
        :image-size="90"
      />

      <div
        v-for="(msg, index) in messages"
        :key="index"
        class="message"
        :class="msg.role"
      >
        <div class="avatar">
          <el-icon v-if="msg.role === 'user'"><User /></el-icon>
          <el-icon v-else><MagicStick /></el-icon>
        </div>

        <div class="bubble-wrap">
          <!-- 工具调用轨迹 -->
          <div v-if="msg.toolCalls?.length" class="trace">
            <el-tag size="small" type="warning" effect="plain">
              <el-icon><Tools /></el-icon>
              调用工具：{{ msg.toolCalls.map(toolLabel).join(' + ') }}
            </el-tag>
            <el-tag v-if="msg.rounds > 1" size="small" effect="plain">
              多轮编排 {{ msg.rounds }} 轮
            </el-tag>
            <el-tag v-if="msg.references?.length" size="small" type="success" effect="plain">
              命中 {{ msg.references.length }} 个片段
            </el-tag>
          </div>

          <div class="bubble" :class="{ error: msg.error }">
            <!-- 助手回答按 Markdown 渲染（表格 / 列表 / 加粗）；用户消息与错误提示保持纯文本 -->
            <div
              v-if="msg.role === 'assistant' && msg.html && !msg.error"
              class="markdown-body"
              v-html="msg.html"
            />
            <template v-else-if="msg.content">{{ msg.content }}</template>
            <span v-else-if="msg.streaming" class="thinking">
              <span class="dot" /><span class="dot" /><span class="dot" />
            </span>
          </div>

          <!-- 引用来源 -->
          <el-collapse v-if="msg.references?.length" class="refs">
            <el-collapse-item>
              <template #title>
                <span class="refs-title">
                  <el-icon><Collection /></el-icon>
                  引用来源（{{ msg.references.length }}）
                </span>
              </template>
              <div
                v-for="(ref, i) in msg.references"
                :key="i"
                class="ref-item"
              >
                <div class="ref-head">
                  <span class="ref-index">[{{ i + 1 }}]</span>
                  <span class="ref-file">{{ ref.filename }}</span>
                  <el-tag size="small" :type="scoreType(ref.score)" effect="plain">
                    相似度 {{ ref.score }}
                  </el-tag>
                </div>
                <div class="ref-text">{{ ref.text }}</div>
              </div>
            </el-collapse-item>
          </el-collapse>
        </div>
      </div>
    </div>

    <div class="chat-footer">
      <div v-if="statusText" class="status-text">
        <el-icon class="is-loading"><Loading /></el-icon>
        {{ statusText }}
      </div>
      <el-input
        v-model="input"
        type="textarea"
        :rows="3"
        resize="none"
        placeholder="输入你的问题，Enter 发送，Shift + Enter 换行"
        :disabled="sending"
        @keydown.enter="onEnter"
      />
      <div class="actions">
        <span class="tip">模型会自主判断是否需要检索知识库</span>
        <el-button type="primary" :loading="sending" :icon="Promotion" @click="send">
          发送
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Promotion } from '@element-plus/icons-vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { conversationApi } from '../api/request'
import { openChatStream } from '../api/sse'

// GFM 提供表格支持；breaks 让单个换行也生效，更贴合对话的排版习惯
marked.setOptions({ gfm: true, breaks: true })

// 工具名的中文展示名；未收录的按原名显示
const TOOL_LABELS = {
  knowledge_search: '知识库检索',
  list_documents: '文档清单',
  get_document_content: '读取文档正文',
}

function toolLabel(name) {
  return TOOL_LABELS[name] || name
}

/** 渲染助手回答：Markdown → HTML，并做 XSS 过滤（内容来自模型，必须净化） */
function renderMarkdown(text) {
  if (!text) return ''
  const html = marked.parse(text, { async: false })
  return DOMPurify.sanitize(html)
}

// ---------------- 流式增量渲染 ----------------
// 朴素的「每次刷新都全量重解析」是 O(n²)：实测典型回答（约 1200 字符）累计约 5ms，
// 尚可接受；但回答到 8000 字符时累计涨到约 217ms，20000 字符时约 1.4s。
// 因此改成增量渲染：把「已经闭合的 Markdown 块」渲染一次缓存进前缀，
// 每次刷新只重解析尚未闭合的尾部，使单次刷新成本与回答总长解耦。
//
// 取舍：分块渲染与整体渲染在个别边界（松散列表、缩进代码块）上可能有细微差异，
// 所以流结束时必须再做一次全量渲染兜底（finalizeRender），保证最终结果与整体解析一致。
const FENCE_RE = /^ {0,3}```/gm

function isFenceBalanced(text) {
  const fences = text.match(FENCE_RE)
  return !fences || fences.length % 2 === 0
}

/** 找出可以安全提交的最后位置：未被「未闭合代码块」包住的那个空行之后。 */
function findCommitPoint(content, from) {
  let limit = content.length
  while (true) {
    const idx = content.lastIndexOf('\n\n', limit - 1)
    if (idx === -1 || idx + 2 <= from) return from
    const candidate = idx + 2
    // 代码块内部允许出现空行，因此必须确认围栏已闭合才敢把它当块边界
    if (isFenceBalanced(content.slice(0, candidate))) return candidate
    limit = idx
  }
}

function renderIncremental(msg) {
  const content = msg.content || ''
  const commit = findCommitPoint(content, msg.committedLength)
  if (commit > msg.committedLength) {
    msg.htmlPrefix += renderMarkdown(content.slice(msg.committedLength, commit))
    msg.committedLength = commit
  }
  msg.html = msg.htmlPrefix + renderMarkdown(content.slice(msg.committedLength))
}

function renderFull(msg) {
  msg.html = msg.error ? '' : renderMarkdown(msg.content || '')
  msg.htmlPrefix = ''
  msg.committedLength = 0
}

// 流式期间每个 delta 都渲染是浪费，这里节流：首个 delta 立即渲染保证响应感，
// 之后最多每 RENDER_INTERVAL 毫秒刷一次；流结束时全量渲染兜底。
const RENDER_INTERVAL = 120
let lastRenderAt = 0
let renderTimer = null
let pendingMsg = null

function flushRender(msg) {
  if (renderTimer !== null) {
    clearTimeout(renderTimer)
    renderTimer = null
  }
  lastRenderAt = Date.now()
  if (msg.error) {
    msg.html = ''
    return
  }
  renderIncremental(msg)
}

function scheduleRender(msg) {
  const elapsed = Date.now() - lastRenderAt
  if (elapsed >= RENDER_INTERVAL) {
    flushRender(msg)
    return
  }
  pendingMsg = msg
  if (renderTimer !== null) return
  renderTimer = setTimeout(() => {
    renderTimer = null
    if (pendingMsg) flushRender(pendingMsg)
  }, RENDER_INTERVAL - elapsed)
}

/** 流结束时调用：全量渲染一次，把增量渲染可能产生的边界差异抹平。 */
function finalizeRender(msg) {
  if (renderTimer !== null) {
    clearTimeout(renderTimer)
    renderTimer = null
  }
  lastRenderAt = Date.now()
  renderFull(msg)
}

function cancelPendingRender() {
  if (renderTimer !== null) {
    clearTimeout(renderTimer)
    renderTimer = null
  }
  pendingMsg = null
}

const props = defineProps({
  conversationId: { type: String, default: null },
})
const emit = defineEmits(['meta', 'done'])

const messages = ref([])
const input = ref('')
const sending = ref(false)
const statusText = ref('')
const scrollRef = ref(null)

let stream = null

// ---------------- 滚动到底部 ----------------
function scrollToBottom() {
  nextTick(() => {
    const el = scrollRef.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

watch(messages, scrollToBottom, { deep: true })

// ---------------- 发送 ----------------
function onEnter(event) {
  if (event.shiftKey) return
  event.preventDefault()
  send()
}

function send() {
  const question = input.value.trim()
  if (!question || sending.value) return

  messages.value.push({ role: 'user', content: question })
  const assistant = reactiveMessage()
  messages.value.push(assistant)
  input.value = ''
  sending.value = true
  statusText.value = '正在思考…'

  stream = openChatStream({
    message: question,
    conversationId: props.conversationId,
    onEvent: (name, data) => handleEvent(name, data, assistant),
  })
}

function reactiveMessage() {
  return {
    role: 'assistant',
    content: '',
    html: '', // 已渲染好的 HTML，供 v-html 绑定
    htmlPrefix: '', // 已闭合块的渲染结果缓存（增量渲染用）
    committedLength: 0, // 已被 htmlPrefix 覆盖的字符数
    references: [],
    toolCalls: [],
    rounds: 0,
    streaming: true,
    error: false,
  }
}

function handleEvent(name, data, assistant) {
  switch (name) {
    case 'meta':
      emit('meta', data)
      break
    case 'status':
      if (data.stage === 'routing') {
        statusText.value =
          data.round > 1
            ? `第 ${data.round} 轮：判断是否需要补充检索…`
            : '正在判断是否检索…'
      } else if (data.stage === 'answering') {
        statusText.value = '正在生成回答…'
      } else {
        statusText.value = '处理中…'
      }
      break
    case 'tool_call':
      // 多轮编排时同一工具会多次触发，工具名去重展示，轮次单独计数
      if (!assistant.toolCalls.includes(data.tool)) assistant.toolCalls.push(data.tool)
      assistant.rounds = data.round ?? assistant.rounds + 1
      statusText.value = `第 ${assistant.rounds} 轮调用「${toolLabel(data.tool)}」…`
      break
    case 'tool_result':
      statusText.value = data.count
        ? `第 ${data.round} 轮「${toolLabel(data.tool)}」返回 ${data.count} 条结果，继续判断…`
        : `第 ${data.round} 轮「${toolLabel(data.tool)}」未返回结果，继续判断…`
      break
    case 'delta':
      assistant.content += data.content || ''
      scheduleRender(assistant)
      break
    case 'done':
      finish(assistant, data.references || [], data.rounds || 0)
      break
    case 'error':
    case 'transport_error':
      assistant.error = true
      assistant.content = assistant.content || (data.message || '发生错误')
      if (!assistant.content) assistant.content = '发生错误'
      ElMessage.error(data.message || '请求失败')
      finish(assistant, [])
      break
  }
}

function finish(assistant, references, rounds = 0) {
  assistant.streaming = false
  if (rounds) assistant.rounds = rounds
  if (references.length) assistant.references = references
  if (!assistant.content) assistant.content = '（模型没有返回内容）'
  finalizeRender(assistant) // 全量渲染兜底，抹平增量渲染的边界差异
  sending.value = false
  statusText.value = ''
  stream = null
  emit('done')
}

function scoreType(score) {
  if (score >= 0.7) return 'success'
  if (score >= 0.4) return 'warning'
  return 'info'
}

// ---------------- 对外方法 ----------------
async function loadConversation(id) {
  stop()
  messages.value = []
  if (!id) return
  const history = await conversationApi.messages(id)
  // 历史消息不是流式的，直接一次全量渲染，无需增量
  messages.value = history.map((m) => {
    const msg = {
      role: m.role,
      content: m.content,
      html: '',
      htmlPrefix: '',
      committedLength: 0,
      references: m.references || [],
      toolCalls: [],
      rounds: 0,
      streaming: false,
      error: false,
    }
    if (m.role === 'assistant') renderFull(msg)
    return msg
  })
}

function reset() {
  stop()
  messages.value = []
}

function stop() {
  if (stream) {
    stream.close()
    stream = null
  }
  cancelPendingRender()
  sending.value = false
  statusText.value = ''
}

defineExpose({ loadConversation, reset })
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-panel);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  overflow: hidden;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 22px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.message {
  display: flex;
  gap: 10px;
  max-width: 100%;
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--brand-soft);
  color: var(--brand);
  flex-shrink: 0;
}

.message.user .avatar {
  background: #e8f5ee;
  color: #22a06b;
}

.bubble-wrap {
  max-width: 78%;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.message.user .bubble-wrap {
  align-items: flex-end;
}

.trace {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.bubble {
  padding: 10px 14px;
  border-radius: 10px;
  background: #f4f6fb;
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.message.user .bubble {
  background: var(--brand);
  color: #fff;
}

.bubble.error {
  background: #fef1f1;
  color: #c0392b;
}

.thinking {
  display: inline-flex;
  gap: 4px;
  align-items: center;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #a9b2c9;
  animation: blink 1.2s infinite;
}

.dot:nth-child(2) {
  animation-delay: 0.2s;
}

.dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes blink {
  0%,
  60%,
  100% {
    opacity: 0.3;
  }
  30% {
    opacity: 1;
  }
}

.refs {
  border: none;
  --el-collapse-header-height: 30px;
}

.refs-title {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--text-sub);
}

.ref-item {
  padding: 8px 10px;
  background: #fbfcff;
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 6px;
}

.ref-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.ref-index {
  color: var(--brand);
  font-weight: 600;
}

.ref-file {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ref-text {
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-sub);
  line-height: 1.6;
  max-height: 110px;
  overflow-y: auto;
}

.chat-footer {
  border-top: 1px solid var(--border);
  padding: 12px 16px 14px;
  background: #fff;
}

.status-text {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--brand);
  margin-bottom: 8px;
}

.actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
}

.tip {
  font-size: 12px;
  color: var(--text-sub);
}
</style>