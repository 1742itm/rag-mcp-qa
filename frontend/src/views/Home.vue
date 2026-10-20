<template>
  <div class="layout">
    <header class="topbar">
      <div class="brand">
        <el-icon class="brand-icon"><MagicStick /></el-icon>
        <div>
          <div class="brand-title">AI Agent 文档问答系统</div>
          <div class="brand-sub">MCP 上下文编排 · RAG 检索增强 · SSE 流式回答</div>
        </div>
      </div>

      <div class="badges">
        <el-tooltip :content="`${health?.llm?.base_url || ''}`" placement="bottom">
          <el-tag :type="health?.llm?.configured ? 'success' : 'danger'" effect="light">
            <el-icon><Cpu /></el-icon>
            {{ health?.llm?.model || '未连接' }}
          </el-tag>
        </el-tooltip>
        <el-tag type="info" effect="light">
          <el-icon><DataAnalysis /></el-icon>
          向量：{{ health?.embedding?.provider || '-' }}
        </el-tag>
        <el-tag type="info" effect="light">
          <el-icon><Files /></el-icon>
          片段：{{ health?.chunks ?? 0 }}
        </el-tag>
      </div>
    </header>

    <div class="body">
      <aside class="side">
        <section class="panel">
          <div class="panel-title">
            <el-icon><FolderOpened /></el-icon>
            知识库文档
          </div>
          <UploadPanel @uploaded="loadDocuments" />
          <DocumentList :documents="documents" @removed="onDocumentRemoved" />
        </section>

        <section class="panel">
          <div class="panel-title">
            <el-icon><ChatDotRound /></el-icon>
            历史会话
            <el-button
              class="new-btn"
              size="small"
              type="primary"
              link
              :icon="Plus"
              @click="newConversation"
            >
              新建
            </el-button>
          </div>
          <el-empty
            v-if="!conversations.length"
            description="暂无会话记录"
            :image-size="50"
          />
          <div v-else class="conv-items">
            <div
              v-for="conv in conversations"
              :key="conv.id"
              class="conv-item"
              :class="{ active: conv.id === currentConversationId }"
              @click="selectConversation(conv.id)"
            >
              <div class="conv-title" :title="conv.title">{{ conv.title }}</div>
              <el-button
                type="danger"
                link
                :icon="Delete"
                @click.stop="removeConversation(conv.id)"
              />
            </div>
          </div>
        </section>
      </aside>

      <main class="main">
        <ChatPanel
          ref="chatRef"
          :conversation-id="currentConversationId"
          @meta="onMeta"
          @done="loadConversations"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Plus } from '@element-plus/icons-vue'

import UploadPanel from '../components/UploadPanel.vue'
import DocumentList from '../components/DocumentList.vue'
import ChatPanel from '../components/ChatPanel.vue'
import { conversationApi, documentApi, systemApi } from '../api/request'

const health = ref(null)
const documents = ref([])
const conversations = ref([])
const currentConversationId = ref(null)
const chatRef = ref(null)

async function loadHealth() {
  try {
    health.value = await systemApi.health()
  } catch {
    health.value = null
  }
}

async function loadDocuments() {
  documents.value = await documentApi.list()
  loadHealth()
}

async function loadConversations() {
  conversations.value = await conversationApi.list()
}

/** SSE 首帧返回会话信息：新建的会话在这里挂上 ID 并刷新列表 */
function onMeta(data) {
  if (data.conversation_id) currentConversationId.value = data.conversation_id
  loadConversations()
}

async function selectConversation(id) {
  currentConversationId.value = id
  await chatRef.value.loadConversation(id)
}

function newConversation() {
  currentConversationId.value = null
  chatRef.value.reset()
}

async function removeConversation(id) {
  try {
    await ElMessageBox.confirm('确认删除该会话及其消息记录？', '删除确认', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  await conversationApi.remove(id)
  ElMessage.success('已删除')
  if (id === currentConversationId.value) newConversation()
  loadConversations()
}

function onDocumentRemoved() {
  loadDocuments()
}

onMounted(() => {
  loadHealth()
  loadDocuments()
  loadConversations()
})
</script>

<style scoped>
.layout {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.topbar {
  height: 62px;
  flex-shrink: 0;
  padding: 0 22px;
  background: #fff;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-icon {
  font-size: 24px;
  color: var(--brand);
}

.brand-title {
  font-size: 16px;
  font-weight: 600;
}

.brand-sub {
  font-size: 12px;
  color: var(--text-sub);
  margin-top: 2px;
}

.badges {
  display: flex;
  gap: 8px;
}

.body {
  flex: 1;
  min-height: 0;
  display: flex;
  gap: 14px;
  padding: 14px;
}

.side {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
}

.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 12px;
  color: var(--text-main);
}

.new-btn {
  margin-left: auto;
}

.conv-items {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 260px;
  overflow-y: auto;
}

.conv-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 10px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.15s;
}

.conv-item:hover {
  background: #f2f5fc;
}

.conv-item.active {
  background: var(--brand-soft);
  color: var(--brand);
}

.conv-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.main {
  flex: 1;
  min-width: 0;
}
</style>