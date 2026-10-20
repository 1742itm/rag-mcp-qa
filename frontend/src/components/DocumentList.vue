<template>
  <div class="doc-list">
    <el-empty v-if="!documents.length" description="暂无文档，先上传一份吧" :image-size="60" />
    <div v-else class="doc-items">
      <div v-for="doc in documents" :key="doc.id" class="doc-item">
        <el-icon class="doc-icon"><Document /></el-icon>
        <div class="doc-info">
          <div class="doc-name" :title="doc.filename">{{ doc.filename }}</div>
          <div class="doc-meta">{{ doc.chunks }} 个片段 · {{ doc.created_at }}</div>
        </div>
        <el-button
          type="danger"
          link
          :icon="Delete"
          title="删除该文档及其向量"
          @click="onRemove(doc)"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ElMessageBox, ElMessage } from 'element-plus'
import { Delete } from '@element-plus/icons-vue'
import { documentApi } from '../api/request'

const props = defineProps({
  documents: { type: Array, default: () => [] },
})
const emit = defineEmits(['removed'])

async function onRemove(doc) {
  try {
    await ElMessageBox.confirm(
      `确认删除《${doc.filename}》？该文档的向量数据会一并移除。`,
      '删除确认',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return // 用户取消
  }
  await documentApi.remove(doc.id)
  ElMessage.success('已删除')
  emit('removed', doc.id)
}
</script>

<style scoped>
.doc-list {
  max-height: 320px;
  overflow-y: auto;
}

.doc-items {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 10px;
}

.doc-icon {
  color: var(--brand);
  font-size: 18px;
  flex-shrink: 0;
}

.doc-info {
  flex: 1;
  min-width: 0;
}

.doc-name {
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.doc-meta {
  font-size: 11px;
  color: var(--text-sub);
  margin-top: 2px;
}
</style>