<template>
  <div class="upload-panel">
    <el-upload
      drag
      :show-file-list="false"
      :http-request="doUpload"
      :before-upload="beforeUpload"
      accept=".pdf,.txt,.md"
      :disabled="uploading"
    >
      <el-icon class="up-icon"><UploadFilled /></el-icon>
      <div class="up-text">拖拽文档到此处，或 <em>点击上传</em></div>
      <div class="up-hint">支持 PDF / TXT / MD，单个不超过 20MB</div>
    </el-upload>

    <div v-if="uploading" class="up-status">
      <el-icon class="is-loading"><Loading /></el-icon>
      <span>正在解析并向量化…</span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { documentApi } from '../api/request'

const emit = defineEmits(['uploaded'])
const uploading = ref(false)

const MAX_SIZE = 20 * 1024 * 1024
const ALLOWED = ['pdf', 'txt', 'md']

function beforeUpload(file) {
  const ext = file.name.split('.').pop()?.toLowerCase()
  if (!ALLOWED.includes(ext)) {
    ElMessage.error('仅支持 PDF / TXT / MD 格式')
    return false
  }
  if (file.size > MAX_SIZE) {
    ElMessage.error('文件超过 20MB 限制')
    return false
  }
  return true
}

async function doUpload({ file }) {
  uploading.value = true
  try {
    const doc = await documentApi.upload(file)
    ElMessage.success(`《${doc.filename}》已入库，共 ${doc.chunks} 个片段`)
    emit('uploaded', doc)
  } catch {
    // 错误提示已在 axios 拦截器中统一处理
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-panel {
  margin-bottom: 12px;
}

.up-icon {
  font-size: 34px;
  color: #9aa4bd;
  margin-top: 8px;
}

.up-text {
  font-size: 13px;
  color: var(--text-main);
}

.up-text em {
  color: var(--brand);
  font-style: normal;
}

.up-hint {
  font-size: 12px;
  color: var(--text-sub);
  margin-bottom: 8px;
}

.up-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--brand);
  margin-top: 8px;
}

:deep(.el-upload-dragger) {
  padding: 14px 10px;
  border-radius: 10px;
  background: #fff;
}
</style>