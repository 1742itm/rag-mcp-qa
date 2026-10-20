import axios from 'axios'
import { ElMessage } from 'element-plus'

// 统一 axios 实例：走 Vite 代理，前缀 /api
const request = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// 统一拆包：直接返回 response.data
request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error?.response?.data?.detail || error.message || '请求失败'
    ElMessage.error(detail)
    return Promise.reject(error)
  },
)

/** 文档相关接口 */
export const documentApi = {
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request.post('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  list: () => request.get('/documents'),
  remove: (id) => request.delete(`/documents/${id}`),
}

/** 会话相关接口 */
export const conversationApi = {
  list: () => request.get('/conversations'),
  create: (title = '新对话') => request.post('/conversations', { title }),
  messages: (id) => request.get(`/conversations/${id}/messages`),
  remove: (id) => request.delete(`/conversations/${id}`),
}

/** 系统状态 */
export const systemApi = {
  health: () => request.get('/health'),
}

export default request