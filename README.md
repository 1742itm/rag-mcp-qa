# AI Agent 文档问答系统（MCP + RAG | Vue3 + FastAPI）

轻量级 Agentic RAG 项目：自研轻量工具协议（MCP 风格）+ RAG 检索增强，实现基于私有文档的知识库问答。
Vue3 前端 + FastAPI 后端，完整前后端联调，适合实训与简历项目。

## ✨ 项目亮点

- **ReAct Agent 编排**：把知识库能力注册成 3 个工具（语义检索 / 文档清单 / 读取正文），
  模型在与回答同一次流式调用里自主决定调哪个工具、调几次，支持多轮编排
  （实测能对「有哪些文档 + 短期记忆存在哪」这类复合问题自发分两轮调用两个工具）
- **RAG 检索增强**：文档向量化存储，**相似度阈值过滤噪声**（阈值经实测标定），
  回答附带原文引用、检索不到时如实说明，降低大模型幻觉；
  **长答案自动补齐相邻上下文**（一道题跨多段时不再"答一半"）
- **流式对话**：SSE 实时逐字输出，前端 EventSource 订阅并动态渲染
- **Markdown 回答渲染**：助手回答按 Markdown 渲染（表格 / 列表 / 代码块 / 加粗），
  经 DOMPurify 净化后注入，兼顾可读性与 XSS 安全；
  流式期间**增量渲染**——已闭合的块只解析一次并缓存，每次仅重解析未闭合的尾部
  （实测解析量降低 10~60 倍），流结束再做一次全量渲染兜底保证最终结果一致
- **轻量化部署**：Chroma 本地向量库 + SQLite，零中间件；向量模型缺失时自动降级，**离线也能跑通全链路**

## 📚 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | Python 3.10+、FastAPI、Chroma、sentence-transformers、pypdf、langchain-text-splitters、sse-starlette、SQLite |
| 前端 | Vue3 + Vite、Element Plus、Axios、EventSource、marked + DOMPurify（Markdown 渲染与净化） |
| AI | 兼容 OpenAI 接口的 LLM（DeepSeek / Ollama / 通义千问），自研轻量工具协议（MCP 风格，未依赖官方 SDK） |

## 📁 目录结构

```
rag-mcp-qa/
├── backend/
│   ├── app/
│   │   ├── mcp/            # 自研 MCP：协议 / 注册表 / 判别器 / 上下文 / Agent 编排 / 内置工具
│   │   ├── rag/            # 文档解析切片 / 向量化 / Chroma 存储
│   │   ├── llm/            # 兼容 OpenAI 接口的 LLM 客户端（含流式）
│   │   ├── db/             # SQLite：文档元数据 + 会话 + 消息
│   │   ├── api/            # 路由：documents / chat / conversations / system
│   │   ├── config.py       # 配置（读取 .env）
│   │   └── main.py         # FastAPI 入口
│   ├── evals/              # 检索评测：内置带标注语料 + 阈值标定（python -m evals.retrieval_eval）
│   ├── tests/              # pytest：判别器 / ReAct 循环 / 向量库阈值与相邻窗口 / 内置工具 / 评测脚本
│   ├── data/               # Chroma 持久化 + app.db（运行时生成）
│   ├── uploads/            # 上传的原始文件（运行时生成）
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/            # axios 请求封装 + SSE 封装
│   │   ├── components/     # UploadPanel / DocumentList / ChatPanel
│   │   └── views/          # Home 页面
│   ├── vite.config.js      # 含 /api 代理，前端无需处理跨域
│   └── package.json
└── README.md
```

## 🚀 快速开始

### 1. 后端（虚拟环境）

```bash
cd backend

# 创建虚拟环境（依赖全部装在项目内，不影响本机环境）
python -m venv .venv

# 激活（Windows PowerShell）
.\.venv\Scripts\Activate.ps1
# macOS / Linux：source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置模型：复制 .env.example 为 .env，填入 API Key
#   DeepSeek:    LLM_BASE_URL=https://api.deepseek.com/v1   LLM_MODEL=deepseek-chat
#   通义千问:     LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-plus
#   本地 Ollama: LLM_BASE_URL=http://localhost:11434/v1     LLM_MODEL=qwen2.5

# 启动
uvicorn app.main:app --reload --port 8000
```

接口文档：http://127.0.0.1:8000/docs

### 2. 前端

```bash
cd frontend
npm install
npm run dev
```

打开 http://127.0.0.1:5173

### 3. 使用

1. 左侧拖拽上传 PDF / TXT / MD，等待解析向量化
2. 在右侧输入问题，模型会自主决定是否调用「知识库检索」工具
3. 回答流式输出，展开「引用来源」可查看命中的原文片段与相似度

### 4. 运行测试

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

共 158 项用例、约 11s 跑完：

| 文件 | 覆盖内容 |
| --- | --- |
| `tests/test_discriminator.py` | 流式判别的四类结论与边界：前置语、未闭合 JSON、围栏、转义、fail-open、增量流式 |
| `tests/test_agent_react.py` | ReAct 循环：**工具 JSON 不泄露**、调用次数预算、收尾轮不暴露工具、Observation 注入、检索为空时如实告知、**多工具编排与重复调用拦截**、去重排序、事件协议 |
| `tests/test_vector_store.py` | 相似度阈值：边界过滤、降级模式单独阈值、元数据完整性；**相邻窗口读取**的越界处理与跨文档隔离（全部离线、确定性） |
| `tests/test_tools.py` | 三个内置工具的返回约定与边界：参数区间、文件名模糊匹配、阈值与 top_k 的相互作用、**相邻上下文扩展与去重截断** |
| `tests/test_retrieval_eval.py` | 评测脚本的指标计算、阈值扫描、建议阈值、评测集标注完整性 |

## 🔌 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/documents/upload` | 上传文档（解析 → 切片 → 向量化） |
| GET | `/api/documents` | 知识库文档列表 |
| DELETE | `/api/documents/{id}` | 删除文档及其向量 |
| GET | `/api/chat/stream` | SSE 流式问答（`message`、`conversation_id`） |
| GET | `/api/conversations` | 会话列表 |
| GET | `/api/conversations/{id}/messages` | 会话消息历史 |
| DELETE | `/api/conversations/{id}` | 删除会话 |
| GET | `/api/health` | 运行状态（模型 / 向量化 / 工具 / 片段数） |

### SSE 事件协议

```
meta         会话信息 {"conversation_id", "title"}
status       阶段状态 {"stage": "routing" | "answering", "round"}
tool_call    模型决定调用工具 {"tool", "arguments", "round"}
tool_result  工具返回结果 {"tool", "count", "fresh", "round"}   count=工具自报条数, fresh=新增可引用片段数
delta        回答文本增量 {"content"}
done         结束 {"message_id", "references", "tool_calls", "rounds"}
error        错误 {"message"}
```

`status` / `tool_call` / `tool_result` 在多轮检索时会**重复触发**，`round` 标识当前轮次。

`done.references` 里除命中的片段外，还包含为补齐长答案而扩展的**相邻上下文片段**——
它们的 `score` 为 `null`、`expanded` 为 `true`（见「长答案为什么只答一半」一节）。

## 🧠 MCP 是怎么实现的

> **关于「标准 MCP」**：本项目**未引入官方 MCP SDK**（`mcp` 包），工具协议是自研的轻量实现。
> 它复刻了 MCP 的核心思想——把能力声明成工具、由模型自主选择调用，并以此与 LLM 厂商解耦；
> 但它**不具备** MCP 的 JSON-RPC 传输与生态互通，无法直接挂到 Claude Desktop / Cursor 等 MCP Host。
> 若需要生态互通，可以再建一个 FastMCP Server，把下面这几个工具按标准 MCP 再暴露一遍
> （handler 直接复用 `app/mcp/tools.py`，业务逻辑不必重写）。

自研的轻量工具协议只做三件事，但完整复刻了「工具调用编排」的思想：

1. **协议层**（`app/mcp/protocol.py`）：用 `Tool` 描述工具的 `name / description / parameters`，并可序列化为 JSON Schema。新增能力只需注册一个工具，无需改动 Agent。
2. **注册表**（`app/mcp/registry.py`）：统一登记与执行工具，并对模型传错的参数做兜底裁剪。
3. **内置工具**（`app/mcp/tools.py`）：三个工具覆盖三类不同形态，用来证明「注册即用、模型自主选择」而非写死一条检索链路。

   | 工具 | 参数 | 返回 | 适用意图 |
   | --- | --- | --- | --- |
   | `knowledge_search` | `query`、可选 `top_k` | 可引用的原文片段 | 「哪段话能回答这个问题」 |
   | `list_documents` | 无 | 文档清单（文件名/片段数/时间） | 「知识库有哪些文档」 |
   | `get_document_content` | `filename`、可选区间 | 指定文档的正文片段 | 「读一下某某文档」 |

   工具返回遵循统一约定，**Agent 不写死任何具体工具**：
   - `summary`（必填）：面向模型的自述，Agent 包成 Observation 注入对话；
   - `references`（仅检索类工具）：Agent 据此收集「引用来源」并做跨轮去重。

   因此再注册新工具时，`agent.py` 一行都不用改。

4. **Agent 编排**（`app/mcp/agent.py`）：**ReAct 风格的单次调用循环**——模型在同一个流式响应里，
   自己决定这一轮输出的是「工具调用」还是「最终回答」：

   ```
   流式接收 → 判别器分流（app/mcp/discriminator.py）
     ├─ TOOL     → 执行工具，把 Observation 追加进对话，进入下一轮
     ├─ ANSWER   → 已边收边展示完，直接收尾
     └─ NO_TOOL  → 模型没给出回答，进入收尾轮重新索取
   ```

   **为什么比「先路由决策、再单独作答」的两段式更快**：旧方案每一轮都要一次独立的
   非流式决策调用，最后还要再来一次作答调用；ReAct 里**模型作答的那一轮本身就是流式回答**，
   所以 N 轮检索只需 N+1 次调用（旧方案 N+2 次）。

   需要最终回答时，收尾轮的提示词里**不再给出工具清单**——模型看不到工具就无法再产出工具调用，
   因此**必定收敛、一定能拿到回答**。

5. **判别器**（`app/mcp/discriminator.py`）：这套方案的关键部件。
   工具调用的 JSON 一旦被当成正文流出去就直接泄露给用户。判别器的做法是锁定文本里第一个 `{`
   ——它一出现就转入「JSON 候选」等待闭合，期间一个字符都不展示；闭合后确认是工具调用就执行、
   确认是正文才放行。一旦判定为回答即进入 latch，后续文本直接透传、不再回头改判。

多轮的价值：单轮检索只拿一次 Top-K，遇到「A 和 B 分别讲了什么」这类需要换角度的问题会漏召回；
多轮让模型可以补充检索，或对复合问题分步调用不同工具。

收敛保障：
- 调用阶段上限 `MAX_TOOL_ROUNDS = 3` 轮
- 同一工具调用签名重复 → 提前终止（防止模型空转）
- 检索类工具没带回新片段 → 提前结束检索
- 收尾轮不暴露工具 → 不可能再产出工具调用
- 跨轮片段按 `(doc_id, chunk_index)` 去重，展示与落库按相似度降序取前 8 条

### 该检索时必须检索（提示词兜底）

问法像在问助手自己时（如「平时你是怎么查看日志的？」），模型曾把它当成闲聊而**完全不检索**
——实测 0 轮检索、零引用，直接用常识作答。修法是在检索阶段的提示词里加一条「先检索再作答」规则，
并**显式豁免纯寒暄**，避免走向另一个极端（对「你好」也白调一次工具）。

实测：该问题现在会先 `knowledge_search` 再作答；「你好」「今天天气怎么样」仍保持不检索。

**实测的多工具选择**（DeepSeek，185 片段知识库）：

| 提问 | 模型选择的工具 |
| --- | --- |
| 知识库里有哪些文档？ | `list_documents` |
| README.md 这份文档主要讲了什么？ | `get_document_content`（自己补了 `filename` / `start` / `limit`） |
| 短期记忆和长期记忆分别存在哪里？ | `knowledge_search` |
| 知识库里有哪些文档？另外短期记忆存在哪里？ | **`list_documents` → `knowledge_search`（两轮两个工具）** |

之所以用「提示词 + JSON」而不是绑定某家的 Function Calling，是为了让同一份代码直接对接 DeepSeek / Ollama / 通义千问等任意 OpenAI 兼容接口。

> **性能实测**（DeepSeek + 185 片段知识库，已启用启动预热）：
>
> | 场景 | 首字延迟 | LLM 调用次数 |
> | --- | --- | --- |
> | 服务重启后的第一个请求 | **3.47s** | 3 次（2 轮工具 + 作答） |
> | 稳态（两轮检索的问题） | **2.50s** | 3 次（旧两段式方案需 4 次） |
> | 无需检索的问题（如「你好」） | **0.77s** | 1 次（旧方案需 2 次） |
>
> 注：`tool_result` 之后到首个字符仅 0.02s——**检索本身几乎不耗时间**，
> 首字延迟主要来自串行的模型往返（每轮约 0.8s）。

### 别把冷启动算进「首字延迟」

早期版本测得的首字延迟高达 **19.6s**，一度以为是多轮编排或检索太慢。加计时探针定位后发现：

```
[预热] 向量库就绪，耗时 16.22s     ← 向量模型加载 + torch 首次推理
LLM 总耗时 4.63s，其中作答前等待 1.79s
检索 5 个片段：0.02s
```

**19.6s 里有 16.6s 是向量模型的懒加载**，与检索、编排都无关。修法是在启动阶段预热
（`app/rag/store.py::warm_up()`，由 `main.py` 的 lifespan 调用），把这 16.6s 从「第一个用户的等待」
挪到「服务启动时的等待」。修复后首个请求首字 **3.47s**。

> 这个坑值得记住：**性能数字必须在稳态下测**，否则会把一次性初始化成本误判成架构问题，
> 然后在错误的方向上做优化。

## 🎯 检索相关性阈值（如何避免「噪声变幻觉」）

不设阈值时，即使问的是知识库完全无关的问题，向量库也会返回「最像的」几条低分片段，
模型拿到这些片段后会把它当作参考资料——**噪声就这样变成了新的幻觉**。

因此 `MIN_SCORE` 之下的片段一律丢弃；**全部低于阈值时，Agent 会把「未检索到相关内容」
作为 Observation 告知模型**，让它如实说明"知识库中没找到"，而不是凭空编造并伪装成来自文档。

### 阈值靠实测标定，不靠猜

仓库自带评测脚本 `backend/evals/retrieval_eval.py`：内置一份**带答案标注的评测语料**，
与用户上传的文档完全隔离，因此换台机器也能复现同一套结论。

```bash
cd backend
python -m evals.retrieval_eval                 # 用配置的向量模型
python -m evals.retrieval_eval --provider hash  # 离线降级模式，秒级完成
python -m evals.retrieval_eval --json           # 输出机器可读结果
```

脚本会分别测「正样本（知识库确有答案）」与「负样本（完全无关）」的分数分布，
扫描候选阈值，并**直接算出建议值**。实测输出：

```
正样本 Top1：min=0.5872  中位=0.7066      Recall@4 = 1.0      Hit@1 = 7/7
负样本 Top1：max=0.3278  中位=0.3140

阈值    正样本保留   负样本误收
0.30     7/7        4/6
0.35     7/7        0/6      ← 完美分离
...      7/7        0/6
0.55     7/7        0/6      ← 完美分离
0.60     6/7        0/6      ← 开始误杀正样本

建议 MIN_SCORE = 0.458   （当前配置 0.45，判定为一致）
```

两类分数在 0.3278 与 0.5872 之间有明显空隙（0.2594 宽），脚本取空隙中值作建议值。

> **换语料/换模型后请重跑**：这个阈值跟向量模型、语料、切分方式都有关。
> 另一份真实语料（185 片段的项目文档）上手工标定得到的是 0.45，与脚本在评测语料上
> 独立算出的 0.458 相符——方法在两个语料上互相印证。
>
> **反面教材**：本项目最初按直觉把阈值写成 **0.35**，实测发现有 **2/8** 的无关问题仍会漏过。

### 降级模式要单独标定

哈希向量衡量的是字符重叠度而非语义，分数整体低约 0.25，且正负样本**分数会发生重叠**
（评测脚本会如实报出「无法用单一阈值分离」）：

| | 语义向量 | 哈希降级 |
| --- | --- | --- |
| 正样本 Top1 | 0.587 ~ 0.765 | 0.113 ~ 0.644 |
| 负样本 Top1 | 0.244 ~ 0.328 | 0.030 ~ 0.130 |

沿用 0.45 会把结果全部过滤掉、让降级模式彻底失效，所以 `MIN_SCORE_HASH` 单独取 0.15，
由 `VectorStore._default_threshold()` 按当前向量化器自动选择。

## 📖 长答案为什么只答一半

一道题（题干 + 详细解析）常被切成多段。实测一份题库 PDF 共 122 段，**单题占用 2~21 段**；
只把命中的 top-k 段喂给模型，回答就会缺头少尾——而且模型未必知道自己在"漏答"。

实测提问「面试题：你平时是怎么查看日志的？请完整给出」（该题跨 5 段）：

| | 修复前 | 修复后 |
| --- | --- | --- |
| 引用片段 | 5 段（其中 2 段属于别的文档） | 8 段（含该题全部 5 段） |
| 回答长度 | 1201 字 | **1774 字** |
| 覆盖小节 | 一、二、三（**漏掉四、六**） | **一~六全覆盖** |
| 缺内容时 | 声称"以上即为完整答案" | 如实说明「该细节片段里没有展开」 |

做法：命中片段自动补齐**同一文档内**前后各 `CONTEXT_WINDOW`（默认 2）段的相邻上下文，
按 `(doc_id, chunk_index)` 去重、按 `MAX_CONTEXT_CHUNKS`（默认 12）截断后注入上下文。

细节：
- 扩展段的 `score` 置 `null`、`expanded=true`，拼 Observation 时标注「命中片段的相邻上下文」，
  避免模型把它们当成与命中段同样相关的内容；
- 扩展严格限定在同一文档内，不会把别的文档片段串进来；
- 单次检索返回的 `count` 仍是**命中片段数**（"命中了几个"），
  真正喂给模型的片段在 `references` 里（命中段 + 扩展段）。

已知边界：如果一道题正好切在片段边界上，扩展出来的相邻段会包含**上一题 / 下一题**的内容
（实测「RR 隔离级别」那题的引用里混入了相邻题目的片段，模型基本会忽略它们）。
彻底避免需要按题号切分，即 parent-document retrieval（小块检索、整题返回）。

想调整行为改 `.env`：`CONTEXT_WINDOW`（设 0 即关闭扩展）、`MAX_CONTEXT_CHUNKS`。

## 🛟 向量化降级说明

默认使用 `sentence-transformers`（`BAAI/bge-small-zh-v1.5`），首次运行会从 HuggingFace 下载模型；
`.env` 里的 `HF_ENDPOINT=https://hf-mirror.com` 已默认指向国内镜像。

若下载失败或想完全离线，把 `EMBEDDING_PROVIDER` 改为 `hash` 即可使用内置的确定性哈希向量，
检索链路（上传 → 检索 → 引用 → 流式回答）依然完整可用。`/api/health` 会显示当前实际使用的向量化器。