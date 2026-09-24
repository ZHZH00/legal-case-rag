# 中文刑事类案检索 RAG 系统

输入一段中文刑事案件事实，系统从 LeCaRDv2 历史案件中检索事实与关键情节相似的案例，并比较案件事实、法院说理和判决结果。

本项目定位为“类案检索与对比”，不是自动判断罪名或预测刑期的 AI 律师。系统输出仅供学习和研究参考，不构成法律意见。

## 核心功能

- 使用 Chroma 完成中文案件事实的 Dense 语义检索。
- 使用 jieba、BM25 完成关键词检索，并将索引持久化到本地。
- 使用 RRF 融合 Dense 与 BM25 排名。
- 从 Hybrid Top 20 候选中使用专用 Reranker 重排。
- 固定保护 Hybrid 前两名，降低重排破坏高质量结果的风险。
- Reranker 不可用时自动降级为 Hybrid 结果，不中断问答。
- 使用 LangChain Prompt、DeepSeek 模型生成最多四个类案的对比分析。
- 使用 LangChain Agent 判断是否需要调用类案检索工具。
- 使用 LangGraph PostgresSaver 按 `thread_id` 持久化多轮聊天状态。
- 使用 NDJSON 流式返回状态、回答片段和引用案件。
- Vue 3 前端展示回答、案件事实、法院说理和判决结果。
- 使用 localStorage 保存浏览器中的聊天记录。
- 使用 Docker Compose、Nginx 运行完整前后端项目。
- 使用 LeCaRDv2 专家相关性标签计算 Precision、Recall、HitRate、NDCG 和 MRR。

## 系统流程

```mermaid
flowchart TD
    A[用户输入] --> B{Agent是否需要检索}
    B -- 否 --> C[DeepSeek结合聊天上下文直接回答]
    B -- 是 --> D[Agent调用类案检索工具]
    D --> E[Dense语义检索]
    E --> F[BM25关键词检索]
    F --> G[RRF Hybrid融合]
    G --> H[Top 20候选案件]
    H --> I[Reranker重排]
    I --> J[固定保护Hybrid前2名并选出Top 4]
    J --> K[更新当前线程的last_cases]
    K --> L[DeepSeek结合类案与聊天上下文生成分析]
    C --> M[NDJSON流式返回]
    L --> M
    M --> N[Vue展示回答与来源卡片]
```

检索阶段只对 `fact` 生成向量。案件命中后，再从 Chroma Metadata 中读取 `reason`、`result`、`charge` 和 `article` 提供给大模型及前端。

## 技术栈

### 后端与检索

- Python 3.11
- FastAPI + Uvicorn
- LangChain
- LangGraph
- PostgreSQL
- Chroma
- jieba + rank-bm25
- RRF Hybrid Retrieval
- Qwen3 Embedding 8B，1024维
- Qwen3 Reranker 8B
- DeepSeek V4 Flash 0731
- OpenRouter API

### 前端与部署

- Vue 3
- Pinia
- Vite
- Nginx
- Docker + Docker Compose

## 数据集

项目使用 [LeCaRDv2](https://github.com/THUIR/LeCaRDv2) 中文法律类案检索数据集。

运行系统所需的候选案件结构大致如下：

```json
{
  "pid": 123,
  "fact": "经审理查明，被告人……",
  "reason": "本院认为，被告人……",
  "result": "判决如下……",
  "charge": ["交通肇事罪"],
  "article": [133, 67, 72]
}
```

数据目录约定：

```text
backend/data/
├── lecardv2/
│   ├── candidate_55192/    候选案件JSON
│   ├── query/              评测查询
│   └── label/              专家相关性标签
├── chroma/                 Dense向量与案件Metadata
└── bm25/
    └── legal_cases_bm25.pkl
```

`backend/data` 不会提交到 Git，也不会复制进当前 Docker 镜像。使用者需要按照 LeCaRDv2 官方说明下载数据并建立本地索引。公开分发数据前，请同时遵守原数据集的许可和使用要求。

## 配置 API Key

复制环境变量示例：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```env
OPENROUTER_API_KEY=你的OpenRouter_API_Key
POSTGRES_USER=postgres
POSTGRES_PASSWORD=你的数据库密码
POSTGRES_DB=langgraph
DATABASE_URL=postgresql://postgres:你的数据库密码@127.0.0.1:5432/langgraph?sslmode=disable
```



## 准备案件索引

将候选案件解压到：

```text
backend/data/lecardv2/candidate_55192
```

只导入100条验证流程：

```powershell
python -m backend.scripts.build_knowledge_base --limit 100 --batch-size 32
```

导入全部55,192条：

```powershell
python -m backend.scripts.build_knowledge_base --limit 55192 --batch-size 32
```

完成Chroma入库后建立BM25索引：

```powershell
python -m backend.scripts.build_keyword_index
```

如果Chroma中已经存在相同案件ID，入库脚本会跳过已有案件。

## Docker启动

要求已经安装并启动 Docker Desktop，并且已经准备好 `backend/data/chroma` 和 `backend/data/bm25`。

第一次构建并启动：

```powershell
docker compose up -d --build
```

Compose 会自动创建 PostgreSQL 数据卷，FastAPI 会在启动时检查并初始化 Checkpointer 数据表，不需要另外执行建表命令。

没有修改代码时再次启动：

```powershell
docker compose up -d
```

浏览器访问：

```text
http://localhost:5173
```

建议始终使用同一个地址访问。`localhost` 与 `127.0.0.1` 属于不同浏览器来源，localStorage 中的聊天列表不会共用。

查看容器状态和日志：

```powershell
docker compose ps
docker compose logs -f
```

停止项目：

```powershell
docker compose stop
```

Docker运行结构：

```text
浏览器 :5173
    ↓
frontend容器中的Nginx :80
    ├── 返回Vue页面
    └── /api/chat → backend:8000/chat
                         ↓
                    FastAPI容器
                         ↓
              Chroma + BM25 + OpenRouter
```

Nginx关闭了代理缓存，并将后端读取超时设置为600秒，以支持耗时较长的RAG流式请求。

## 本地开发启动

本地开发前先停止占用5173和8000端口的Docker容器：

```powershell
docker compose stop
```

只启动 PostgreSQL：

```powershell
docker compose up -d postgres
```

启动后端：

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload
```

在另一个终端启动前端：

```powershell
npm install
npm run dev
```

Vite会在开发模式下把 `/api/chat` 转发到 `http://127.0.0.1:8000/chat`。

## API

### `POST /chat`

请求：

```json
{
  "question": "被告人驾驶小型轿车转弯时与电动三轮车相撞，造成一人死亡……",
  "thread_id": "由前端为当前聊天生成的稳定ID"
}
```

接口使用NDJSON逐行返回事件：

```json
{"type": "status", "message": "正在检索相似案件……"}
{"type": "token", "content": "可比类案……"}
{"type": "sources", "sources": []}
```

前端通过Nginx访问 `/api/chat`，Nginx会将请求转发到后端的 `/chat`。

## 检索评测

项目使用LeCaRDv2测试查询和专家相关性标签评测检索结果。

评测按照真实检索流程分为两层：Hybrid候选召回阶段使用Recall@20和HitRate@20，判断相关案件能否进入Reranker候选池；最终展示阶段使用Precision@4、NDCG@4和MRR@4评价返回给用户的四个案件，并保留Recall@4作为辅助指标。

完整160条测试查询的Top 20候选召回结果：

| 方法 | Precision@20 | Recall@20 | HitRate@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | **0.3484** | **0.2664** | **0.9062** | **0.4044** | 0.7025 |
| Dense | 0.2453 | 0.1970 | 0.8688 | 0.3004 | 0.5913 |
| Hybrid | 0.3275 | 0.2566 | 0.8875 | 0.3923 | **0.7236** |

Reranker受外部接口速率限制，因此策略比较使用20条查询的Top 4结果：

| 方法 | Precision@4 | Recall@4 | HitRate@4 | NDCG@4 | MRR@4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid直接取前4名 | 0.4875 | 0.0734 | 0.8000 | **0.4679** | 0.6667 |
| Reranker完全重排 | **0.5000** | **0.0904** | 0.8000 | 0.4327 | 0.6083 |
| 固定保护Hybrid前2名 | 0.4875 | 0.0739 | **0.8500** | 0.4608 | **0.6833** |

实验表明，完全重排提高了Top 4的Precision和Recall，但可能降低头部排序质量。固定保护Hybrid前两名后，HitRate@4由0.8000提升至0.8500，MRR@4由0.6667提升至0.6833，同时保持Precision基本不变，因此生产流程采用固定头部保护策略。

运行完整BM25、Dense和Hybrid评测：

```powershell
python -m backend.scripts.evaluate_retrieval --method bm25 --split test --top-k 20 --limit 0
python -m backend.scripts.evaluate_retrieval --method vector --split test --top-k 20 --limit 0
python -m backend.scripts.evaluate_retrieval --method hybrid --split test --top-k 20 --limit 0
```

评测Hybrid前四名、Reranker完全重排和固定保护前两名时，三种方案会共用同一次Reranker结果并输出一张对比表：

```powershell
python -m backend.scripts.evaluate_retrieval --method rerank_ablation --split test --top-k 4 --limit 20
```

该命令会把每条查询的Hybrid候选顺序和Reranker原始分数保存到断点文件。再次运行相同命令会复用已有结果，不会为已完成的查询重复请求Reranker。

`--limit 0` 表示评测所选数据集中的全部查询。Reranker评测会调用外部接口，执行前需要考虑费用和速率限制。

## 项目结构

```text
.
├── backend/
│   ├── main.py                    FastAPI流式接口
│   ├── services/
│   │   ├── knowledge_base.py      读取并校验案件JSON
│   │   ├── vector_store.py        Chroma读写
│   │   ├── embedding.py           文档与查询向量
│   │   ├── retriever.py           Dense检索
│   │   ├── keyword_retriever.py   BM25索引与检索
│   │   ├── hybrid_retriever.py    RRF融合
│   │   ├── reranker.py            重排、过滤与头部保护
│   │   ├── rag.py                 Agent、RAG工具和流式事件
│   │   └── llm.py                 对话模型配置
│   └── scripts/
│       ├── build_knowledge_base.py
│       ├── build_keyword_index.py
│       └── evaluate_retrieval.py
├── src/                            Vue前端
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── nginx.conf
└── README.md
```

## 已知限制

- 仅使用公开历史案件进行相似性检索，不保证覆盖所有真实裁判文书。
- 类案相似不代表用户案件会得到相同裁判结果。
- Embedding、Reranker和回答生成依赖外部模型服务，速度会受到网络、额度和限流影响。
- 前端聊天列表保存在当前浏览器的localStorage中，后端Agent上下文按 `thread_id` 保存在PostgreSQL中；当前没有用户账号和跨设备聊天列表同步。
- 当前Docker镜像不包含案件数据，新环境仍需准备Chroma和BM25索引。
- 当前项目适合作品展示和学习研究；公开部署前还需要增加身份认证、访问限流、监控与密钥管理。
