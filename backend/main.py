from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel

from backend.services.rag import set_case_checkpointer, stream_answer_question


# 本地开发从项目根目录读取数据库连接地址，Docker中则优先使用容器环境变量。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """在FastAPI运行期间保持PostgreSQL Checkpointer连接。"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("没有找到DATABASE_URL，请先配置.env")

    # 退出with时连接才会关闭，因此Agent运行期间始终可以读写聊天状态。
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        # 首次启动会建表，后续启动只检查已执行的数据库迁移。
        checkpointer.setup()
        set_case_checkpointer(checkpointer)
        try:
            yield
        finally:
            set_case_checkpointer(None)


# 创建类案检索API，并使用lifespan管理数据库连接。
app = FastAPI(title="中文刑事类案检索系统", lifespan=lifespan)

# Vue 开发服务器和 FastAPI 使用不同端口，浏览器需要 CORS 许可才能发送请求。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    # question 是用户问题，thread_id 用于区分不同的聊天上下文。
    question: str
    thread_id: str


@app.post("/chat")
def stream_chat(request: ChatRequest):
    """以 NDJSON 数据流逐段返回回答，并在最后返回引用案件。"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="案件事实不能为空")

    def generate_events():
        try:
            # 每个事件独占一行，前端可以边接收边解析而不必等待完整回答。
            for event in stream_answer_question(
                request.question,
                request.thread_id,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception:
            logger.exception("流式类案检索与分析执行失败")
            error_event = {
                "type": "error",
                "message": "类案检索服务暂时不可用，请稍后重试。",
            }
            yield json.dumps(error_event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
