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


# 加载数据库连接配置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """管理FastAPI运行期的Checkpointer连接。"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("没有找到DATABASE_URL，请先配置.env")

    # 在应用生命周期内保持数据库连接。
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        # 初始化或校验Checkpoint表。
        checkpointer.setup()
        set_case_checkpointer(checkpointer)
        try:
            yield
        finally:
            set_case_checkpointer(None)


# 创建带数据库生命周期的FastAPI应用。
app = FastAPI(title="中文刑事类案检索系统", lifespan=lifespan)

# 允许本地Vue开发地址跨域访问。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    # thread_id标识独立聊天上下文。
    question: str
    thread_id: str


@app.post("/chat")
def stream_chat(request: ChatRequest):
    """以NDJSON流返回回答与来源。"""
    # 拒绝空问题。
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="案件事实不能为空")

    def generate_events():
        # 将Agent事件编码为NDJSON行。
        try:
            for event in stream_answer_question(
                request.question,
                request.thread_id,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception:
            # 记录内部异常并返回统一错误事件。
            logger.exception("流式类案检索与分析执行失败")
            error_event = {
                "type": "error",
                "message": "类案检索服务暂时不可用，请稍后重试。",
            }
            yield json.dumps(error_event, ensure_ascii=False) + "\n"

    # 禁用代理缓冲以保持流式响应。
    return StreamingResponse(
        generate_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
