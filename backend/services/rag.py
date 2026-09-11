"""编排类案Agent、检索与流式输出。"""

from functools import lru_cache
import re
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.messages import AIMessageChunk, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from backend.services.hybrid_retriever import retrieve_hybrid_cases
from backend.services.llm import create_chat_model
from backend.services.reranker import rerank_cases_with_hybrid_protection


# Reranker接收的Hybrid候选数量。
RERANK_CANDIDATE_K = 20


class CaseAgentState(AgentState):
    """扩展Agent状态以保存最近类案。"""

    # 新聊天允许暂时缺少检索结果。
    last_cases: NotRequired[list[dict]]


# FastAPI生命周期注入Checkpointer。
CASE_CHECKPOINTER = None


@tool
def retrieve_similar_cases(question: str, runtime: ToolRuntime) -> Command:
    """检索并保存最相似的历史刑事案件。"""
    # 发送Hybrid检索进度。
    # 发送Reranker处理进度。
    runtime.stream_writer(
        {"type": "status", "message": "正在检索相似案件……"}
    )
    candidate_cases = retrieve_hybrid_cases(
        question.strip(),
        top_k=RERANK_CANDIDATE_K,
    )

    runtime.stream_writer(
        {"type": "status", "message": "正在筛选最相关案件……"}
    )
    try:
        cases = rerank_cases_with_hybrid_protection(
            question.strip(),
            candidate_cases,
            top_k=4,
        )
    except RuntimeError:
        # Reranker失败时降级到Hybrid结果。
        runtime.stream_writer(
            {
                "type": "status",
                "message": "重排服务暂时不可用，已使用Hybrid检索结果……",
            }
        )
        cases = candidate_cases[:4]

    # 新结果覆盖当前线程的last_cases。
    return Command(
        update={
            "last_cases": cases,
            "messages": [
                ToolMessage(
                    content=f"已检索到{len(cases)}个相似案件。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# 系统提示词约束工具路由与回答边界。
AGENT_SYSTEM_PROMPT = """你是中文刑事类案检索助手，不是代替律师作出法律结论的 AI 律师。

工具使用规则：
- 用户提供新的案件事实并要求检索或比较类案时，调用 retrieve_similar_cases。
- 当前请求已经有工具返回结果时，不得重复调用工具，应使用最新类案生成回答。
- 用户追问“类案2”“上一个案件”或最新检索结果时，直接使用最新类案，不要重新检索。
- 问候、功能介绍和不需要案件资料的普通对话直接回答，不调用工具。
- 如果用户追问类案，但当前没有最新类案，请用户先提供案情进行检索。

回答类案问题时：
- 只能依据提供的案件文字，未记载的内容必须说“文书未记载”。
- 每次提到历史案件都使用【类案 N】，不得混淆不同案件的事实。
- “如实供述”不等于“认罪认罚”或“自首”，“有前科”不等于“累犯”。
- 不得给用户案情认定罪名、预测刑期或自行断言事实差异导致了某个判决结果。
- 进行完整类案比较时，使用“可比类案”“经确认的共同点”“关键差异”“法院明确关注的因素”四个标题。
- 使用纯文本，不要输出 Markdown 符号，并提醒用户结果仅供研究和学习参考。
""".strip()


def build_context(cases):
    """构建带编号的类案上下文。"""
    context_parts = []
    for index, case in enumerate(cases, start=1):
        metadata = case["metadata"]
        context_parts.append(
            "\n".join(
                [
                    (
                        f"【类案 {index}｜案件编号 {case['id']}｜"
                        f"罪名 {metadata.get('charge') or '未提供'}｜"
                        f"法条 {metadata.get('article') or '未提供'}】"
                    ),
                    f"案件事实：{case['content']}",
                    f"法院说理：{metadata.get('reason') or '未提供'}",
                    f"判决结果：{metadata.get('result') or '未提供'}",
                ]
            )
        )
    return "\n\n".join(context_parts)


def extract_cited_case_numbers(answer):
    """提取中英文数字形式的类案编号。"""
    # 宽松匹配常见类案引用格式。
    case_numbers = re.findall(r"类案\s*([一二三四五六七八九十\d]+)", answer)
    chinese_numbers = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    # 将中文序号转换为整数。
    normalized_numbers = [
        int(case_number)
        if case_number.isdigit()
        else chinese_numbers.get(case_number)
        for case_number in case_numbers
    ]
    return list(
        dict.fromkeys(
            case_number for case_number in normalized_numbers if case_number is not None
        )
    )


def select_cited_cases(cases, cited_case_numbers):
    """按引用序号选择有效案件。"""
    # 集合用于去重引用编号。
    selected_cases = []
    seen_numbers = set()
    for case_number in cited_case_numbers:
        # 跳过重复或越界编号。
        if case_number in seen_numbers or not 1 <= case_number <= len(cases):
            continue
        seen_numbers.add(case_number)
        # 类案序号转换为列表下标。
        selected_cases.append(cases[case_number - 1])
    return selected_cases


def build_sources(cases):
    """构建前端来源卡片数据。"""
    # 排除前端无需展示的内部字段。
    return [
        {
            "case_id": case["id"],
            "score": float(case["score"]),
            "fact": case["content"],
            "reason": case["metadata"].get("reason", ""),
            "result": case["metadata"].get("result", ""),
            "charge": case["metadata"].get("charge", ""),
            "article": case["metadata"].get("article", ""),
        }
        for case in cases
    ]


@dynamic_prompt
def build_agent_prompt(request: ModelRequest) -> str:
    """动态注入当前线程的最近类案。"""
    cases = request.state.get("last_cases", [])
    case_context = build_context(cases) if cases else "暂无。"
    return f"""{AGENT_SYSTEM_PROMPT}

当前聊天最近一次检索出的案件：
{case_context}"""


@lru_cache(maxsize=1)
def get_case_agent():
    """创建并缓存持久化Agent。"""
    if CASE_CHECKPOINTER is None:
        raise RuntimeError("PostgreSQL Checkpointer尚未初始化")

    # Agent共享检索工具、动态提示词和状态结构。
    return create_agent(
        model=create_chat_model(),
        tools=[retrieve_similar_cases],
        middleware=[build_agent_prompt],
        state_schema=CaseAgentState,
        checkpointer=CASE_CHECKPOINTER,
    )


def set_case_checkpointer(checkpointer):
    """更新Checkpointer并失效Agent缓存。"""
    global CASE_CHECKPOINTER
    CASE_CHECKPOINTER = checkpointer
    get_case_agent.cache_clear()


def stream_answer_question(question, thread_id):
    """按thread_id流式运行Agent。"""
    # 清理并校验用户输入。
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("案件事实不能为空")

    # thread_id绑定独立Checkpoint状态。
    agent = get_case_agent()
    config = {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }
    answer_parts = []

    # 同时接收模型消息和工具自定义事件。
    for stream_part in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": clean_question,
                }
            ]
        },
        config=config,
        stream_mode=["messages", "custom"],
        version="v2",
    ):
        # 转发工具状态事件。
        if stream_part["type"] == "custom":
            custom_event = stream_part["data"]
            if isinstance(custom_event, dict) and custom_event.get("type") == "status":
                yield custom_event
            continue

        # 仅转发模型正文片段。
        if stream_part["type"] == "messages":
            message_chunk, _metadata = stream_part["data"]
            if not isinstance(message_chunk, AIMessageChunk):
                continue
            if message_chunk.text:
                # 同步累计正文用于引用解析。
                answer_parts.append(message_chunk.text)
                yield {"type": "token", "content": message_chunk.text}

    # 根据正文引用筛选最终来源卡片。
    generated_answer = "".join(answer_parts)
    state = agent.get_state(config)
    cases = state.values.get("last_cases", [])
    cited_case_numbers = extract_cited_case_numbers(generated_answer)
    cited_cases = select_cited_cases(cases, cited_case_numbers)
    yield {"type": "sources", "sources": build_sources(cited_cases)}
