"""组织类案检索、案件对比、模型回答和案件来源。"""

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


# Hybrid 先广泛召回二十件案件，保护头部结果后再由Reranker排列其余类案。
RERANK_CANDIDATE_K = 20


class CaseAgentState(AgentState):
    """保存聊天消息，并用 last_cases 记住最近一次检索的四个案件。"""

    # 新聊天可能还没有检索过案件，因此该字段可以暂时不存在。
    last_cases: NotRequired[list[dict]]


# FastAPI启动时会把保持连接的PostgreSQL Checkpointer放到这里。
CASE_CHECKPOINTER = None


@tool
def retrieve_similar_cases(question: str, runtime: ToolRuntime) -> Command:
    """当用户提供新的案件事实并要求查找类案时，检索最相似的历史刑事案件。"""
    # Tool 内的进度信息会在后续由 Agent 流式入口转发给前端。
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
        # Reranker 不可用时保留 Hybrid 前四名，避免整个 Agent 中断。
        runtime.stream_writer(
            {
                "type": "status",
                "message": "重排服务暂时不可用，已使用Hybrid检索结果……",
            }
        )
        cases = candidate_cases[:4]

    # last_cases 没有追加规则，因此新的四个案件会覆盖上一次结果。
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


# Agent 根据用户当前问题和历史消息决定是否需要调用类案检索工具。
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
    """把检索结果的案件事实、说理和结果直接拼接成模型上下文。"""
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
    """提取类案引用，并兼容模型偶尔输出的中文数字或缺失方括号。"""
    # 宽松读取引用可以避免“类案一”之类的格式偏差导致来源卡片全部丢失。
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
    """把模型返回的类案序号转换回真实案件，并忽略重复或越界序号。"""
    # seen_numbers 用于避免同一案件在前端来源中重复出现。
    selected_cases = []
    seen_numbers = set()
    for case_number in cited_case_numbers:
        # 重复编号和超出候选范围的编号都不会进入最终来源列表。
        if case_number in seen_numbers or not 1 <= case_number <= len(cases):
            continue
        seen_numbers.add(case_number)
        # 模型编号从 1 开始，而 Python 列表下标从 0 开始。
        selected_cases.append(cases[case_number - 1])
    return selected_cases


def build_sources(cases):
    """整理前端案件卡片需要的编号、事实、说理和判决结果。"""
    # 这里只返回前端需要的字段，隐藏内部使用的排名信息。
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
    """在每次调用模型前，把当前聊天最近检索的类案加入系统提示词。"""
    cases = request.state.get("last_cases", [])
    case_context = build_context(cases) if cases else "暂无。"
    return f"""{AGENT_SYSTEM_PROMPT}

当前聊天最近一次检索出的案件：
{case_context}"""


@lru_cache(maxsize=1)
def get_case_agent():
    """使用FastAPI启动时提供的PostgreSQL Checkpointer创建并复用Agent。"""
    if CASE_CHECKPOINTER is None:
        raise RuntimeError("PostgreSQL Checkpointer尚未初始化")

    return create_agent(
        model=create_chat_model(),
        tools=[retrieve_similar_cases],
        middleware=[build_agent_prompt],
        state_schema=CaseAgentState,
        checkpointer=CASE_CHECKPOINTER,
    )


def set_case_checkpointer(checkpointer):
    """保存FastAPI生命周期内的Checkpointer，并清除旧Agent缓存。"""
    global CASE_CHECKPOINTER
    CASE_CHECKPOINTER = checkpointer
    get_case_agent.cache_clear()


def stream_answer_question(question, thread_id):
    """按 thread_id 运行 Agent，并把 Agent 事件转换为前端能读取的数据流。"""
    # 用户输入在进入路由和检索之前先清理并检查空值。
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("案件事实不能为空")

    agent = get_case_agent()
    config = {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }
    answer_parts = []

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
        # Tool 通过 custom 模式发送检索和重排进度。
        if stream_part["type"] == "custom":
            custom_event = stream_part["data"]
            if isinstance(custom_event, dict) and custom_event.get("type") == "status":
                yield custom_event
            continue

        # messages 模式同时会产生工具调用片段，这里只转发可见正文。
        if stream_part["type"] == "messages":
            message_chunk, _metadata = stream_part["data"]
            if not isinstance(message_chunk, AIMessageChunk):
                continue
            if message_chunk.text:
                # 正文一份发给前端，另一份留给后端判断引用了哪些类案。
                answer_parts.append(message_chunk.text)
                yield {"type": "token", "content": message_chunk.text}

    # Agent 完成回答后，只把正文实际提到的最新类案作为来源卡片返回。
    generated_answer = "".join(answer_parts)
    state = agent.get_state(config)
    cases = state.values.get("last_cases", [])
    cited_case_numbers = extract_cited_case_numbers(generated_answer)
    cited_cases = select_cited_cases(cases, cited_case_numbers)
    yield {"type": "sources", "sources": build_sources(cited_cases)}
