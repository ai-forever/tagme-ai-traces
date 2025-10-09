# pylint: disable=protected-access
import uuid
from typing import List, Optional
from unittest.mock import AsyncMock

import pytest
from langchain.schema import ChatGeneration, LLMResult
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.tagme_ai_traces.tracer import TagMeAgentTracer


def _make_llm_result_from_ai(ai: AIMessage) -> LLMResult:
    gen = ChatGeneration(message=ai, generation_info={"finish_reason": "stop"})
    return LLMResult(generations=[[gen]], llm_output={"model_name": "dummy"})


def _invocation_params_with_tools(extra: Optional[dict] = None) -> dict:
    base = {
        "model": "GigaChat-2-Max",
        "temperature": None,
        "profanity": None,
        "streaming": False,
        "max_tokens": None,
        "top_p": None,
        "repetition_penalty": None,
        "_type": "giga-chat-model",
        "stop": None,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_phone_data_by_name",
                    "description": "Return phone details by exact name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "return_parameters": None,
                    "few_shot_examples": None,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_order",
                    "description": "Create a new order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "phone": {"type": "string"}},
                        "required": ["name", "phone"],
                    },
                    "return_parameters": None,
                    "few_shot_examples": None,
                },
            },
        ],
    }
    if extra:
        base.update(extra)
    return base


def create_tracer(*args, **kwargs) -> TagMeAgentTracer:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TAGME_TOKEN", "dummy")
        tracer = TagMeAgentTracer(*args, **kwargs)
        tracer.client = AsyncMock()
        return tracer


@pytest.mark.asyncio
async def test_chat_model_start_initializes_state_and_functions_and_metadata():
    tracer = create_tracer()
    tid = str(uuid.uuid4())
    msgs: List[BaseMessage] = [
        SystemMessage(content="sys"),
        HumanMessage(content="hello"),
    ]

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[msgs],
        metadata={"thread_id": tid},
        invocation_params=_invocation_params_with_tools(extra={"IGNORED_KEY": "x"}),
    )

    assert tracer._thread_id == tid
    assert len(tracer.dialog) == 2
    assert tracer.dialog[0].role == "system" and tracer.dialog[0].content == "sys"
    assert tracer.dialog[1].role == "user" and tracer.dialog[1].content == "hello"

    assert {f.name for f in tracer.functions} == {"get_phone_data_by_name", "create_order"}

    inv = tracer.metadata["invocation_params"]

    assert set(inv.keys()) <= {
        "temperature",
        "model",
        "profanity",
        "streaming",
        "max_tokens",
        "top_p",
        "repetition_penalty",
        "_type",
        "stop",
    }
    assert inv["model"] == "GigaChat-2-Max"


@pytest.mark.asyncio
async def test_on_llm_end_appends_calls_and_final_answer_with_ids():
    tracer = create_tracer()
    tid = str(uuid.uuid4())

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="start")]],
        metadata={"thread_id": tid},
        invocation_params=_invocation_params_with_tools(),
    )

    ai = AIMessage(
        content="Done.",
        tool_calls=[
            {"name": "get_phone_data_by_name", "args": {"name": "LG Velvet 2"}, "id": "tc1"},
            {"name": "create_order", "args": {"name": "LG Velvet 2", "phone": "+7999"}, "id": "tc2"},
        ],
    )
    await tracer.on_llm_end(_make_llm_result_from_ai(ai))

    assert tracer.dialog[0].role == "user"
    assert (
        tracer.dialog[-3].role == "assistant"
        and tracer.dialog[-3].function_call
        and tracer.dialog[-3].function_call.name == "get_phone_data_by_name"
    )
    assert (
        tracer.dialog[-2].role == "assistant"
        and tracer.dialog[-2].function_call
        and tracer.dialog[-2].function_call.name == "create_order"
    )
    assert tracer.dialog[-1].role == "assistant" and tracer.dialog[-1].content == "Done."


@pytest.mark.asyncio
async def test_on_llm_end_ignores_empty_generations():
    tracer = create_tracer()
    tid = str(uuid.uuid4())

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="q")]],
        metadata={"thread_id": tid},
        invocation_params=_invocation_params_with_tools(),
    )

    await tracer.on_llm_end(LLMResult(generations=[], llm_output={}))
    assert len(tracer.dialog) == 1 and tracer.dialog[0].role == "user"


@pytest.mark.asyncio
async def test_tool_message_conversion_in_batch():
    tracer = create_tracer()
    tid = str(uuid.uuid4())

    tool_msg = ToolMessage(
        content='{"status":"ok"}',
        name="get_phone_data_by_name",
        tool_call_id="tc1",
    )

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="q"), tool_msg]],
        metadata={"thread_id": tid},
        invocation_params=_invocation_params_with_tools(),
    )

    assert tracer.dialog[0].role == "user"
    assert tracer.dialog[1].role == "function"
    assert tracer.dialog[1].content == '{"status":"ok"}'


@pytest.mark.asyncio
async def test_content_list_conversion_in_batch():
    tracer = create_tracer()
    tid = str(uuid.uuid4())

    hm = HumanMessage(content=["Hello ", {"a": 1}])

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[hm]],
        metadata={"thread_id": tid},
        invocation_params=_invocation_params_with_tools(),
    )

    # dict часть сериализована в JSON и склеена
    assert tracer.dialog[0].role == "user"
    assert tracer.dialog[0].content and tracer.dialog[0].content.replace(" ", "") in ('Hello{"a":1}', 'Hello{"a":1}')


@pytest.mark.asyncio
async def test_configurable_thread_id_has_priority_over_metadata():
    tracer = create_tracer()
    tid_cfg = str(uuid.uuid4())
    tid_meta = str(uuid.uuid4())

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="hi")]],
        configurable={"thread_id": tid_cfg},
        metadata={"thread_id": tid_meta},
        invocation_params=_invocation_params_with_tools(),
    )

    assert tracer._thread_id == tid_cfg
    assert tracer.metadata["thread_id"] == tid_cfg


@pytest.mark.asyncio
async def test_thread_switch_triggers_single_flush():
    tracer = create_tracer()
    tracer.flush = AsyncMock(return_value="mocked")  # не пишем в файл, просто проверяем вызов

    tid1 = str(uuid.uuid4())
    tid2 = str(uuid.uuid4())

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="hi")]],
        metadata={"thread_id": tid1},
        invocation_params=_invocation_params_with_tools(),
    )
    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="new")]],
        metadata={"thread_id": tid2},
        invocation_params=_invocation_params_with_tools(),
    )

    tracer.flush.assert_awaited_once()
    assert tracer.flush.await_args
    _, kwargs = tracer.flush.await_args
    assert kwargs.get("reason") == "thread_switch"
    assert tracer._thread_id == tid2
    assert len(tracer.dialog) == 1 and tracer.dialog[0].content == "new"


@pytest.mark.asyncio
async def test_same_thread_does_not_flush_and_overwrites_dialog_and_functions():
    tracer = create_tracer()
    tracer.flush = AsyncMock()

    tid = str(uuid.uuid4())

    inv1 = _invocation_params_with_tools()
    inv2 = _invocation_params_with_tools()

    inv2["tools"][0]["function"]["name"] = "get_phone_data_by_name_v2"

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="first")]],
        metadata={"thread_id": tid},
        invocation_params=inv1,
    )

    await tracer.on_chat_model_start(
        serialized={"name": "GigaChat"},
        messages=[[HumanMessage(content="second")]],
        metadata={"thread_id": tid},
        invocation_params=inv2,
    )

    tracer.flush.assert_not_called()
    assert len(tracer.dialog) == 1 and tracer.dialog[0].content == "second"
    assert {f.name for f in tracer.functions} == {"get_phone_data_by_name_v2", "create_order"}
