"""TagmeAgentTracer - custom callback handler for Langchain, catching chat_model_start and llm_end events and sending them to TagMe."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .client import TagmeIntegrationClientAsync
from .entities import ChatMessage, DialogData, FunctionCall, FunctionDef

logger = logging.getLogger(__name__)


_INV_KEYS = (
    "temperature",
    "model",
    "profanity",
    "streaming",
    "max_tokens",
    "top_p",
    "repetition_penalty",
    "_type",
    "stop",
)


def _thread_id_from_kw(old_thread_id: Optional[str], **kw) -> str:
    """Extract the thread_id from callback parameters or generate a new one if not provided.

    This function checks the `configurable` and `metadata` dictionaries (if present) in the passed keyword arguments for a 'thread_id'. If found, it returns that thread_id; otherwise, it generates a new UUID string.
    """
    logger.debug("_thread_id_from_kw called with args: %s", kw)
    cfgb = kw.get("configurable") or {}
    meta = kw.get("metadata") or {}
    trace_id = cfgb.get("thread_id") or meta.get("thread_id")
    if trace_id:
        logger.debug("Using provided thread_id: %s", trace_id)
        return trace_id
    if old_thread_id:
        return old_thread_id
    trace_id = str(uuid.uuid4())
    logger.debug("No thread_id provided. Generated new thread_id: %s", trace_id)
    return trace_id


def _extract_functions(invocation_params: Dict[str, Any]) -> List[FunctionDef]:
    """Extract function definitions from invocation parameters.

    Iterates over the 'tools' list in `invocation_params`. For each tool, retrieves its 'function' details and constructs a FunctionDef object. Returns a list of FunctionDef objects.
    """
    out: List[FunctionDef] = []
    for t in (invocation_params or {}).get("tools", []):
        f = t.get("function") or {}
        out.append(
            FunctionDef(
                name=str(f.get("name")),
                description=f.get("description"),
                parameters=f.get("parameters"),
                return_parameters=f.get("return_parameters"),
                few_shot_examples=f.get("few_shot_examples"),
            )
        )
    return out


def _ensure_content_str(content: Union[str, List[Union[str, dict]], None]) -> str:
    """Convert a BaseMessage content field to a safe string representation.

    Handles various content types:
    - None: returns an empty string.
    - str: returns the string as is.
    - list of str/dict: concatenates items into one string; dict items are converted to JSON string.
    - other types: uses str() to convert to string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except (TypeError, ValueError):
                    logger.warning("Failed to serialize dict item: %s", item)
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _msg_to_chat_messages(m: BaseMessage) -> List[ChatMessage]:
    """Convert one BaseMessage into one or multiple ChatMessage objects.

    If the message contains tool call instructions (e.g., an AIMessage with pending function calls), this will produce multiple ChatMessage entries (one for each function call and one for the assistant's content). Otherwise, it produces a single ChatMessage mirroring the original message.
    """
    if isinstance(m, HumanMessage):
        return [ChatMessage(role="user", content=_ensure_content_str(m.content))]
    if isinstance(m, SystemMessage):
        return [ChatMessage(role="system", content=_ensure_content_str(m.content))]
    if isinstance(m, ToolMessage):
        return [ChatMessage(role="function", content=_ensure_content_str(m.content))]
    if isinstance(m, AIMessage):
        res: List[ChatMessage] = []
        for tc in getattr(m, "tool_calls", None) or []:
            res.append(
                ChatMessage(
                    role="assistant",
                    function_call=FunctionCall(name=tc.get("name"), arguments=(tc.get("args") or {})),
                )
            )
        if (m.content or "") != "":
            res.append(ChatMessage(role="assistant", content=_ensure_content_str(m.content)))
        return res

    return [ChatMessage(role="assistant", content=_ensure_content_str(getattr(m, "content", "")))]


def _batch_msgs_to_chat(messages_batch: List[BaseMessage]) -> List[ChatMessage]:
    """Convert a list of BaseMessage objects into a flat list of ChatMessage objects.

    Processes each BaseMessage via _msg_to_chat_messages and concatenates the results.
    """
    out: List[ChatMessage] = []
    for m in messages_batch:
        out.extend(_msg_to_chat_messages(m))
    return out


class TagMeAgentTracer(AsyncCallbackHandler):
    """Asynchronous callback handler that traces conversation dialogs and tool usage.

    This tracer collects messages exchanged in a conversation (including user, system, assistant, and function calls) along with tool definitions and relevant metadata. It can flush (save) the conversation and reset when a conversation is completed or a new one starts.

    Attributes:
        functions (List[FunctionDef]): List of tool function definitions available in the current conversation context.
        dialog (List[ChatMessage]): Accumulated messages in the current conversation.
        metadata (Dict[str, Any]): Metadata for the current conversation (e.g., invocation parameters and thread_id).

    Behavior:
        - on_chat_model_start: If a new thread_id is detected, flushes the previous conversation and initializes a new one. Updates functions, dialog, and metadata for the new conversation.
        - on_llm_end: After the LLM generates a response, appends the assistant's message(s) (function calls and content) to the dialog.
        - flush: Can be called to persist the conversation data to disk and clear internal state.
    """

    def __init__(
        self, *args, verbose_send: bool = False, client: Optional[TagmeIntegrationClientAsync] = None, **kwargs
    ) -> None:
        assert not (client and (args or kwargs)), "You must provide either a client or args/kwargs, not both."
        self.client = client or TagmeIntegrationClientAsync(*args, **kwargs)
        self.verbose_send = verbose_send
        self._thread_id: Optional[str] = None
        self.functions: List[FunctionDef] = []
        self.dialog: List[ChatMessage] = []
        self.metadata: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def flush(self, reason: str = "manual") -> None:
        """Save the current conversation data to a JSON file and reset the tracer's state.

        Collects the current dialog, function definitions, and metadata into a dictionary and writes it to a JSON file in the `captures` directory. The filename includes the thread_id and a random suffix. After saving, resets internal state for a new conversation.

        Args:
            reason (str): Reason for flushing (e.g., 'manual' or 'thread_switch'), recorded in the metadata.
        """
        logger.debug("Flush called (reason=%s)", reason)
        async with self._lock:
            if not self.functions and not self.dialog and not self.metadata:
                logger.debug("No conversation data to flush; nothing was saved.")
                return None

            tid = self._thread_id
            funcs_snapshot = list(self.functions)
            dialog_snapshot = list(self.dialog)
            meta_snapshot = self.metadata | {"flush_reason": reason, "ts": time.time()}

            self.reset_cache()

        if self.verbose_send:
            logger_level_func = logger.info
        else:
            logger_level_func = logger.debug

        if funcs_snapshot:
            logger_level_func("Sending %d function definitions (thread_id=%s)", len(funcs_snapshot), tid or "N/A")
            await self.client.send_functions(funcs_snapshot)

        if dialog_snapshot:
            logger_level_func("Sending %d chat messages (thread_id=%s)", len(dialog_snapshot), tid or "N/A")
            await self.client.send_dialog(DialogData(dialog_snapshot, meta_snapshot))

    def reset_cache(self) -> None:
        """Clear the stored conversation data and reset the tracer.

        Empties the current dialog list, functions list, metadata, and thread_id, preparing the tracer for a new conversation.
        """
        self._thread_id = None
        self.functions = []
        self.dialog = []
        self.metadata = {}

    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kw: Any
    ) -> None:
        """Process the start of a new LLM chat invocation.

        If a new `thread_id` is provided that differs from the current one, the tracer flushes the existing conversation (marking its end) before starting to track a new conversation.

        Sets up internal state for the new conversation by recording initial messages, tool definitions, and capturing relevant invocation parameters (filtered to a safe subset).
        """
        logger.debug("on_chat_model_start called.")
        tid = _thread_id_from_kw(self._thread_id, **kw)

        need_flush = False
        async with self._lock:
            if (
                self._thread_id is not None
                and self._thread_id != tid
                and (self.dialog or self.functions or self.metadata)
            ):
                need_flush = True

        if need_flush:
            logger.debug("Thread id changed from %s to %s, flushing previous conversation.", self._thread_id, tid)
            await self.flush(reason="thread_switch")

        inv: dict[str, Any] = kw.get("invocation_params", {})
        meta: dict[str, Any] = kw.get("metadata", {})
        inv_filtered: Dict[str, Any] = {k: inv.get(k) for k in _INV_KEYS if k in inv}
        funcs: List[FunctionDef] = _extract_functions(inv)
        batch: List[BaseMessage] = messages[0] if messages else []
        dialog: List[ChatMessage] = _batch_msgs_to_chat(batch)

        async with self._lock:
            logger.debug("Initialized tracing for thread_id %s. Captured %d prompt message(s).", tid, len(self.dialog))
            logger.debug("Registered %d tool function(s): %s", len(self.functions), [f.name for f in self.functions])
            self._thread_id = tid
            self.functions = funcs
            self.dialog = dialog
            self.metadata = {"thread_id": tid, "invocation_params": inv_filtered} | meta

    async def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        """Process the end of an LLM response generation.

        Appends the assistant's response to the dialog history. If the assistant invoked any function calls (tools) as part of its response, those are added as separate messages (with role 'assistant' and function_call data) before the final answer content.
        """
        logger.debug("on_llm_end called.")
        async with self._lock:
            if self._thread_id is None:
                return
            try:
                gen = response.generations[0][0]
            except (IndexError, TypeError, AttributeError):
                logger.warning("Unexpected on_llm_end response format: need at least one generation")
                return
            msg = getattr(gen, "message", None)
            if msg is None:
                return

            new_messages = _msg_to_chat_messages(msg)
            self.dialog.extend(new_messages)
            calls = [
                cm.function_call.name for cm in new_messages if getattr(cm, "function_call", None) and cm.function_call
            ]
            final_answer = any((cm.role == "assistant" and cm.content) for cm in new_messages)
            if calls:
                logger.debug("Assistant invoked function call(s): %s", calls)
            if final_answer:
                logger.debug("Assistant produced a final answer message.")
            logger.debug("Appended %d new message(s) to dialog.", len(new_messages))
