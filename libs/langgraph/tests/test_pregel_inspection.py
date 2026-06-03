from __future__ import annotations

import json
import operator
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel._inspection import (
    DebugStep,
    DebugTrace,
    StateDiff,
    ToolCallSummary,
    collect_debug_trace,
)


class LinearState(TypedDict):
    values: Annotated[list[str], operator.add]


class RouteState(TypedDict):
    which: str
    log: Annotated[list[str], operator.add]


def _linear_graph() -> Any:
    def one(state: LinearState) -> LinearState:
        return {"values": ["one"]}

    def two(state: LinearState) -> LinearState:
        return {"values": ["two"]}

    builder = StateGraph(LinearState)
    builder.add_node("one", one)
    builder.add_node("two", two)
    builder.add_edge(START, "one")
    builder.add_edge("one", "two")
    builder.add_edge("two", END)
    return builder.compile()


def test_linear_execution_captures_ordered_steps() -> None:
    trace = collect_debug_trace(_linear_graph(), {"values": []})

    assert isinstance(trace, DebugTrace)
    node_steps = [step for step in trace.steps if step.name in {"one", "two"}]
    assert [step.name for step in node_steps] == ["one", "two"]

    for step in node_steps:
        assert isinstance(step, DebugStep)
        assert step.status == "completed"
        assert isinstance(step.diff, StateDiff)
        assert "values" in step.diff.changed_keys

    text = trace.to_text()
    assert "LangGraph debug trace" in text
    assert "one" in text
    assert "two" in text
    assert trace.status == "finished"


def test_to_json_is_stable_and_parseable() -> None:
    trace = collect_debug_trace(_linear_graph(), {"values": []})

    payload = json.loads(trace.to_json())
    assert set(payload) >= {"status", "thread_id", "checkpoint_ids", "steps"}

    step = next(s for s in payload["steps"] if s["name"] == "one")
    assert set(step) >= {
        "step",
        "name",
        "namespace",
        "status",
        "diff",
        "tool_calls",
        "error",
        "interrupts",
        "next",
    }
    assert step["status"] == "completed"
    assert step["diff"]["changed_keys"] == ["values"]


def test_conditional_routing_records_only_taken_branch() -> None:
    def start(state: RouteState) -> RouteState:
        return {"log": ["start"]}

    def left(state: RouteState) -> RouteState:
        return {"log": ["left"]}

    def right(state: RouteState) -> RouteState:
        return {"log": ["right"]}

    def route(state: RouteState) -> str:
        return state["which"]

    builder = StateGraph(RouteState)
    builder.add_node("start", start)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", route, {"left": "left", "right": "right"})
    builder.add_edge("left", END)
    builder.add_edge("right", END)
    graph = builder.compile()

    trace = collect_debug_trace(graph, {"which": "left", "log": []})
    names = [step.name for step in trace.steps]
    assert "left" in names
    assert "right" not in names


def test_error_path_marks_step_and_trace_failed() -> None:
    class ErrState(TypedDict):
        values: Annotated[list[str], operator.add]

    def boom(state: ErrState) -> ErrState:
        raise ValueError("kaboom")

    builder = StateGraph(ErrState)
    builder.add_node("boom", boom)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    graph = builder.compile()

    trace = collect_debug_trace(graph, {"values": []})

    assert trace.status == "error"
    assert trace.error is not None
    assert "kaboom" in trace.error
    failed = [step for step in trace.steps if step.status == "error"]
    assert failed and failed[0].name == "boom"
    assert "kaboom" in (failed[0].error or "")
    assert "kaboom" in trace.to_text()


def test_checkpointed_run_captures_thread_and_checkpoints() -> None:
    graph = _linear_graph_with_saver()
    config = {"configurable": {"thread_id": "thread-42"}}

    trace = collect_debug_trace(graph, {"values": []}, config)

    assert trace.thread_id == "thread-42"
    assert trace.checkpoint_ids
    assert all(isinstance(cid, str) for cid in trace.checkpoint_ids)


def _linear_graph_with_saver() -> Any:
    def one(state: LinearState) -> LinearState:
        return {"values": ["one"]}

    builder = StateGraph(LinearState)
    builder.add_node("one", one)
    builder.add_edge(START, "one")
    builder.add_edge("one", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_redaction_hides_sensitive_values() -> None:
    class SecretState(TypedDict):
        password: str
        custom_private: str
        visible: str

    def emit(state: SecretState) -> SecretState:
        return {
            "password": "topsecret",
            "custom_private": "hideme",
            "visible": "public",
        }

    builder = StateGraph(SecretState)
    builder.add_node("emit", emit)
    builder.add_edge(START, "emit")
    builder.add_edge("emit", END)
    graph = builder.compile()

    trace = collect_debug_trace(
        graph,
        {"password": "", "custom_private": "", "visible": ""},
        redact_keys=("custom_private",),
    )

    text = trace.to_text()
    assert "topsecret" not in text
    assert "hideme" not in text
    assert "public" in text

    step = next(s for s in trace.steps if s.name == "emit")
    assert step.diff is not None
    assert step.diff.values["password"] == "[redacted]"
    assert step.diff.values["custom_private"] == "[redacted]"
    assert step.diff.values["visible"] == "public"


def test_value_truncation_limits_length() -> None:
    class BigState(TypedDict):
        blob: str

    def emit(state: BigState) -> BigState:
        return {"blob": "abcdefghij"}

    builder = StateGraph(BigState)
    builder.add_node("emit", emit)
    builder.add_edge(START, "emit")
    builder.add_edge("emit", END)
    graph = builder.compile()

    trace = collect_debug_trace(graph, {"blob": ""}, max_value_chars=5)

    step = next(s for s in trace.steps if s.name == "emit")
    assert step.diff is not None
    assert step.diff.values["blob"] == "abcde..."
    assert "abcdefghij" not in trace.to_text()


def test_tool_calls_and_results_extracted_from_messages() -> None:
    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent(state: ChatState) -> ChatState:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search",
                            "args": {"query": "weather"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def tools(state: ChatState) -> ChatState:
        return {
            "messages": [
                ToolMessage(content="sunny", name="search", tool_call_id="call-1")
            ]
        }

    builder = StateGraph(ChatState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    trace = collect_debug_trace(graph, {"messages": []})

    agent_step = next(s for s in trace.steps if s.name == "agent")
    calls = [tc for tc in agent_step.tool_calls if tc.args is not None]
    assert calls
    assert isinstance(calls[0], ToolCallSummary)
    assert calls[0].name == "search"
    assert calls[0].id == "call-1"
    assert calls[0].args == {"query": "weather"}

    tools_step = next(s for s in trace.steps if s.name == "tools")
    results = [tc for tc in tools_step.tool_calls if tc.result is not None]
    assert results
    assert results[0].name == "search"
    assert results[0].id == "call-1"
    assert results[0].result == "sunny"

    text = trace.to_text()
    assert "search" in text


def test_single_string_stream_mode_is_accepted() -> None:
    trace = collect_debug_trace(_linear_graph(), {"values": []}, stream_mode="tasks")
    assert [s.name for s in trace.steps if s.name in {"one", "two"}] == ["one", "two"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
