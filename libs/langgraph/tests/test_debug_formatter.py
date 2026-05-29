from __future__ import annotations

from langgraph.debug import explain_debug_stream


def test_explain_debug_stream_formats_basic_task_trace() -> None:
    trace = explain_debug_stream(
        [
            {
                "type": "task",
                "timestamp": "2026-01-01T00:00:00Z",
                "step": 1,
                "payload": {
                    "id": "task-1",
                    "name": "agent",
                    "input": {"messages": []},
                    "triggers": ("branch:to:agent",),
                },
            },
            {
                "type": "task_result",
                "timestamp": "2026-01-01T00:00:01Z",
                "step": 1,
                "payload": {
                    "id": "task-1",
                    "name": "agent",
                    "error": None,
                    "result": {"messages": ["next message"]},
                    "interrupts": [],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": "2026-01-01T00:00:02Z",
                "step": 1,
                "payload": {
                    "config": {"configurable": {"thread_id": "thread-1"}},
                    "metadata": {"source": "loop", "step": 1, "parents": {}},
                    "values": {"messages": ["next message"]},
                    "next": ["tools"],
                    "parent_config": None,
                    "tasks": [],
                },
            },
        ]
    )

    text = trace.to_text()
    assert "LangGraph debug trace" in text
    assert "1. agent" in text
    assert "task: completed" in text
    assert "changed: messages" in text
    assert "next: tools" in text
    assert "Thread: thread-1" in text

    data = trace.to_dict()
    assert data["steps"][0]["name"] == "agent"
    assert data["steps"][0]["changed"] == ["messages"]
    assert data["steps"][0]["next"] == ["tools"]
    assert data["checkpoints"][0]["thread_id"] == "thread-1"


def test_explain_debug_stream_accepts_wrapped_and_unknown_chunks() -> None:
    trace = explain_debug_stream(
        [
            (
                ("child:task-1",),
                {
                    "type": "task",
                    "step": 1,
                    "payload": {"id": "child-task", "name": "child"},
                },
            ),
            (
                "debug",
                {
                    "type": "task_result",
                    "step": 1,
                    "payload": {
                        "id": "child-task",
                        "name": "child",
                        "error": None,
                        "result": {"messages": ["done"]},
                        "interrupts": [],
                    },
                },
            ),
            {
                "type": "debug",
                "ns": (),
                "data": {
                    "type": "task",
                    "step": 2,
                    "payload": {"id": "root-task", "name": "root"},
                },
            },
            {"unexpected": "value"},
        ]
    )

    data = trace.to_dict()
    assert [step["name"] for step in data["steps"]] == ["child", "root"]
    assert data["steps"][0]["namespace"] == ("child:task-1",)
    assert data["unknown_events"] == 1
    assert "Unknown events: 1" in trace.to_text()


def test_explain_debug_stream_redacts_values() -> None:
    trace = explain_debug_stream(
        [
            {
                "type": "task_result",
                "step": 1,
                "payload": {
                    "id": "task-1",
                    "name": "agent",
                    "error": None,
                    "result": {
                        "password": "private_value",
                        "nested": {"token": "redacted_value"},
                        "visible": "public_value",
                    },
                    "interrupts": [],
                },
            }
        ]
    )

    step = trace.to_dict()["steps"][0]
    assert step["result"]["password"] == "[redacted]"
    assert step["result"]["nested"]["token"] == "[redacted]"
    assert step["result"]["visible"] == "public_value"
    assert "private_value" not in trace.to_text()
    assert "redacted_value" not in trace.to_text()


def test_explain_debug_stream_applies_extra_redact_keys() -> None:
    trace = explain_debug_stream(
        [
            {
                "type": "task_result",
                "step": 1,
                "payload": {
                    "id": "task-1",
                    "name": "agent",
                    "error": None,
                    "result": {"custom_private": "private_value"},
                    "interrupts": [],
                },
            }
        ],
        redact_keys=("custom_private",),
    )

    assert trace.to_dict()["steps"][0]["result"]["custom_private"] == "[redacted]"


def test_explain_debug_stream_truncates_values() -> None:
    trace = explain_debug_stream(
        [
            {
                "type": "task_result",
                "step": 1,
                "payload": {
                    "id": "task-1",
                    "name": "agent",
                    "error": None,
                    "result": {"messages": ["abcdef"]},
                    "interrupts": [],
                },
            }
        ],
        max_value_chars=5,
    )

    text = trace.to_text()
    assert "abcde..." in text
    assert "abcdef" not in text


def test_explain_debug_stream_formats_error_and_interrupts() -> None:
    trace = explain_debug_stream(
        [
            {
                "type": "task_result",
                "step": 2,
                "payload": {
                    "id": "task-2",
                    "name": "tools",
                    "error": "ValueError('bad input')",
                    "result": {},
                    "interrupts": [{"value": "review required"}],
                },
            }
        ]
    )

    text = trace.to_text()
    assert "1. tools" in text
    assert "task: error" in text
    assert "error: ValueError('bad input')" in text
    assert "interrupts:" in text
    assert trace.to_dict()["steps"][0]["status"] == "error"
