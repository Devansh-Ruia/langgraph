"""Experimental, inspection-only debug trace for local LangGraph debugging.

This module is experimental and not part of the stable public API. It runs a
graph with `graph.stream(...)` and normalizes the resulting `updates`, `tasks`,
and `checkpoints` stream events into a compact, readable execution timeline. It
is a pure consumer of the stream: it does not change how the graph runs, alter
execution semantics, or replace tracing tools like LangSmith.

Example:

```python
from langgraph.pregel._inspection import collect_debug_trace

trace = collect_debug_trace(graph, {"messages": []}, config)
print(trace.to_text())
```
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from langchain_core.messages import BaseMessage

__all__ = (
    "DebugTrace",
    "DebugStep",
    "StateDiff",
    "ToolCallSummary",
    "collect_debug_trace",
)

_DEFAULT_REDACT_KEYS: Final = frozenset(
    {"api_key", "token", "password", "secret", "authorization"}
)
_REDACTED: Final = "[redacted]"
_DEFAULT_MODES: Final = ("updates", "tasks", "checkpoints")


@dataclass
class ToolCallSummary:
    """A tool call or tool result extracted from a message.

    A summary produced from an `AIMessage.tool_calls` entry has `args` set and
    `result` left as `None`. A summary produced from a `ToolMessage` has
    `result` set and `args` left as `None`.
    """

    name: str | None
    id: str | None = None
    args: Any = None
    result: Any = None


@dataclass
class StateDiff:
    """The state changes recorded for a single step."""

    changed_keys: list[str]
    values: dict[str, Any]


@dataclass
class DebugStep:
    """A single node execution within a `DebugTrace`."""

    name: str
    namespace: tuple[str, ...] = ()
    status: str = "started"
    step: int | None = None
    diff: StateDiff | None = None
    tool_calls: list[ToolCallSummary] = field(default_factory=list)
    error: str | None = None
    interrupts: list[Any] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    checkpoint_id: str | None = None


@dataclass
class DebugTrace:
    """Experimental compact timeline of a single graph run.

    Build one with `collect_debug_trace`. Render it with `to_text` for human
    reading or `to_json` for a stable, machine-readable view.
    """

    steps: list[DebugStep] = field(default_factory=list)
    thread_id: str | None = None
    checkpoint_ids: list[str] = field(default_factory=list)
    status: str = "empty"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the trace."""
        return {
            "status": self.status,
            "thread_id": self.thread_id,
            "checkpoint_ids": list(self.checkpoint_ids),
            "error": self.error,
            "steps": [_step_to_dict(step) for step in self.steps],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the trace serialized as JSON text."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        """Return a compact, human-readable rendering of the trace."""
        lines = ["LangGraph debug trace"]
        if self.thread_id is not None:
            lines.append(f"Thread: {self.thread_id}")
        if self.checkpoint_ids:
            lines.append(f"Checkpoint: {self.checkpoint_ids[-1]}")
        lines.append(f"Status: {self.status}")
        if self.error:
            lines.append(f"Error: {self.error}")
        lines.append("")

        if not self.steps:
            lines.append("No task events found")
            return "\n".join(lines).rstrip()

        for index, step in enumerate(self.steps, start=1):
            lines.extend(_step_lines(index, step))
            lines.append("")
        return "\n".join(lines).rstrip()


def collect_debug_trace(
    graph: Any,
    input: Any,
    config: Any = None,
    *,
    stream_mode: str | Sequence[str] | None = None,
    redact_keys: Iterable[str] = (),
    max_value_chars: int = 500,
) -> DebugTrace:
    """Run `graph` and normalize its debug stream into a `DebugTrace`.

    This experimental helper runs the graph with `graph.stream(...)` using the
    `updates`, `tasks`, and `checkpoints` stream modes (overridable via
    `stream_mode`) and folds the events into a stable, inspectable object. It
    does not change execution semantics.

    Args:
        graph: A compiled graph (a `Pregel` instance) exposing `stream`.
        input: The input passed to `graph.stream`.
        config: Optional config passed to `graph.stream` (e.g. with a
            `thread_id` for a checkpointed run).
        stream_mode: Stream mode(s) to request. Defaults to `updates`, `tasks`,
            and `checkpoints`. A bare string is accepted and wrapped in a list.
        redact_keys: Extra mapping keys whose values are replaced with
            `[redacted]` (a small default set is always applied).
        max_value_chars: Maximum length for any captured string value before it
            is truncated.

    Returns:
        A `DebugTrace` describing the run.
    """
    if stream_mode is None:
        modes: list[str] = list(_DEFAULT_MODES)
    elif isinstance(stream_mode, str):
        modes = [stream_mode]
    else:
        modes = list(stream_mode)

    builder = _TraceBuilder(redact_keys, max_value_chars)
    try:
        for chunk in graph.stream(input, config, stream_mode=modes):
            builder.consume(chunk)
    except Exception as exc:  # noqa: BLE001 - inspection records, never re-raises
        builder.record_error(exc)
    return builder.finish()


class _TraceBuilder:
    """Folds raw stream chunks into a `DebugTrace`."""

    def __init__(self, redact_keys: Iterable[str], max_value_chars: int) -> None:
        self.redact = (
            frozenset(k.casefold() for k in redact_keys) | _DEFAULT_REDACT_KEYS
        )
        self.max_chars = max(0, max_value_chars)
        self.steps: list[DebugStep] = []
        self.by_id: dict[str, DebugStep] = {}
        self.open_by_name: dict[str, DebugStep] = {}
        self.thread_id: str | None = None
        self.checkpoint_ids: list[str] = []
        self.error: str | None = None
        self.error_type: str | None = None
        self.diff_from_updates: set[int] = set()
        self.seen_tool_calls: dict[int, set[tuple[Any, Any, Any]]] = {}

    # -- event dispatch ------------------------------------------------------

    def consume(self, chunk: Any) -> None:
        parsed = _parse_chunk(chunk)
        if parsed is None:
            return
        ns, mode, data = parsed
        if mode == "tasks":
            self._handle_tasks(ns, data)
        elif mode == "updates":
            self._handle_updates(data)
        elif mode == "checkpoints":
            self._handle_checkpoints(data)

    def _handle_tasks(self, ns: tuple[str, ...], data: Any) -> None:
        payload = _mapping(data)
        if payload is None:
            return
        is_result = "result" in payload or "error" in payload
        task_id = _string(payload.get("id"))
        name = payload.get("name") or "<unknown>"

        if not is_result:
            step = DebugStep(
                name=name,
                namespace=ns,
                status="started",
                triggers=_strings(payload.get("triggers")),
            )
            self.steps.append(step)
            if task_id is not None:
                self.by_id[task_id] = step
            self.open_by_name[name] = step
            return

        step = self._lookup_step(task_id, name, ns)
        error = payload.get("error")
        step.status = "error" if error else "completed"
        if error:
            step.error = _string(_clean(error, self.redact, self.max_chars))
        result = _mapping(payload.get("result"))
        if result is not None:
            self._record_changes(step, result, from_updates=False)
            self._add_tool_calls(step, result)
        interrupts = payload.get("interrupts")
        if interrupts:
            step.interrupts = _as_list(_clean(interrupts, self.redact, self.max_chars))
        self.open_by_name.pop(name, None)

    def _handle_updates(self, data: Any) -> None:
        payload = _mapping(data)
        if payload is None:
            return
        for name, update in payload.items():
            step = self.open_by_name.get(name) or self._last_step_named(name)
            if step is None:
                step = DebugStep(name=name, status="started")
                self.steps.append(step)
                self.open_by_name[name] = step
            for part in _as_list(update):
                mapping = _mapping(part)
                if mapping is not None:
                    self._record_changes(step, mapping, from_updates=True)
                self._add_tool_calls(step, part)

    def _handle_checkpoints(self, data: Any) -> None:
        payload = _mapping(data)
        if payload is None:
            return
        config = _mapping(payload.get("config")) or {}
        configurable = _mapping(config.get("configurable")) or {}
        if configurable.get("thread_id") is not None:
            self.thread_id = _string(configurable.get("thread_id"))
        checkpoint_id = _string(configurable.get("checkpoint_id"))
        if checkpoint_id and checkpoint_id not in self.checkpoint_ids:
            self.checkpoint_ids.append(checkpoint_id)

        metadata = _mapping(payload.get("metadata")) or {}
        step_no = (
            metadata.get("step") if isinstance(metadata.get("step"), int) else None
        )
        for step in self.steps:
            if step.step is None and step.status in ("completed", "error"):
                step.step = step_no
                if checkpoint_id and step.checkpoint_id is None:
                    step.checkpoint_id = checkpoint_id

        next_nodes = _strings(payload.get("next"))
        if next_nodes and self.steps and not self.steps[-1].next:
            self.steps[-1].next = next_nodes

        self._apply_checkpoint_interrupts(payload.get("tasks"))

    def record_error(self, exc: BaseException) -> None:
        self.error = str(exc) or repr(exc)
        self.error_type = type(exc).__name__
        for step in self.steps:
            if step.status == "started":
                step.status = "error"
                if not step.error:
                    step.error = self.error

    def finish(self) -> DebugTrace:
        return DebugTrace(
            steps=self.steps,
            thread_id=self.thread_id,
            checkpoint_ids=self.checkpoint_ids,
            status=self._status(),
            error=self.error,
        )

    # -- helpers -------------------------------------------------------------

    def _lookup_step(
        self, task_id: str | None, name: str, ns: tuple[str, ...]
    ) -> DebugStep:
        if task_id is not None and task_id in self.by_id:
            return self.by_id[task_id]
        existing = self.open_by_name.get(name) or self._last_step_named(name)
        if existing is not None:
            return existing
        step = DebugStep(name=name, namespace=ns, status="started")
        self.steps.append(step)
        if task_id is not None:
            self.by_id[task_id] = step
        return step

    def _last_step_named(self, name: str) -> DebugStep | None:
        for step in reversed(self.steps):
            if step.name == name:
                return step
        return None

    def _record_changes(
        self, step: DebugStep, mapping: Mapping[str, Any], *, from_updates: bool
    ) -> None:
        keys = _public_keys(mapping)
        if step.diff is None:
            step.diff = StateDiff(changed_keys=[], values={})
        for key in keys:
            if key not in step.diff.changed_keys:
                step.diff.changed_keys.append(key)
        prefer = from_updates or id(step) not in self.diff_from_updates
        if prefer:
            cleaned = _clean(dict(mapping), self.redact, self.max_chars)
            if isinstance(cleaned, Mapping):
                step.diff.values.update(cleaned)
            if from_updates:
                self.diff_from_updates.add(id(step))

    def _add_tool_calls(self, step: DebugStep, value: Any) -> None:
        seen = self.seen_tool_calls.setdefault(id(step), set())
        for summary in _extract_tool_calls(value, self.redact, self.max_chars):
            key = (
                "call" if summary.args is not None else "result",
                summary.name,
                summary.id,
            )
            if key in seen:
                continue
            seen.add(key)
            step.tool_calls.append(summary)

    def _apply_checkpoint_interrupts(self, tasks: Any) -> None:
        for task in _as_list(tasks):
            item = _mapping(task)
            if item is None:
                continue
            interrupts = item.get("interrupts")
            if not interrupts:
                continue
            step = self.by_id.get(_string(item.get("id")) or "")
            if step is None:
                step = self._last_step_named(item.get("name") or "")
            if step is not None and not step.interrupts:
                step.interrupts = _as_list(
                    _clean(interrupts, self.redact, self.max_chars)
                )

    def _status(self) -> str:
        if self.error is not None:
            if self.error_type and "interrupt" in self.error_type.casefold():
                return "interrupted"
            return "error"
        if any(step.status == "error" for step in self.steps):
            return "error"
        if any(step.interrupts for step in self.steps):
            return "interrupted"
        if any(step.status == "started" for step in self.steps):
            return "pending"
        if self.steps and self.steps[-1].next:
            return "pending"
        if self.steps or self.checkpoint_ids:
            return "finished"
        return "empty"


# -- module-level helpers ----------------------------------------------------


def _parse_chunk(chunk: Any) -> tuple[tuple[str, ...], str, Any] | None:
    if not isinstance(chunk, tuple):
        return None
    if len(chunk) == 2:
        first, second = chunk
        if isinstance(first, str):
            return (), first, second
        return None
    if len(chunk) == 3:
        first, second, third = chunk
        ns = _ns(first)
        if ns is not None and isinstance(second, str):
            return ns, second, third
    return None


def _extract_tool_calls(
    value: Any, redact: frozenset[str], max_chars: int
) -> list[ToolCallSummary]:
    summaries: list[ToolCallSummary] = []
    for msg in _iter_messages(value):
        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str):
            for call in tool_calls:
                item = _mapping(call)
                if item is None:
                    continue
                summaries.append(
                    ToolCallSummary(
                        name=_string(item.get("name")),
                        id=_string(item.get("id")),
                        args=_clean(item.get("args", {}), redact, max_chars),
                    )
                )
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id is not None or getattr(msg, "type", None) == "tool":
            summaries.append(
                ToolCallSummary(
                    name=_string(getattr(msg, "name", None)),
                    id=_string(tool_call_id),
                    result=_clean(getattr(msg, "content", None), redact, max_chars),
                )
            )
    return summaries


def _iter_messages(value: Any) -> Iterable[Any]:
    if isinstance(value, BaseMessage):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, BaseMessage):
                yield item
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_messages(item)


def _step_to_dict(step: DebugStep) -> dict[str, Any]:
    return {
        "step": step.step,
        "name": step.name,
        "namespace": list(step.namespace),
        "status": step.status,
        "diff": (
            None
            if step.diff is None
            else {
                "changed_keys": list(step.diff.changed_keys),
                "values": step.diff.values,
            }
        ),
        "tool_calls": [
            {
                "name": tc.name,
                "id": tc.id,
                "args": tc.args,
                "result": tc.result,
            }
            for tc in step.tool_calls
        ],
        "error": step.error,
        "interrupts": step.interrupts,
        "next": list(step.next),
        "triggers": list(step.triggers),
        "checkpoint_id": step.checkpoint_id,
    }


def _step_lines(index: int, step: DebugStep) -> list[str]:
    ns = " / ".join(step.namespace)
    header = f"{index}. {step.name}"
    if ns:
        header += f" [{ns}]"
    if step.step is not None:
        header += f" (step {step.step})"
    lines = [header, f"   status: {step.status}"]
    if step.diff and step.diff.changed_keys:
        lines.append(f"   changed: {', '.join(step.diff.changed_keys)}")
    for call in step.tool_calls:
        if call.args is not None:
            lines.append(f"   tool call: {call.name}({_format(call.args)})")
        else:
            lines.append(f"   tool result: {call.name} -> {_format(call.result)}")
    if step.error:
        lines.append(f"   error: {step.error}")
    if step.interrupts:
        lines.append(f"   interrupts: {_format(step.interrupts)}")
    if step.next:
        lines.append(f"   next: {', '.join(step.next)}")
    if step.diff and step.diff.values:
        lines.append(f"   writes: {_format(step.diff.values)}")
    return lines


def _clean(value: Any, redact: frozenset[str], max_chars: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, Mapping):
        return {
            key: _REDACTED
            if any(rk in str(key).casefold() for rk in redact)
            else _clean(item, redact, max_chars)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_clean(item, redact, max_chars) for item in value]
    if isinstance(value, (bool, int, float, type(None))):
        return value
    return _truncate(repr(value), max_chars)


def _format(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _truncate(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else f"{value[:max_chars]}..."


def _public_keys(mapping: Mapping[Any, Any]) -> list[str]:
    names: list[str] = []
    for key in mapping:
        name = str(key)
        if not name.startswith("__") and name not in names:
            names.append(name)
    return names


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _ns(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    items = tuple(value)
    return items if all(isinstance(item, str) for item in items) else None


def _string(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)
