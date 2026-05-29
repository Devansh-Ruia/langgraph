from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Final, cast

__all__ = ("DebugTrace", "explain_debug_stream")

_DEFAULT_REDACT_KEYS: Final = frozenset(
    {"api_key", "token", "password", "secret", "authorization"}
)
_REDACTED: Final = "[redacted]"
_IGNORE: Final = object()
_Event = tuple[str, Mapping[str, Any], int | None, str | None, tuple[str, ...]]


@dataclass
class DebugTrace:
    """Experimental formatted view over already-produced debug stream events."""

    steps: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    unknown_events: int = 0

    def to_text(self) -> str:
        lines = ["LangGraph debug trace"]
        latest = self.checkpoints[-1] if self.checkpoints else {}
        if latest.get("thread_id") is not None:
            lines.append(f"Thread: {latest['thread_id']}")
        if latest.get("checkpoint_ns"):
            lines.append(f"Checkpoint ns: {latest['checkpoint_ns']}")
        if latest.get("checkpoint_id") is not None:
            lines.append(f"Checkpoint: {latest['checkpoint_id']}")
        lines.append("")

        if self.steps:
            for index, step in enumerate(self.steps, start=1):
                lines.extend(_step_lines(index, step))
                lines.append("")
        elif self.checkpoints:
            for index, checkpoint in enumerate(self.checkpoints, start=1):
                lines.extend(_checkpoint_lines(index, checkpoint))
                lines.append("")
        else:
            lines.append("No task or checkpoint events found")
            lines.append("")

        if self.unknown_events:
            lines.append(f"Unknown events: {self.unknown_events}")
            lines.append("")

        status = _trace_status(self.steps, self.checkpoints)
        if status:
            lines.append(status)
        return "\n".join(lines).rstrip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": deepcopy(self.steps),
            "checkpoints": deepcopy(self.checkpoints),
            "unknown_events": self.unknown_events,
        }


def explain_debug_stream(
    events: Iterable[Any],
    *,
    redact_keys: Iterable[str] = (),
    max_value_chars: int = 500,
) -> DebugTrace:
    """Format existing LangGraph debug stream chunks for quick local inspection.

    This experimental helper does not run a graph, change execution behavior, or
    replace LangSmith. It only formats chunks that were already produced for
    quick local debugging, for example:

    ```python
    events = list(graph.stream(input, config=config, stream_mode="debug"))
    trace = explain_debug_stream(events)
    print(trace.to_text())
    ```
    """

    redact = frozenset(key.casefold() for key in redact_keys) | _DEFAULT_REDACT_KEYS
    max_chars = max(0, max_value_chars)
    steps: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    last_values: dict[tuple[str, ...], Mapping[str, Any]] = {}
    unknown_events = 0

    for chunk in events:
        event = _unwrap(chunk)
        if event is _IGNORE:
            continue
        if event is None:
            unknown_events += 1
            continue

        kind, payload, step_no, timestamp, ns = cast(_Event, event)
        if kind == "checkpoint":
            checkpoint = _checkpoint(
                payload, step_no, timestamp, ns, last_values, redact, max_chars
            )
            checkpoints.append(checkpoint)
            _apply_checkpoint(checkpoint, steps)
            continue

        task_id = _string(payload.get("id"))
        step = _step(steps, by_id, task_id, payload, step_no, ns)
        if kind == "task":
            step["status"] = "started"
            step["started_at"] = timestamp
            if "input" in payload:
                step["input"] = _clean(payload["input"], redact, max_chars)
            _add_names(step, "triggers", _strings(payload.get("triggers")))
            continue

        error = payload.get("error")
        step["ended_at"] = timestamp
        step["status"] = "error" if error else "completed"
        if error:
            step["error"] = _clean(error, redact, max_chars)
        if "result" in payload:
            result = payload["result"]
            step["result"] = _clean(result, redact, max_chars)
            _add_names(step, "changed", _changed_result(result))
        if payload.get("interrupts"):
            step["interrupts"] = _clean(payload["interrupts"], redact, max_chars)

    return DebugTrace(
        steps=steps,
        checkpoints=checkpoints,
        unknown_events=unknown_events,
    )


def _step_lines(index: int, step: Mapping[str, Any]) -> list[str]:
    name = step.get("name") or "<unknown>"
    ns = _format_ns(step.get("namespace"))
    lines = [f"{index}. {name}" + (f" [{ns}]" if ns else "")]

    for label, key in (
        ("task", "status"),
        ("changed", "changed"),
        ("next", "next"),
        ("pending", "pending"),
    ):
        value = step.get(key)
        if not value:
            continue
        text = value if isinstance(value, str) else ", ".join(_strings(value))
        lines.append(f"   {label}: {text}")

    if step.get("error"):
        lines.append(f"   error: {_format(step['error'])}")
    if step.get("interrupts"):
        lines.append(f"   interrupts: {_format(step['interrupts'])}")
    if step.get("result"):
        lines.append(f"   writes: {_format(step['result'])}")
    return lines


def _checkpoint_lines(index: int, checkpoint: Mapping[str, Any]) -> list[str]:
    title = f"{index}. checkpoint"
    if checkpoint.get("step") is not None:
        title += f" step {checkpoint['step']}"
    lines = [title]
    for label, key in (
        ("source", "source"),
        ("changed", "changed"),
        ("next", "next"),
        ("pending", "pending"),
    ):
        value = checkpoint.get(key)
        if value:
            text = value if isinstance(value, str) else ", ".join(_strings(value))
            lines.append(f"   {label}: {text}")
    return lines


def _trace_status(
    steps: Sequence[Mapping[str, Any]], checkpoints: Sequence[Mapping[str, Any]]
) -> str | None:
    if any(step.get("status") == "error" for step in steps):
        return "Finished with errors"
    if any(step.get("interrupts") for step in steps) or any(
        checkpoint.get("interrupts") for checkpoint in checkpoints
    ):
        return "Interrupted"
    if any(step.get("status") == "started" for step in steps):
        return "Pending"
    if checkpoints and checkpoints[-1].get("next"):
        return "Pending"
    if steps or checkpoints:
        return "Finished"
    return None


def _unwrap(chunk: Any) -> _Event | object | None:
    ns: tuple[str, ...] = ()
    mode: str | None = None
    data = chunk

    if isinstance(chunk, tuple):
        parsed = _tuple_chunk(chunk)
        if parsed is None:
            return None
        ns, mode, data = parsed

    if mode == "debug":
        return _debug_event(data, ns)
    if mode == "tasks":
        return _task_event(data, ns)
    if mode == "checkpoints":
        return _checkpoint_event(data, ns)
    if mode is not None:
        return _IGNORE

    item = _mapping(data)
    if item is None:
        return None

    kind = item.get("type")
    if kind == "debug":
        return _debug_event(item.get("data"), _ns(item.get("ns")) or ns)
    if kind == "tasks":
        return _task_event(item.get("data"), _ns(item.get("ns")) or ns)
    if kind == "checkpoints":
        return _checkpoint_event(item.get("data"), _ns(item.get("ns")) or ns)
    if kind in {"values", "updates", "messages", "custom"}:
        return _IGNORE

    event = _debug_event(item, ns)
    if event is not None:
        return event
    if "id" in item and "name" in item:
        return _task_event(item, ns)
    if {"values", "next", "tasks"} <= item.keys():
        return _checkpoint_event(item, ns)
    return None


def _tuple_chunk(
    chunk: tuple[Any, ...],
) -> tuple[tuple[str, ...], str | None, Any] | None:
    if len(chunk) == 2:
        first, second = chunk
        if isinstance(first, str):
            return (), first, second
        ns = _ns(first)
        if ns is not None:
            return ns, None, second
    if len(chunk) == 3:
        first, second, third = chunk
        ns = _ns(first)
        if ns is not None and isinstance(second, str):
            return ns, second, third
    return None


def _debug_event(data: Any, ns: tuple[str, ...]) -> _Event | None:
    item = _mapping(data)
    if item is None or item.get("type") not in {"task", "task_result", "checkpoint"}:
        return None
    payload = _mapping(item.get("payload")) or {}
    return (
        cast(str, item["type"]),
        payload,
        item.get("step") if isinstance(item.get("step"), int) else None,
        _string(item.get("timestamp")),
        ns,
    )


def _task_event(data: Any, ns: tuple[str, ...]) -> _Event | None:
    payload = _mapping(data)
    if payload is None:
        return None
    kind = "task_result" if "result" in payload or "error" in payload else "task"
    return (kind, payload, None, None, ns)


def _checkpoint_event(data: Any, ns: tuple[str, ...]) -> _Event | None:
    payload = _mapping(data)
    if payload is None:
        return None
    metadata = _mapping(payload.get("metadata"))
    step = (
        metadata.get("step")
        if metadata and isinstance(metadata.get("step"), int)
        else None
    )
    return ("checkpoint", payload, step, None, ns)


def _step(
    steps: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    task_id: str | None,
    payload: Mapping[str, Any],
    step_no: int | None,
    ns: tuple[str, ...],
) -> dict[str, Any]:
    if task_id is not None and task_id in by_id:
        step = by_id[task_id]
        if step["name"] == "<unknown>":
            step["name"] = payload.get("name") or "<unknown>"
        return step

    step = {
        "id": task_id,
        "name": payload.get("name") or "<unknown>",
        "step": step_no,
        "namespace": ns,
        "status": "started",
        "changed": [],
        "next": [],
        "pending": [],
    }
    if task_id is not None:
        by_id[task_id] = step
    steps.append(step)
    return step


def _checkpoint(
    payload: Mapping[str, Any],
    step_no: int | None,
    timestamp: str | None,
    ns: tuple[str, ...],
    last_values: dict[tuple[str, ...], Mapping[str, Any]],
    redact: frozenset[str],
    max_chars: int,
) -> dict[str, Any]:
    config = _mapping(payload.get("config"))
    parent_config = _mapping(payload.get("parent_config"))
    configurable = _mapping(config.get("configurable")) if config else None
    parent_configurable = (
        _mapping(parent_config.get("configurable")) if parent_config else None
    )
    metadata = _mapping(payload.get("metadata"))
    tasks = _sequence(payload.get("tasks"))
    values = _mapping_value(payload.get("values"))

    changed = _changed_values(values, last_values.get(ns)) if values is not None else []
    if values is not None:
        last_values[ns] = values

    checkpoint = {
        "step": step_no,
        "timestamp": timestamp,
        "namespace": ns,
        "source": metadata.get("source") if metadata else None,
        "thread_id": configurable.get("thread_id") if configurable else None,
        "checkpoint_id": configurable.get("checkpoint_id") if configurable else None,
        "checkpoint_ns": configurable.get("checkpoint_ns") if configurable else None,
        "parent_checkpoint_id": parent_configurable.get("checkpoint_id")
        if parent_configurable
        else None,
        "changed": changed,
        "next": _strings(payload.get("next")),
        "pending": _pending(tasks),
        "interrupts": _clean(_interrupts(tasks), redact, max_chars),
    }
    if "values" in payload:
        checkpoint["values"] = _clean(payload.get("values"), redact, max_chars)
    return checkpoint


def _apply_checkpoint(
    checkpoint: Mapping[str, Any], steps: list[dict[str, Any]]
) -> None:
    for step in reversed(steps):
        if step.get("step") != checkpoint.get("step"):
            continue
        if step.get("namespace") != checkpoint.get("namespace"):
            continue
        _add_names(step, "changed", _strings(checkpoint.get("changed")))
        _add_names(step, "next", _strings(checkpoint.get("next")))
        _add_names(step, "pending", _strings(checkpoint.get("pending")))
        if checkpoint.get("interrupts") and not step.get("interrupts"):
            step["interrupts"] = checkpoint["interrupts"]
        return


def _changed_result(result: Any) -> list[str]:
    mapping = _mapping_value(result)
    return _public_keys(mapping) if mapping is not None else []


def _changed_values(
    values: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> list[str]:
    if previous is None:
        return _public_keys(values)

    changed: list[str] = []
    missing = object()
    for key, value in values.items():
        try:
            is_changed = previous.get(key, missing) != value
        except Exception:
            is_changed = True
        if is_changed:
            changed.extend(_public_keys({key: value}))
    return changed


def _public_keys(mapping: Mapping[Any, Any]) -> list[str]:
    names: list[str] = []
    for key in mapping:
        name = str(key)
        if not name.startswith("__") and name not in names:
            names.append(name)
    return names


def _pending(tasks: Sequence[Any]) -> list[str]:
    names = []
    for task in tasks:
        item = _mapping(task)
        if item is None or "result" in item or item.get("error"):
            continue
        name = _string(item.get("name"))
        if name:
            names.append(name)
    return names


def _interrupts(tasks: Sequence[Any]) -> list[Any]:
    values: list[Any] = []
    for task in tasks:
        item = _mapping(task)
        interrupts = item.get("interrupts") if item else None
        if isinstance(interrupts, Sequence) and not isinstance(interrupts, str):
            values.extend(interrupts)
        elif interrupts:
            values.append(interrupts)
    return values


def _add_names(target: dict[str, Any], key: str, names: Iterable[str]) -> None:
    current = _strings(target.get(key))
    for name in names:
        if name and name not in current:
            current.append(name)
    target[key] = current


def _clean(value: Any, redact: frozenset[str], max_chars: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, Mapping):
        return {
            key: _REDACTED
            if any(redact_key in str(key).casefold() for redact_key in redact)
            else _clean(item, redact, max_chars)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clean(item, redact, max_chars) for item in value)
    if isinstance(value, list):
        return [_clean(item, redact, max_chars) for item in value]
    if isinstance(value, (bool, int, float, type(None))):
        return value
    return _truncate(repr(value), max_chars)


def _format(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else pformat(value, compact=True, sort_dicts=False)
    )


def _truncate(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else f"{value[:max_chars]}..."


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else None


def _mapping_value(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:
            return None
        if isinstance(dumped, Mapping):
            return cast(Mapping[str, Any], dumped)
    return None


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _ns(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    items = tuple(value)
    return (
        cast(tuple[str, ...], items)
        if all(isinstance(item, str) for item in items)
        else None
    )


def _format_ns(value: Any) -> str:
    ns = _ns(value)
    return " / ".join(ns) if ns else ""


def _string(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)
