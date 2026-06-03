# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository is a monorepo. Each library lives in a subdirectory under `libs/`. Python libraries use `uv` for dependency management and `make` targets for common tasks. JS/TS libraries (`sdk-js`, parts of `cli`) use `yarn`.

## Development commands

Run all commands from inside the relevant `libs/<name>` directory.

- `make install` – install deps (`uv sync --frozen --all-extras --all-packages --group dev`)
- `make format` – run formatters (ruff format + ruff check --fix)
- `make lint` – run linters (ruff check, ruff format --diff, import sort, mypy)
- `make type` – mypy type checking only
- `make test` – run the test suite
- `make spell_check` / `make spell_fix` – codespell

### Running a single test

`TEST` is passed straight to pytest, so it accepts files, node IDs, and pytest options:

```
TEST=tests/test_pregel.py make test
TEST="tests/test_pregel.py::test_invoke -k async -x" make test
```

### `langgraph` library test specifics

`make test` in `libs/langgraph` auto-detects Docker. If Docker is present it spins up Postgres + Redis (`make start-services`) and a dev server before running pytest; otherwise it runs with `NO_DOCKER=true` (skips tests needing those services). Other useful targets there:

- `make test_parallel` – pytest under xdist with `--lf` (last-failed first)
- `make start-services` / `make stop-services` – Postgres + Redis via docker compose
- `make start-dev-server` / `make stop-dev-server` – local `langgraph dev` server
- `make benchmark` / `make benchmark-fast` – run benchmarks in `bench/`
- `make coverage` – tests with coverage report

Python target: `>=3.10`. Snapshot tests use `syrupy` (update with `--snapshot-update` via `TEST`).

## Before opening a PR

Run `make format`, `make lint`, and `make test` in the directory of every library you changed. A change to a library may break its dependents (see dependency map) — test those too.

## Libraries

- **checkpoint** (`langgraph-checkpoint`) – base interfaces for checkpointers (state persistence).
- **checkpoint-postgres** – Postgres checkpoint saver.
- **checkpoint-sqlite** – SQLite checkpoint saver.
- **checkpoint-conformance** – shared conformance test suite that checkpoint implementations run against.
- **cli** – official command-line interface (`langgraph` command); builds/deploys LangGraph apps.
- **langgraph** – core framework for stateful, multi-actor agents.
- **prebuilt** (`langgraph-prebuilt`) – high-level agent/tool APIs (e.g. `create_react_agent`, `ToolNode`).
- **sdk-js** – JS/TS SDK for the LangGraph REST API (standalone).
- **sdk-py** – Python SDK for the LangGraph Server API.

### Dependency map

Downstream libraries for each production dependency, per that library's `pyproject.toml` / `package.json`:

```text
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (standalone)
```

## Core architecture (`libs/langgraph`)

LangGraph is a low-level orchestration framework. Two layers matter:

**Build layer — `langgraph/graph/state.py` (`StateGraph`).** User-facing API. You define a typed state schema (`TypedDict` / pydantic / dataclass), add nodes (functions) and edges (static or conditional), then call `.compile()`. Reducers on state fields (e.g. `add_messages` in `graph/message.py`) control how node outputs merge into state. The functional API lives in `langgraph/func/` (`@entrypoint`, `@task`).

**Runtime layer — `langgraph/pregel/` (`Pregel`).** `.compile()` lowers a `StateGraph` into a `Pregel` instance. The runtime is a Bulk Synchronous Parallel (BSP) message-passing engine inspired by Google Pregel: execution proceeds in discrete **super-steps**. Each step, all triggered nodes run (in parallel), their writes are applied to channels, and the next step's nodes are selected by which channels updated. Key modules: `_loop.py` (the super-step loop), `_algo.py` (which tasks to run, apply writes), `_runner.py` / `_executor.py` (node execution), `_read.py` / `_write.py` (channel I/O).

**Channels — `langgraph/channels/`.** State is stored in channels, not a flat dict. `LastValue` (default — keeps last write), `Topic` (pub/sub list), `BinaryOperatorAggregate` (reducer-backed), `EphemeralValue`, `NamedBarrierValue`, etc. The reducer you attach to a state field picks the channel type.

**Checkpointing & durability.** After each super-step the runtime can persist channel values + pending writes via a checkpointer (`langgraph-checkpoint` interface). This is what enables durable execution, time-travel, human-in-the-loop interrupts (`interrupt()` / `Command(resume=...)` in `types.py`), and memory across threads. Interrupts work by raising out of the loop and resuming from the saved checkpoint.

**Streaming — `langgraph/stream/`.** Multiple stream modes (`values`, `updates`, `messages`, `debug`, `custom`) multiplexed out of the same run.

Public entry points: `from langgraph.graph import StateGraph, START, END, add_messages`.

## Conventions

- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`` `code` ``) for inline code references in docstrings and comments.
