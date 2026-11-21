# langgraph-support-agent

A support assistant for a (fictional) property-management app, built as a
[LangGraph](https://langchain-ai.github.io/langgraph/) state machine. It doesn't just answer
questions: it **changes things**, such as assigning tasks, closing tasks and sending emails.
So the interesting part is the safety around those actions.

**Status:** proof of concept. The tests use an offline scripted planner. The Claude planner
is implemented, but this repo doesn't include an evaluation of its tool choices yet.

```
START → agent ──(tool calls?)──→ tools ──→ agent → … → END
              └──(plain answer)──→ END
```

## What it shows

| Problem | How it's handled |
|---|---|
| The agent must only touch its own customer's data | `org_id` is bound when the graph is built. No tool has an org parameter, and the backend returns "not found" for other orgs' ids. |
| Some actions are risky | Every tool has a risk level: `read` (runs), `write` (runs, can be undone), `approve` (pauses for a human). |
| Waiting for a human | The tools node calls LangGraph's `interrupt()`. State is saved by a checkpointer, and the graph resumes with `Command(resume={"approved": True})`. |
| An action that runs twice | On resume LangGraph re-runs the interrupted node from the top. So the node collects **all** approvals first and only then executes. A test checks that an approved close runs exactly once. |
| Mistakes | `assign_task` and `close_task` save a snapshot; `undo_last_action` restores it. |
| The agent can't help | `escalate_to_human` creates a ticket with a summary for a person. |
| Runaway loops | At most 6 tool rounds per user message, and a cap on message length. |
| Prompt injection through data | The system prompt treats tool output as data, not instructions. |

## Run it

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q

# Offline: a keyword-based planner stands in for the LLM
python -m support_agent.cli

# With Claude
export ANTHROPIC_API_KEY=...        # from console.anthropic.com
python -m support_agent.cli --planner anthropic
```

Example session:
```
you> assign T-101 to sanne@acme.example
agent> Done. ...
you> close T-103
[approval] Allow the assistant to run close_task? {'task_id': 'T-103'} [y/N] y
agent> Done. ...
you> undo
agent> Done. Result: {"undone": "close T-103", ...}
```

## Layout

```
support_agent/
  graph.py     LangGraph state, nodes, routing, interrupt-based approvals
  tools.py     tool definitions with risk levels
  backend.py   fake multi-tenant backend with undo snapshots
  planner.py   Claude planner (Anthropic API) and an offline scripted planner
  cli.py       terminal chat with approval prompts
```

## Next steps

- Persist state with `SqliteSaver`/`PostgresSaver` so a conversation survives restarts.
- Stream tokens to a web UI.
- Evaluate tool choice on a set of scripted conversations.
