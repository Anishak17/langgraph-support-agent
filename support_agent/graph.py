"""The agent as a LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                  \\--(plain answer)--> END

The tools node pauses with `interrupt()` before any tool marked "approve" and resumes
when a human answers with `Command(resume={"approved": True/False})`.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .backend import Backend
from .planner import Planner
from .tools import TOOLS, run_tool

MAX_TOOL_ROUNDS = 6      # stop runaway loops within one user turn
MAX_USER_CHARS = 2000


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]
    tool_rounds: int


def build_graph(planner: Planner, backend: Backend, org_id: str, checkpointer=None):
    """org_id is bound here, by the application, never by the model."""
    tool_specs = [t.spec() for t in TOOLS.values()]

    def agent(state: State) -> dict:
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            text = "I'm stopping here because this took too many steps. Try a simpler request."
            return {"messages": [{"role": "assistant", "content": [{"type": "text", "text": text}]}]}
        blocks = planner.next(state["messages"], tool_specs)
        return {"messages": [{"role": "assistant", "content": blocks}]}

    def tools(state: State) -> dict:
        calls = [b for b in state["messages"][-1]["content"] if b.get("type") == "tool_use"]

        # Ask for ALL approvals before executing anything. On resume LangGraph re-runs this
        # node from the top; if a tool had already run before an interrupt, it would run twice.
        approved: dict[str, bool] = {}
        for call in calls:
            if TOOLS.get(call["name"]) and TOOLS[call["name"]].risk == "approve":
                answer = interrupt({"tool": call["name"], "input": call["input"],
                                    "question": f"Allow the assistant to run {call['name']}?"})
                approved[call["id"]] = bool(answer.get("approved")) if isinstance(answer, dict) else bool(answer)

        results = []
        for call in calls:
            if approved.get(call["id"], True):
                text, is_error = run_tool(backend, org_id, call["name"], call["input"])
            else:
                text, is_error = "The user did not approve this action, so it was not done.", True
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": text, "is_error": is_error})
        return {"messages": [{"role": "user", "content": results}],
                "tool_rounds": state.get("tool_rounds", 0) + 1}

    def route(state: State) -> str:
        last = state["messages"][-1]["content"]
        return "tools" if any(b.get("type") == "tool_use" for b in last) else END

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def user_turn(text: str) -> dict:
    text = text.strip()
    if not text or len(text) > MAX_USER_CHARS:
        raise ValueError(f"message must be 1-{MAX_USER_CHARS} characters")
    return {"messages": [{"role": "user", "content": text}], "tool_rounds": 0}
