"""Chat with the agent in your terminal.

    python -m support_agent.cli                     # offline scripted planner
    ANTHROPIC_API_KEY=... python -m support_agent.cli --planner anthropic
"""
from __future__ import annotations

import argparse
import uuid

from langgraph.types import Command

from .backend import Backend
from .graph import build_graph, user_turn
from .planner import make_planner, text_of


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--org", default="acme-housing")
    p.add_argument("--planner", choices=["scripted", "anthropic"], default=None)
    a = p.parse_args()

    graph = build_graph(make_planner(a.planner), Backend(), a.org)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"Support agent for {a.org}. Try: 'show open tasks', 'assign T-101 to sanne@acme.example', "
          "'close T-103', 'undo', 'email tenant@example.com the repair is planned'. Ctrl+C to quit.")
    while True:
        try:
            text = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            return
        try:
            result = graph.invoke(user_turn(text), config)
        except ValueError as e:
            print(f"(rejected: {e})")
            continue
        while result.get("__interrupt__"):
            req = result["__interrupt__"][0].value
            ok = input(f"\n[approval] {req['question']} {req['input']} [y/N] ").strip().lower() == "y"
            result = graph.invoke(Command(resume={"approved": ok}), config)
        print("agent>", text_of(result["messages"][-1]))


if __name__ == "__main__":
    main()
