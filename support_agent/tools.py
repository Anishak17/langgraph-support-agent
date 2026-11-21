"""Tools the agent can call, each with a risk level that decides how it is executed:

  read      runs immediately
  write     runs immediately, and can be undone
  approve   pauses the graph and asks a human first (destructive or can't be undone)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .backend import Backend, NotFound

Risk = Literal["read", "write", "approve"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    risk: Risk
    run: Callable[[Backend, str, dict], Any]

    def spec(self) -> dict:
        """Tool definition in the format the Anthropic Messages API expects."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_tasks", "Search this organization's maintenance tasks.",
         _obj({"status": {"type": "string", "enum": ["open", "closed"]},
               "assignee": {"type": "string", "description": "staff email"},
               "text": {"type": "string", "description": "words in the title or address"}}),
         "read", lambda b, org, a: b.search_tasks(org, **a)),
    Tool("get_task", "Get one task by id, e.g. T-101.",
         _obj({"task_id": {"type": "string"}}, ["task_id"]),
         "read", lambda b, org, a: b.get_task(org, a["task_id"])),
    Tool("assign_task", "Assign a task to a staff member. Can be undone.",
         _obj({"task_id": {"type": "string"}, "assignee": {"type": "string"}}, ["task_id", "assignee"]),
         "write", lambda b, org, a: b.assign_task(org, a["task_id"], a["assignee"])),
    Tool("close_task", "Close a task. Needs human approval.",
         _obj({"task_id": {"type": "string"}}, ["task_id"]),
         "approve", lambda b, org, a: b.close_task(org, a["task_id"])),
    Tool("send_email", "Send an email. Cannot be undone, so it needs human approval.",
         _obj({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
              ["to", "subject", "body"]),
         "approve", lambda b, org, a: b.send_email(org, a["to"], a["subject"], a["body"])),
    Tool("undo_last_action", "Undo the most recent task change (assign or close).",
         _obj({}), "write", lambda b, org, a: b.undo_last(org)),
    Tool("escalate_to_human", "Hand the conversation to a human support agent with a short summary. "
         "Use when you cannot help or the user asks for a person.",
         _obj({"summary": {"type": "string"}}, ["summary"]),
         "write", lambda b, org, a: b.create_ticket(org, a["summary"])),
]}


def run_tool(backend: Backend, org_id: str, name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool. Returns (result text for the model, is_error)."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"unknown tool {name}", True
    try:
        return json.dumps(tool.run(backend, org_id, args or {}), default=str), False
    except (NotFound, ValueError, TypeError) as e:
        return f"error: {e}", True
