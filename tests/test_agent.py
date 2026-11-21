import json

import pytest

from support_agent.backend import Backend
from support_agent.planner import ScriptedPlanner
from support_agent.tools import TOOLS, run_tool

ORG = "acme-housing"


# ---- backend + tools (no LangGraph needed) -------------------------------------------
def test_other_orgs_tasks_are_invisible():
    b = Backend()
    text, err = run_tool(b, ORG, "get_task", {"task_id": "T-201"})  # belongs to globex
    assert err and "not found" in text
    assert all(t["task_id"].startswith("T-1") for t in b.search_tasks(ORG))


def test_assign_rejects_people_outside_org():
    text, err = run_tool(Backend(), ORG, "assign_task", {"task_id": "T-101", "assignee": "lena@globex.example"})
    assert err and "not a staff member" in text


def test_assign_then_undo_restores_previous_state():
    b = Backend()
    run_tool(b, ORG, "assign_task", {"task_id": "T-102", "assignee": "sanne@acme.example"})
    assert b.get_task(ORG, "T-102")["assignee"] == "sanne@acme.example"
    text, err = run_tool(b, ORG, "undo_last_action", {})
    assert not err and b.get_task(ORG, "T-102")["assignee"] == "daan@acme.example"
    text, err = run_tool(b, ORG, "undo_last_action", {})
    assert err and "nothing to undo" in text


def test_risk_levels():
    assert TOOLS["send_email"].risk == "approve"
    assert TOOLS["close_task"].risk == "approve"
    assert TOOLS["search_tasks"].risk == "read"
    assert all("org" not in json.dumps(t.input_schema) for t in TOOLS.values())  # model can't pick the org


def test_scripted_planner_picks_tools():
    p = ScriptedPlanner()
    call = p.next([{"role": "user", "content": "please assign t-101 to sanne@acme.example"}], [])[0]
    assert call["name"] == "assign_task" and call["input"]["task_id"] == "T-101"
    assert p.next([{"role": "user", "content": "close T-103"}], [])[0]["name"] == "close_task"
    assert p.next([{"role": "user", "content": "hello"}], [])[0]["type"] == "text"


# ---- full graph (needs langgraph) ------------------------------------------------------
@pytest.fixture
def graph_env():
    pytest.importorskip("langgraph")
    from support_agent.graph import build_graph
    b = Backend()
    g = build_graph(ScriptedPlanner(), b, ORG)
    return g, b, {"configurable": {"thread_id": "t1"}}


def test_read_tool_runs_without_approval(graph_env):
    from support_agent.graph import user_turn
    g, b, cfg = graph_env
    out = g.invoke(user_turn("show open tasks"), cfg)
    assert "__interrupt__" not in out or not out["__interrupt__"]
    assert "T-101" in out["messages"][-1]["content"][0]["text"]


def test_email_waits_for_approval_and_can_be_declined(graph_env):
    from langgraph.types import Command
    from support_agent.graph import user_turn
    g, b, cfg = graph_env
    out = g.invoke(user_turn("email tenant@example.com the plumber comes Monday"), cfg)
    assert out["__interrupt__"][0].value["tool"] == "send_email"
    assert b.orgs[ORG].outbox == []  # nothing sent while waiting
    out = g.invoke(Command(resume={"approved": False}), cfg)
    assert b.orgs[ORG].outbox == []
    assert "did not approve" in out["messages"][-1]["content"][0]["text"]


def test_close_runs_once_after_approval(graph_env):
    from langgraph.types import Command
    from support_agent.graph import user_turn
    g, b, cfg = graph_env
    g.invoke(user_turn("close T-103"), cfg)
    g.invoke(Command(resume={"approved": True}), cfg)
    assert b.get_task(ORG, "T-103")["status"] == "closed"
    assert len(b.orgs[ORG].undo_stack) == 1  # executed exactly once
