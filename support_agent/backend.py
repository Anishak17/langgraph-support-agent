"""A fake property-management backend: tasks, staff, outgoing email and support tickets.

Every method takes org_id, and data is stored per organization, so one customer's agent
can never see or change another customer's data. Changes record a snapshot so they can
be undone.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone


class NotFound(KeyError):
    pass


@dataclass
class Task:
    task_id: str
    title: str
    property: str
    status: str = "open"          # open | closed
    assignee: str | None = None


@dataclass
class Org:
    org_id: str
    staff: set[str]
    tasks: dict[str, Task]
    outbox: list[dict] = field(default_factory=list)
    tickets: list[dict] = field(default_factory=list)
    undo_stack: list[tuple[str, Task]] = field(default_factory=list)  # (description, previous task)


def seed() -> dict[str, Org]:
    return {
        "acme-housing": Org(
            "acme-housing",
            staff={"sanne@acme.example", "daan@acme.example"},
            tasks={t.task_id: t for t in [
                Task("T-101", "Leaking tap in kitchen", "Kerkstraat 12-A"),
                Task("T-102", "Broken intercom", "Kerkstraat 12", assignee="daan@acme.example"),
                Task("T-103", "Annual boiler service", "Dorpsweg 4"),
                Task("T-104", "Replace hallway light", "Dorpsweg 4", status="closed"),
            ]},
        ),
        "globex-estates": Org(
            "globex-estates",
            staff={"lena@globex.example"},
            tasks={t.task_id: t for t in [
                Task("T-201", "Roof terrace door sticks", "Havenkade 1"),
                Task("T-202", "Elevator inspection", "Havenkade 1"),
            ]},
        ),
    }


class Backend:
    def __init__(self, orgs: dict[str, Org] | None = None):
        self.orgs = orgs if orgs is not None else seed()
        self._ticket_ids = itertools.count(5001)

    def _org(self, org_id: str) -> Org:
        if org_id not in self.orgs:
            raise NotFound(f"unknown organization {org_id}")
        return self.orgs[org_id]

    def _task(self, org_id: str, task_id: str) -> Task:
        task = self._org(org_id).tasks.get(task_id)
        if task is None:  # also what you get for another org's task id
            raise NotFound(f"task {task_id} not found")
        return task

    # ---- reads -----------------------------------------------------------
    def search_tasks(self, org_id: str, status: str | None = None, assignee: str | None = None,
                     text: str | None = None) -> list[dict]:
        out = []
        for t in self._org(org_id).tasks.values():
            if status and t.status != status:
                continue
            if assignee and t.assignee != assignee:
                continue
            if text and text.lower() not in f"{t.title} {t.property}".lower():
                continue
            out.append(vars(t).copy())
        return out

    def get_task(self, org_id: str, task_id: str) -> dict:
        return vars(self._task(org_id, task_id)).copy()

    # ---- writes (undoable) -------------------------------------------------
    def _snapshot(self, org_id: str, task: Task, description: str) -> None:
        self._org(org_id).undo_stack.append((description, copy.deepcopy(task)))

    def assign_task(self, org_id: str, task_id: str, assignee: str) -> dict:
        org, task = self._org(org_id), self._task(org_id, task_id)
        if assignee not in org.staff:
            raise ValueError(f"{assignee} is not a staff member of this organization")
        self._snapshot(org_id, task, f"assign {task_id} to {assignee}")
        task.assignee = assignee
        return vars(task).copy()

    def close_task(self, org_id: str, task_id: str) -> dict:
        task = self._task(org_id, task_id)
        if task.status == "closed":
            raise ValueError(f"task {task_id} is already closed")
        self._snapshot(org_id, task, f"close {task_id}")
        task.status = "closed"
        return vars(task).copy()

    def undo_last(self, org_id: str) -> dict:
        org = self._org(org_id)
        if not org.undo_stack:
            raise ValueError("there is nothing to undo")
        description, previous = org.undo_stack.pop()
        org.tasks[previous.task_id] = previous
        return {"undone": description, "task": vars(previous).copy()}

    # ---- writes (not undoable) ----------------------------------------------
    def send_email(self, org_id: str, to: str, subject: str, body: str) -> dict:
        msg = {"to": to, "subject": subject, "body": body,
               "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._org(org_id).outbox.append(msg)
        return {"sent": True, "to": to, "subject": subject}

    def create_ticket(self, org_id: str, summary: str) -> dict:
        ticket = {"ticket_id": next(self._ticket_ids), "summary": summary, "status": "new"}
        self._org(org_id).tickets.append(ticket)
        return ticket
