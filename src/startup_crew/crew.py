"""Startup Crew: a hierarchical team of specialist agents managed by whichever
agent is flagged `manager: true` in config/agents.yaml (the Chief of Staff, by
default). The team is built entirely from config/agents.yaml and
config/tasks.yaml at runtime — add, remove, or edit agents in that file (or
via the web UI's Team tab) and they show up here with no code changes.

The LLM/provider is also generic (LLM_MODEL / LLM_API_KEY / LLM_BASE_URL env
vars) so it can be switched from the web UI's "LLM & Model" tab — see
app.py's PROVIDER_PRESETS.
"""

import os
from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task, LLM

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def get_llm() -> LLM:
    """Single LLM shared by every agent. Reads whichever provider/model is
    currently active in the environment (set via the web UI or .env)."""
    base_url = os.environ.get("LLM_BASE_URL", "").strip() or None
    kwargs = dict(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ.get("LLM_API_KEY") or None,
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.4")),
        # Without an explicit timeout, a stalled/hung connection to a flaky
        # free-tier endpoint can block a run indefinitely instead of failing
        # and letting CrewAI's own retry logic kick in.
        timeout=120,
    )
    if base_url:
        kwargs["base_url"] = base_url
    return LLM(**kwargs)


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


class StartupCrew:
    def __init__(self):
        self.agents_data = _load_yaml("agents.yaml")
        self.tasks_data = _load_yaml("tasks.yaml")

    def _build_agents(self) -> tuple[dict[str, Agent], str]:
        agents: dict[str, Agent] = {}
        manager_key = None
        for key, spec in self.agents_data.items():
            is_manager = bool(spec.get("manager"))
            if is_manager:
                manager_key = key
            agents[key] = Agent(
                role=spec["role"].strip(),
                goal=spec["goal"].strip(),
                backstory=spec["backstory"].strip(),
                llm=get_llm(),
                allow_delegation=is_manager,
                verbose=True,
            )
        if manager_key is None:
            # No agent explicitly flagged as manager: fall back to the first
            # one defined rather than failing outright.
            manager_key = next(iter(self.agents_data))
        return agents, manager_key

    def _build_tasks(self) -> list[Task]:
        built: dict[str, Task] = {}
        for key, spec in self.tasks_data.items():
            context_keys = spec.get("context", []) or []
            built[key] = Task(
                description=spec["description"].strip(),
                expected_output=spec["expected_output"].strip(),
                context=[built[c] for c in context_keys if c in built],
            )
        return list(built.values())

    def crew(self) -> Crew:
        agents, manager_key = self._build_agents()
        manager_agent = agents.pop(manager_key)
        tasks = self._build_tasks()
        return Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.hierarchical,
            manager_agent=manager_agent,
            verbose=True,
        )
