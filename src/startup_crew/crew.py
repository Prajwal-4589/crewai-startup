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
        # Reasoning models (Nemotron 3 Ultra and friends) emit a long chain of
        # thinking tokens before any answer, so a single agent call can run for
        # many minutes. A short timeout here does not "fail fast" — it aborts
        # good runs and burns the whole fallback chain on a model that was
        # working. Generous by default; override with LLM_TIMEOUT (seconds).
        timeout=float(os.environ.get("LLM_TIMEOUT", "900")),
    )
    if base_url:
        kwargs["base_url"] = base_url

    # litellm also looks for a provider-specific env var, and falls back to it
    # when api_key is None. Export it so a provider that ignores the explicit
    # kwarg still authenticates instead of failing with a bare
    # "Missing Authentication header".
    _provider_env = {
        "nvidia_nim": "NVIDIA_NIM_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    _prefix = kwargs["model"].split("/", 1)[0]
    if kwargs.get("api_key") and _prefix in _provider_env:
        os.environ[_provider_env[_prefix]] = kwargs["api_key"]

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
