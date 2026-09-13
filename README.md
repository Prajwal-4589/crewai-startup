# Startup Crew

A CrewAI multi-agent team that helps run your startup, wrapped in a local
chat-style control panel (`app.py`) — no code editing required for
day-to-day use.

## The team (default roster)

| Agent | Role |
|---|---|
| Chief of Staff & Head of Strategy | **Manager.** Reads your message, decides who's relevant, delegates, resolves conflicts, synthesizes the final answer. |
| Market & Competitive Intelligence Analyst | Market size, segments, competitors, trends. |
| Product Manager | What to build, what to defer, sequencing. |
| Growth & Marketing Lead | Go-to-market, channels, growth experiments. |
| Sales & Business Development Lead | Pipeline, buyer, pitch, partnerships. |
| Finance & Operations Analyst | Cost, runway, unit economics, ops bottlenecks. |
| Customer Success & Insights Lead | User reaction, adoption friction, retention/churn risk. |

This roster is just the starting point — **add, edit, or remove anyone, and
reassign who manages the team, entirely from the web UI's Team tab.** Nothing
is hardcoded: `crew.py` builds the crew from `src/startup_crew/config/agents.yaml`
at runtime, so changes there (via the UI or by hand) take effect on the next
message automatically.

## Setup (already done if you set this up with Claude)

```bash
uv venv --python 3.12
uv pip install --python .venv "crewai[litellm]" crewai-tools python-dotenv streamlit
```

Your active LLM/provider lives in `.env` as generic `LLM_MODEL` /
`LLM_API_KEY` / `LLM_BASE_URL` values — set these from the web UI's "LLM &
Model" tab rather than editing `.env` by hand.

## Run the web UI

```bash
cd /c/Users/Prajwal/crewai-startup
.venv/Scripts/streamlit run app.py
```

Opens at `http://localhost:8501` as a chat app:

- **Chats (sidebar)** — each one is a separate "project" with its own memory.
  Click **✚ New chat** to start a new thread, click any past chat to reopen
  it, use **⋮** to rename or delete one. The crew sees everything said
  earlier *in that chat* as context, so follow-ups build on prior answers
  instead of starting cold. Every message, answer, and the full agent
  delegation log are saved immediately to `projects/<id>.json` — nothing is
  lost on refresh or restart.
- **🗂 All projects** — browse/search every chat ever started (not just the
  most recent ones in the sidebar), with created/updated timestamps, open,
  and delete.
- **🧑‍🤝‍🧑 Team** — add/edit/remove team members (role, goal,
  backstory/personality), and reassign the manager. No code changes, ever.
- **🔌 LLM & Model** — a priority-ordered **chain** of models/keys
  (`llm_profiles.json`). Add multiple OpenRouter keys for the same free
  model to dodge rate limits, or mix providers (OpenAI, Anthropic, Gemini,
  local Ollama, custom endpoints). If #1 fails or times out, the crew
  automatically retries the whole run with #2, and so on — reorder, edit,
  test, or delete any entry. Takes effect immediately, no restart.
- **ℹ️ About** — quick reference.

## Run it from the terminal instead (single-shot, no memory)

```bash
.venv/Scripts/python -m src.startup_crew.main "Should we launch a free tier or stay paid-only?"
```

Or with no argument, to be prompted interactively:

```bash
.venv/Scripts/python -m src.startup_crew.main
```

## Notes on the free-tier model

The default provider (Nemotron 3 Ultra via OpenRouter's free tier) works, but
can be slow or occasionally return an "overloaded" / timeout error — that's
the shared free endpoint, not a bug here. If that gets in your way, use
**🔌 LLM & Model** to switch to a paid provider/model for faster, more
reliable responses.

## Extending it further

- **Give agents real tools** (web search, file/CSV reading, etc.): add
  packages from `crewai_tools` (e.g. `SerperDevTool` for web search — needs
  its own API key from serper.dev) to an agent's `tools=[...]` in `crew.py`'s
  `_build_agents`.
- **Change what the crew is asked**: edit `src/startup_crew/config/tasks.yaml`
  — add more tasks, or change the existing ones. `{history}` is the running
  conversation memory for the current chat; `context: [task_key, ...]` wires
  one task's output into another's input.

## Check the setup

```bash
.venv/Scripts/python check_setup.py            # Linux/macOS: .venv/bin/python
.venv/Scripts/python check_setup.py --quick    # skip the live crew run
```

Checks the Python version, every dependency, where the config is coming
from, each API key with a real call, and finally runs a one-agent crew end
to end. Exits 0 when everything passes. Run it locally, or from the hosted
app's terminal, to tell a broken key apart from a broken install.

## Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick this repo, branch `main`, main file `app.py`.
   Under **Advanced settings**, set **Python 3.13**. This is not optional:
   CrewAI declares `requires-python <3.14`, and on Python 3.14 its chromadb
   dependency dies on import with
   `unable to infer type for attribute "chroma_server_nofile"`. Pinning
   pydantic does not help — the break is in 3.14 itself. Community Cloud
   defaults to the newest Python, and **the version cannot be changed after
   deployment** — you have to delete the app and redeploy to change it.
3. Open **Advanced settings → Secrets** and paste the contents of
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with
   your real key(s) filled in. `app.py` copies those into the environment at
   startup, so no `.env` file is needed in the cloud.

   Set **`LLM_PROFILES`** there too (a JSON array, see the example file) and
   the whole priority chain — several keys for one free model, or a mix of
   providers — is rebuilt on first run, so the **🔌 LLM & Model** tab comes
   up already populated instead of empty. It is read only when
   `llm_profiles.json` is absent, so edits you make in the UI are never
   overwritten.
4. Deploy. Dependencies come from `requirements.txt`.

**Caveats on a hosted deployment:**

- Streamlit Cloud's filesystem is ephemeral. Chats saved to `projects/`,
  team edits written to `agents.yaml`, and LLM profiles in
  `llm_profiles.json` are lost when the app sleeps or redeploys. For
  durable state, point those paths at S3/a database, or keep running it
  locally.
- Community Cloud apps are public by default — anyone with the URL can use
  your API keys. Restrict viewers in the app's settings, or deploy
  somewhere private, before sharing the link.
- CrewAI runs can take minutes; the free tier's 1 GB memory limit is tight
  for large crews.

## Local setup from a fresh clone

```bash
git clone https://github.com/<your-user>/crewai-startup.git
cd crewai-startup
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # Linux/macOS: .venv/bin/pip
cp .env.example .env                             # then fill in LLM_API_KEY
cp llm_profiles.example.json llm_profiles.json
.venv/Scripts/streamlit run app.py
```

`.env` and `llm_profiles.json` hold API keys and are gitignored — keep it
that way.
