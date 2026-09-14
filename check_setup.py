#!/usr/bin/env python3
"""Startup Crew — end-to-end setup check.

Run this to find out, in one shot, whether the whole stack actually works:
Python version, dependencies, config, every API key, and a real crew run.

    python check_setup.py              # full check, including a live crew run
    python check_setup.py --quick      # skip the crew run (fast, no tokens used)

Works locally and on a hosted deploy. Exits 0 if everything passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- reporting

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_MARK = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]", INFO: "[..]  "}
_results: list[tuple[str, str, str]] = []


def say(status: str, title: str, detail: str = "") -> None:
    _results.append((status, title, detail))
    line = f"{_MARK[status]} {title}"
    print(line)
    for chunk in filter(None, detail.split("\n")):
        print(f"        {chunk}")


def section(name: str) -> None:
    print(f"\n--- {name} " + "-" * max(0, 62 - len(name)))


# ---------------------------------------------------------------- 1. python


def check_python() -> bool:
    v = sys.version_info
    label = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 14):
        say(FAIL, f"Python {label} is too new",
            "CrewAI declares requires-python <3.14, and its chromadb dependency\n"
            "crashes on import under 3.14 with:\n"
            '  unable to infer type for attribute "chroma_server_nofile"\n'
            "Pinning pydantic does NOT help. Use Python 3.13 or older.\n"
            "On Streamlit Cloud the version cannot be changed after deployment:\n"
            "delete the app and redeploy, choosing Python 3.13.")
        return False
    if v < (3, 10):
        say(FAIL, f"Python {label} is too old", "CrewAI needs >=3.10.")
        return False
    say(PASS, f"Python {label}", "Supported (>=3.10, <3.14).")
    return True


# ---------------------------------------------------------- 2. dependencies


def check_imports() -> bool:
    required = ["streamlit", "crewai", "yaml", "dotenv", "litellm"]
    ok = True
    for mod in required:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "")
            say(PASS, f"import {mod}", f"version {ver}" if ver else "")
        except Exception as exc:  # noqa: BLE001
            say(FAIL, f"import {mod}", f"{type(exc).__name__}: {exc}")
            ok = False
    # chromadb is the one that breaks on 3.14 — import it explicitly so the
    # failure is attributed here rather than surfacing mid-crew.
    try:
        import chromadb  # noqa: F401
        from chromadb.config import Settings

        Settings()
        say(PASS, "import chromadb", f"version {chromadb.__version__}")
    except Exception as exc:  # noqa: BLE001
        say(FAIL, "import chromadb", f"{type(exc).__name__}: {exc}")
        ok = False
    return ok


# ---------------------------------------------------------------- 3. config


def load_config() -> list[dict]:
    """Return the profile chain from whichever source the app would use."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass

    # Streamlit secrets, when running inside a deployed app context.
    try:
        import streamlit as st

        for k, v in st.secrets.items():
            if isinstance(v, str) and not os.environ.get(k):
                os.environ[k] = v
    except Exception:  # noqa: BLE001
        pass

    profiles_file = ROOT / "llm_profiles.json"
    if profiles_file.exists():
        try:
            data = json.loads(profiles_file.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                say(PASS, "config source: llm_profiles.json",
                    f"{len(data)} profile(s)")
                return [p for p in data if p.get("model")]
        except json.JSONDecodeError as exc:
            say(FAIL, "llm_profiles.json is not valid JSON", str(exc))

    raw = os.environ.get("LLM_PROFILES", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                say(PASS, "config source: LLM_PROFILES secret",
                    f"{len(data)} profile(s)")
                return [p for p in data if p.get("model")]
        except json.JSONDecodeError as exc:
            say(FAIL, "LLM_PROFILES is not valid JSON", str(exc))

    if os.environ.get("LLM_MODEL"):
        say(WARN, "config source: single LLM_MODEL / LLM_API_KEY",
            "No profile chain found, so there is no failover if this key fails.")
        return [{
            "label": os.environ.get("LLM_PROVIDER_LABEL", "Primary"),
            "model": os.environ["LLM_MODEL"],
            "base_url": os.environ.get("LLM_BASE_URL", ""),
            "api_key": os.environ.get("LLM_API_KEY", ""),
            "temperature": os.environ.get("LLM_TEMPERATURE", "0.4"),
        }]

    say(FAIL, "no LLM configuration found",
        "Expected llm_profiles.json, an LLM_PROFILES secret, or LLM_MODEL in .env.")
    return []


# ------------------------------------------------------------------ 4. keys


# litellm model strings are "<provider>/<the provider's own model id>".
# A raw HTTP call to the provider must send only the second part — sending
# the whole thing is what makes NVIDIA answer "404 page not found".
LITELLM_PREFIXES = {
    "openrouter", "nvidia_nim", "openai", "anthropic", "groq", "mistral",
    "deepseek", "together_ai", "fireworks_ai", "azure", "vertex_ai",
    "gemini", "cohere", "perplexity", "xai", "ollama", "bedrock",
}

# litellm reads the key from a provider-specific env var as well.
PROVIDER_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def split_model(model: str) -> tuple[str, str]:
    """('nvidia_nim/nvidia/nemotron-x') -> ('nvidia_nim', 'nvidia/nemotron-x')"""
    head, sep, rest = model.partition("/")
    if sep and head in LITELLM_PREFIXES:
        return head, rest
    return "", model


def _post(url: str, key: str, payload: dict, timeout: int = 45):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def check_keys(profiles: list[dict]) -> int:
    """Send a 1-token completion through each profile. Returns live count."""
    live = 0
    for i, p in enumerate(profiles, 1):
        label = p.get("label") or f"profile {i}"
        key = p.get("api_key", "")
        base = (p.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
        model = p.get("model", "")
        _prov, api_model = split_model(model)
        if not key:
            say(WARN, f"{label}: no API key set", "Skipped.")
            continue
        started = time.time()
        try:
            status, body = _post(
                f"{base}/chat/completions", key,
                {"model": api_model,
                 "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                 "max_tokens": 5},
            )
            took = time.time() - started
            text = ""
            try:
                text = body["choices"][0]["message"]["content"].strip()
            except Exception:  # noqa: BLE001
                pass
            if status == 200 and text:
                say(PASS, f"{label}: key works",
                    f"model replied {text!r} in {took:.1f}s  (...{key[-6:]})")
                live += 1
            else:
                say(FAIL, f"{label}: unexpected response",
                    json.dumps(body)[:300])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            # 429 is not a broken key — it is the free tier doing what it does,
            # and it is exactly what the profile chain exists to route around.
            if exc.code == 429:
                say(WARN, f"{label}: rate limited (HTTP 429)",
                    detail + "\nNormal on the free tier. The chain falls through "
                             "to the next key, so this is not a setup problem.")
                continue
            hint = ""
            if exc.code == 404:
                hint = (f"\nThe endpoint did not recognise model "
                        f"{api_model!r} at {base}.\n"
                        "Check the model id against what this provider serves.")
            if exc.code == 401:
                hint = "\nKey is invalid or revoked — create a new one at openrouter.ai/keys."
            elif exc.code == 402:
                hint = "\nOut of credits for this key."
            say(FAIL, f"{label}: HTTP {exc.code}", detail + hint)
        except Exception as exc:  # noqa: BLE001
            say(FAIL, f"{label}: {type(exc).__name__}", str(exc)[:300])
    return live


# ------------------------------------------------------------- 5. crew test


def check_crew(profiles: list[dict]) -> bool:
    if not profiles:
        return False
    p = profiles[0]
    model = p.get("model", "")
    os.environ["LLM_MODEL"] = model
    os.environ["LLM_API_KEY"] = p.get("api_key", "")
    os.environ["LLM_BASE_URL"] = p.get("base_url", "")
    # litellm reads provider keys from the conventional env var too.
    prov, _api_model = split_model(model)
    if prov in PROVIDER_ENV:
        os.environ[PROVIDER_ENV[prov]] = p.get("api_key", "")

    say(INFO, "running a real 1-agent crew (may take 30-90s on the free tier)")
    try:
        from crewai import Agent, Crew, Task, LLM

        llm = LLM(model=model, api_key=p.get("api_key", ""),
                  base_url=p.get("base_url") or None, temperature=0.2)
        agent = Agent(
            role="Setup Tester",
            goal="Confirm the toolchain runs end to end.",
            backstory="You answer in as few words as possible.",
            llm=llm, verbose=False, allow_delegation=False,
        )
        task = Task(
            description="Reply with exactly this and nothing else: SETUP OK",
            expected_output="The text SETUP OK",
            agent=agent,
        )
        started = time.time()
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        took = time.time() - started
        text = str(result).strip()
        if "SETUP OK" in text.upper():
            say(PASS, "crew ran end to end", f"returned {text!r} in {took:.0f}s")
            return True
        say(WARN, "crew ran but the answer was unexpected",
            f"got {text[:200]!r} in {took:.0f}s\n"
            "The pipeline works; the free model just wandered off-script.")
        return True
    except Exception as exc:  # noqa: BLE001
        say(FAIL, f"crew run failed: {type(exc).__name__}", str(exc)[:600])
        return False


# ------------------------------------------------------------------- runner


def main() -> int:
    ap = argparse.ArgumentParser(description="Startup Crew setup check")
    ap.add_argument("--quick", action="store_true",
                    help="skip the live crew run")
    args = ap.parse_args()

    print("=" * 70)
    print("Startup Crew — setup check")
    print(f"folder: {ROOT}")
    print("=" * 70)

    section("1. Python version")
    py_ok = check_python()

    section("2. Dependencies")
    deps_ok = check_imports() if py_ok else (
        say(WARN, "skipped", "Fix the Python version first.") or False)

    section("3. Configuration")
    profiles = load_config()
    for i, p in enumerate(profiles, 1):
        key = p.get("api_key", "")
        masked = f"...{key[-6:]}" if key else "(none)"
        print(f"        {i}. {p.get('label','?')}  {p.get('model','?')}  {masked}")

    section("4. API keys (live calls)")
    live = check_keys(profiles) if profiles else 0
    if profiles:
        say(INFO, f"{live} of {len(profiles)} key(s) answered")

    section("5. Crew end-to-end")
    if args.quick:
        say(INFO, "skipped (--quick)")
    elif deps_ok and live:
        check_crew([p for p in profiles if p.get("api_key")])
    else:
        say(WARN, "skipped", "Needs working dependencies and at least one live key.")

    section("Summary")
    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    for _, title, _d in fails:
        print(f"  FAIL  {title}")
    for _, title, _d in warns:
        print(f"  warn  {title}")
    if not fails:
        print("\n  Everything that matters passed."
              + ("  (with warnings above)" if warns else ""))
        return 0
    print(f"\n  {len(fails)} check(s) failed — see the detail above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
