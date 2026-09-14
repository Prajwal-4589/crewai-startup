#!/usr/bin/env python3
"""Point the Startup Crew at the model provider OpenCode already uses.

OpenCode keeps its provider credentials in auth.json. This reads that file,
asks the provider which models it will actually serve you, verifies one with
a real call, and writes it into llm_profiles.json as the crew's first-choice
profile — ahead of the free-tier keys, which stay as fallback.

    python connect_opencode.py                 # auto-pick the best model
    python connect_opencode.py --list          # just show what is available
    python connect_opencode.py --model NAME    # force a specific model id
    python connect_opencode.py --dry-run       # show the change, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "llm_profiles.json"
ENV_PATH = ROOT / ".env"

# provider name in OpenCode's auth.json -> (litellm prefix, api base)
PROVIDERS = {
    "nvidia":     ("nvidia_nim", "https://integrate.api.nvidia.com/v1"),
    "openrouter": ("openrouter", "https://openrouter.ai/api/v1"),
    "openai":     ("",           "https://api.openai.com/v1"),
    "anthropic":  ("anthropic",  "https://api.anthropic.com/v1"),
    "groq":       ("groq",       "https://api.groq.com/openai/v1"),
    "mistral":    ("mistral",    "https://api.mistral.ai/v1"),
    "deepseek":   ("deepseek",   "https://api.deepseek.com/v1"),
}

# Preference order when auto-picking. Earlier = better for this workload
# (long multi-agent reasoning, tolerant of latency).
PREFER = ["nemotron-3-ultra", "nemotron-ultra", "nemotron-super", "nemotron",
          "llama-3.3-70b", "llama-3.1-405b", "qwen2.5-72b", "deepseek"]


def die(msg: str) -> None:
    print(f"\n  ERROR: {msg}\n")
    sys.exit(1)


def find_auth() -> Path:
    home = Path.home()
    candidates = [
        home / ".local" / "share" / "opencode" / "auth.json",
        home / ".config" / "opencode" / "auth.json",
        Path(os.environ.get("XDG_DATA_HOME", "")) / "opencode" / "auth.json",
        Path(os.environ.get("APPDATA", "")) / "opencode" / "auth.json",
    ]
    for c in candidates:
        if str(c) != "auth.json" and c.is_file():
            return c
    die("could not find OpenCode's auth.json.\n"
        "  Looked in:\n    " + "\n    ".join(str(c) for c in candidates) +
        "\n  Run `opencode auth login` first, or pass --auth <path>.")
    raise AssertionError  # unreachable


def read_credentials(auth_file: Path) -> tuple[str, str, str, str]:
    try:
        data = json.loads(auth_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        die(f"could not read {auth_file}: {exc}")
    usable = []
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if entry.get("type") == "api" and isinstance(key, str) and key:
            usable.append((name, key))
        elif "access" in entry:
            print(f"  note: '{name}' uses an OAuth login, not an API key — "
                  "it cannot be reused outside OpenCode. Skipping.")
    if not usable:
        die("no reusable API keys in OpenCode's auth.json.\n"
            "  Subscription/OAuth logins (Claude Pro, ChatGPT) only work "
            "inside OpenCode itself.")
    name, key = usable[0]
    if len(usable) > 1:
        print(f"  found {len(usable)} providers; using '{name}'. "
              f"Others: {', '.join(n for n, _ in usable[1:])}")
    prefix, base = PROVIDERS.get(name, ("", ""))
    if not base:
        die(f"provider '{name}' is not mapped yet. Add it to PROVIDERS.")
    return name, key, prefix, base


def list_models(base: str, key: str) -> list[str]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:200]
        if exc.code == 401:
            die(f"the key from OpenCode was rejected (401).\n  {detail}")
        die(f"HTTP {exc.code} listing models.\n  {detail}")
    except Exception as exc:  # noqa: BLE001
        die(f"could not reach {base}: {type(exc).__name__}: {exc}")
    return sorted(m.get("id", "") for m in body.get("data", []) if m.get("id"))


def pick(models: list[str]) -> str:
    for want in PREFER:
        for m in models:
            if want in m.lower():
                return m
    if not models:
        die("the provider returned no models.")
    return models[0]


def verify(base: str, key: str, model_id: str) -> str:
    payload = {"model": model_id,
               "messages": [{"role": "user", "content": "Reply with: ok"}],
               "max_tokens": 5}
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        die(f"model {model_id!r} did not answer: HTTP {exc.code}\n"
            f"  {exc.read().decode()[:300]}\n"
            "  Try --list to see what this key can actually serve.")
    except Exception as exc:  # noqa: BLE001
        die(f"verification call failed: {type(exc).__name__}: {exc}")
    raise AssertionError  # unreachable


def write_profile(profile: dict, dry: bool) -> None:
    existing = []
    if PROFILES.exists():
        try:
            existing = json.loads(PROFILES.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    # Replace any previous OpenCode-sourced profile rather than piling up.
    existing = [p for p in existing if p.get("id") != profile["id"]]
    merged = [profile] + existing

    print("\n  new profile chain:")
    for i, p in enumerate(merged, 1):
        star = "  <- first choice" if i == 1 else ""
        print(f"    {i}. {p.get('label','?')}  [{p.get('model','?')}]{star}")

    if dry:
        print("\n  --dry-run: nothing written.")
        return

    PROFILES.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\n  wrote {PROFILES.name}")

    # Keep .env aligned so the CLI entry point uses the same model.
    if ENV_PATH.exists():
        lines, seen = [], set()
        updates = {
            "LLM_PROVIDER_LABEL": profile["label"],
            "LLM_MODEL": profile["model"],
            "LLM_API_KEY": profile["api_key"],
            "LLM_BASE_URL": profile["base_url"],
        }
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            k = raw.split("=", 1)[0].strip()
            if k in updates:
                lines.append(f"{k}={updates[k]}")
                seen.add(k)
            else:
                lines.append(raw)
        for k, v in updates.items():
            if k not in seen:
                lines.append(f"{k}={v}")
        ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {ENV_PATH.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", help="path to OpenCode's auth.json")
    ap.add_argument("--model", help="force a specific model id")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    args = ap.parse_args()

    print("=" * 68)
    print("Connect Startup Crew to OpenCode's provider")
    print("=" * 68)

    auth_file = Path(args.auth) if args.auth else find_auth()
    print(f"  auth file: {auth_file}")
    name, key, prefix, base = read_credentials(auth_file)
    print(f"  provider : {name}  ({base})")
    print(f"  key      : ...{key[-6:]}")

    models = list_models(base, key)
    print(f"  models   : {len(models)} available")

    if args.list:
        for m in models:
            print(f"    {m}")
        return 0

    model_id = args.model or pick(models)
    if args.model and args.model not in models:
        print(f"  warning: {args.model!r} is not in the provider's list; "
              "trying it anyway.")
    print(f"  chosen   : {model_id}")

    reply = verify(base, key, model_id)
    print(f"  verified : model replied {reply!r}")

    profile = {
        "id": "opencode",            # stable, so re-running replaces in place
        "label": f"OpenCode / {name} — {model_id.split('/')[-1]}",
        "model": f"{prefix}/{model_id}" if prefix else model_id,
        "base_url": base,
        "api_key": key,
        "temperature": 0.4,
    }
    write_profile(profile, args.dry_run)

    if not args.dry_run:
        print("\n  Done. Restart the app (or just send a message) and the crew\n"
              "  will use this first, falling back to your old keys if it fails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
