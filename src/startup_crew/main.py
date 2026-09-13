"""Entry point: run the startup crew against a single objective.

Usage:
    python -m src.startup_crew.main "Should we launch a free tier or stay paid-only?"
    python -m src.startup_crew.main   # prompts interactively if no argument given
"""

import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't print the emoji/arrows
# CrewAI's own logging emits — reconfigure to UTF-8 (with a safe fallback for
# any character that still can't render) before anything else prints.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from .crew import StartupCrew  # noqa: E402  (needs env vars loaded first)


def main() -> None:
    if len(sys.argv) > 1:
        objective = " ".join(sys.argv[1:])
    else:
        objective = input("What do you want the crew to work on?\n> ").strip()

    if not objective:
        print("No objective given, exiting.")
        return

    result = StartupCrew().crew().kickoff(inputs={"objective": objective, "history": ""})

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(result.raw)

    out_dir = Path(__file__).resolve().parents[2] / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "last_result.md"
    out_file.write_text(result.raw, encoding="utf-8")
    print(f"\n(also saved to {out_file})")


if __name__ == "__main__":
    main()
