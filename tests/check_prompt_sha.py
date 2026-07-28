"""Verify prompt SHAs against the config lock (invoked by `make prompt_sha`).

Recomputes each agent prompt's SHA exactly as the framework does at decision
time (Orchestrator._compute_prompts_sha: frontmatter-stripped body, sha256,
first 16 hex chars) and compares against PROMPT_SHAS in config/config.yaml.
Exits non-zero on any mismatch. Not a pytest module.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.agents.base import load_prompt_template  # noqa: E402


def main() -> int:
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    locked_shas: dict[str, str] = config["PROMPT_SHAS"]
    versions: dict[str, int] = config["PROMPT_VERSIONS"]

    print(f"Prompt lock timestamp: {config['PROMPT_LOCK_TIMESTAMP']}")
    ok = True
    for agent, ver in versions.items():
        body = load_prompt_template(agent, ver)
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        status = "OK" if sha == locked_shas[agent] else "MISMATCH"
        ok &= status == "OK"
        print(f"  {agent}_v{ver}: computed {sha}  locked {locked_shas[agent]}  {status}")

    if ok:
        print("prompt_sha PASSED: all prompt hashes match the config lock.")
        return 0
    print("prompt_sha FAILED: at least one prompt file does not match its lock.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
