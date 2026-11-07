from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

DEFAULT_REPO = "socrabs151/frontedge-ml"
DEFAULT_DOC = Path(__file__).resolve().parents[1] / "docs" / "frontedge-feedback.md"


def _token() -> str:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("No GitHub token: set GITHUB_TOKEN or store credentials for github.com")
    for line in result.stdout.splitlines():
        if line.startswith("password="):
            return line.removeprefix("password=")
    raise SystemExit("No GitHub password in the credential helper output")


def _entry_section(path: Path, entry_id: str) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"### {entry_id}"
    start = next((index for index, line in enumerate(lines) if line.startswith(prefix)), None)
    if start is None:
        raise SystemExit(f"Entry not found in {path}: {entry_id}")
    heading = lines[start].removeprefix(prefix).lstrip(": ").strip()
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("### ") or lines[index].startswith("## ")
        ),
        len(lines),
    )
    body = "\n".join(lines[start + 1 : end]).strip()
    return heading, body


def _create_issue(repo: str, title: str, body: str, labels: list[str], token: str) -> str:
    payload: dict[str, object] = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "rag-eval-bench",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub API error {error.code}: {detail}") from error
    return str(result["html_url"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="File one frontedge feedback entry as a GitHub issue."
    )
    parser.add_argument("--entry", help="entry id in the feedback log, for example F-002")
    parser.add_argument("--title", help="issue title; defaults to the entry heading")
    parser.add_argument("--body", help="issue body; defaults to the entry section")
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_DOC,
        help="path to the feedback log",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="target repository")
    parser.add_argument("--label", action="append", default=[], help="issue label")
    parser.add_argument("--dry-run", action="store_true", help="print the request only")
    args = parser.parse_args(argv)

    title = args.title
    body = args.body
    if args.entry:
        entry_title, entry_body = _entry_section(args.doc, args.entry)
        title = title or entry_title
        body = body or entry_body
    if not title or not body:
        raise SystemExit("Provide --entry or both --title and --body")

    if args.dry_run:
        print(json.dumps({"repo": args.repo, "title": title, "body": body}, indent=2))
        return 0

    url = _create_issue(args.repo, title, body, args.label, _token())
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
