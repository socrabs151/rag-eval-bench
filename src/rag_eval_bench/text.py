from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize(text: str) -> str:
    return " ".join(tokenize(text))
