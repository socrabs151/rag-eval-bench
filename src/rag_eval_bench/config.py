from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime paths, overridable through environment variables."""

    data_dir: Path
    artifacts_dir: Path
    config_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.environ.get("RAG_EVAL_DATA_DIR", "data")),
            artifacts_dir=Path(os.environ.get("RAG_EVAL_ARTIFACTS_DIR", "artifacts")),
            config_path=Path(os.environ.get("RAG_EVAL_CONFIG", "frontedge.toml")),
        )
