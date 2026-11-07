from __future__ import annotations

from pathlib import Path

import frontedge
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .config import Settings
from .providers import DatasetsProvider, EvalsProvider, RunsProvider
from .storage import Store


def create_app(
    store: Store | None = None,
    *,
    config_path: Path | str | None = None,
) -> FastAPI:
    settings = Settings.from_env()
    data_store = store or Store(settings.data_dir, settings.artifacts_dir)
    config_file = Path(config_path) if config_path is not None else settings.config_path
    config = config_file if config_file.is_file() else None

    application = FastAPI(title="RAG Eval Bench", version="0.1.0")
    application.include_router(
        frontedge.mount(
            RunsProvider(data_store),
            evals=EvalsProvider(data_store),
            datasets=DatasetsProvider(data_store),
            config=config,
            title="RAG Eval Bench",
        )
    )

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/frontedge")

    return application
