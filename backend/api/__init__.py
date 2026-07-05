"""FastAPI application entrypoint.

Only the quality router is mounted for now. Future stage routers
(preprocessing, acne, pigmentation, ...) mount here the same way —
this file never needs restructuring as modules are added.
"""

from __future__ import annotations

from fastapi import FastAPI

from .quality_router import router as quality_router

app = FastAPI(
    title="Skin Analysis API",
    version="0.1.0",
    description="Modular AI skin analysis pipeline.",
)

app.include_router(quality_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}