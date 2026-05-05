import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.runtime_config import runtime_config
from app.core.model_cache import preload_model
from app.routers import image, video, stream
from app.routers.config_router import router as config_router
from app.routers.export_router import router as export_router
from app.routers.health_router import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    snap = runtime_config.snapshot()
    os.makedirs(snap["upload_dir"], exist_ok=True)
    os.makedirs(snap["output_dir"], exist_ok=True)
    preload_model(snap["model_path"])
    yield


app = FastAPI(
    title="Chicken Wing Counter",
    version="2.0.0",
    description=(
        "Production-grade wing counting API. "
        "Supports live video streams and uploaded files. "
        "All inference parameters are tunable at runtime via PATCH /api/config "
        "without restarting the container."
    ),
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(config_router)
app.include_router(export_router)
app.include_router(image.router)
app.include_router(video.router)
app.include_router(stream.router)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
