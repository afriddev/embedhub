import os
import asyncio
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
import torch
from services import EmbeddingService, RerankerService
from controllers import EmbeddingController, RerankController
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

port = int(os.getenv("PORT", "8000"))

embeddingService = EmbeddingService()
rerankerService = RerankerService()
embeddingController = EmbeddingController(embeddingService)
crossEncoderRerankerController = RerankController(rerankerService)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(embeddingService.LoadModel)
    await rerankerService.LoadModel()
    await embeddingService.batcher.start()
    await rerankerService.batcher.start()

    try:
        if (
            torch.cuda.is_available()
            and getattr(__import__("services").EmbeddingService, "keepaliveTask", None)
            is None
        ):
            loop = asyncio.get_running_loop()
            __import__("services").EmbeddingService.keepaliveTask = loop.create_task(
                embeddingService.GpuKeepAlive()
            )
    except Exception:
        pass

    try:
        yield
    finally:
        if embeddingService.batcher.task is not None:
            embeddingService.batcher.task.cancel()
        if rerankerService.batcher.task is not None:
            rerankerService.batcher.task.cancel()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=800)
app.include_router(embeddingController.router, prefix="/api/v1")
app.include_router(crossEncoderRerankerController.router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=300,
    )
