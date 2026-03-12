"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.data.database import get_connection, init_schema
from backend.api.routes import router as api_router
from backend.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and resources on startup."""
    conn = get_connection()
    init_schema(conn)
    app.state.db = conn
    yield
    conn.close()


app = FastAPI(
    title="Crypto ML Prediction Dashboard",
    description="Live crypto perpetual futures prediction signals using AFML methodology",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
