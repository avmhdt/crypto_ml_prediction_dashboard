"""REST API routes for the dashboard."""
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.config import BAR_TYPES, LABELING_METHODS, SYMBOLS, TripleBarrierConfig
from backend.data.database import load_bars, load_signals

router = APIRouter(prefix="/api")

# Mutable config state
_barrier_config = TripleBarrierConfig()


class BarrierConfigUpdate(BaseModel):
    sl_mult: float | None = None
    pt_mult: float | None = None
    max_hold: int | None = None


@router.get("/symbols")
async def get_symbols():
    return SYMBOLS


@router.get("/config")
async def get_config():
    return {
        "bar_types": BAR_TYPES,
        "labeling_methods": LABELING_METHODS,
        "symbols": SYMBOLS,
    }


@router.get("/bars/{symbol}/{bar_type}")
async def get_bars(
    request: Request,
    symbol: str,
    bar_type: str,
    limit: int = Query(default=500, le=5000),
    start: int | None = Query(default=None),
    end: int | None = Query(default=None),
):
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    if bar_type not in BAR_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid bar type: {bar_type}")

    conn = request.app.state.db
    df = load_bars(conn, symbol, bar_type, start_time=start, end_time=end, limit=limit)

    return df.to_dict(orient="records")


@router.get("/signals/{symbol}")
async def get_signals(
    request: Request,
    symbol: str,
    bar_type: str | None = Query(default=None),
    labeling: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
):
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    conn = request.app.state.db
    df = load_signals(conn, symbol, bar_type=bar_type,
                      labeling_method=labeling, limit=limit)

    return df.to_dict(orient="records")


@router.post("/config/barriers")
async def update_barriers(update: BarrierConfigUpdate):
    global _barrier_config
    if update.sl_mult is not None:
        _barrier_config.sl_multiplier = update.sl_mult
    if update.pt_mult is not None:
        _barrier_config.pt_multiplier = update.pt_mult
    if update.max_hold is not None:
        _barrier_config.max_holding_period = update.max_hold
    return {"status": "updated"}


@router.get("/metrics/{symbol}")
async def get_metrics(
    request: Request,
    symbol: str,
    bar_type: str | None = Query(default=None),
    labeling: str | None = Query(default=None),
):
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    conn = request.app.state.db
    df = load_signals(conn, symbol, bar_type=bar_type,
                      labeling_method=labeling, limit=1000)

    if df.empty:
        return {
            "total_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "avg_meta_prob": 0.0,
            "avg_bet_size": 0.0,
        }

    return {
        "total_signals": len(df),
        "long_signals": int((df["side"] == 1).sum()),
        "short_signals": int((df["side"] == -1).sum()),
        "avg_meta_prob": float(df["meta_probability"].mean()) if "meta_probability" in df.columns else 0.0,
        "avg_bet_size": float(df["size"].mean()) if "size" in df.columns else 0.0,
    }
