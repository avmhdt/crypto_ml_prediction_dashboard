"""REST API routes for the dashboard."""
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.config import BAR_TYPES, LABELING_METHODS, SYMBOLS, TripleBarrierConfig
from backend.data.database import load_bars, load_signals
from backend.simulation.equity import simulate_equity, SimulationResult
from backend.simulation.config import SimulationConfig, VIP_FEE_TABLE

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


@router.post("/seed-signals/{symbol}")
async def seed_signals(request: Request, symbol: str, bar_type: str = "time"):
    """Generate demo signals from existing bar data for dashboard display."""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    conn = request.app.state.db
    bars_df = load_bars(conn, symbol, bar_type, limit=500)
    if bars_df.empty:
        return {"status": "no bars", "count": 0}

    np.random.seed(42)
    count = 0
    for _, row in bars_df.iterrows():
        if np.random.random() > 0.12:
            continue
        side = int(np.random.choice([-1, 1]))
        vol = float(row["high"] - row["low"])
        if vol <= 0:
            continue
        meta_prob = round(float(np.random.uniform(0.45, 0.95)), 4)
        if meta_prob < 0.5:
            continue
        size = float(np.random.choice(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            p=[0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05]))
        sl = float(row["close"]) - side * vol * 2.0
        pt = float(row["close"]) + side * vol * 2.0
        tb = int(row["timestamp"]) + 50 * 60000

        for labeling in LABELING_METHODS:
            try:
                conn.execute(
                    """INSERT INTO signals (symbol, bar_type, labeling_method,
                       timestamp, side, size, entry_price, sl_price, pt_price,
                       time_barrier, meta_probability)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [symbol, bar_type, labeling, int(row["timestamp"]),
                     side, size, float(row["close"]), sl, pt, tb, meta_prob],
                )
                count += 1
            except Exception:
                pass

    conn.execute("CHECKPOINT")
    return {"status": "seeded", "count": count}


@router.get("/config/simulation")
async def get_simulation_config():
    """Return simulation configuration including VIP fee tiers."""
    tiers = [
        {"tier": t, "maker_bps": m, "taker_bps": k}
        for t, (m, k) in sorted(VIP_FEE_TABLE.items())
    ]
    return {
        "vip_tiers": tiers,
        "defaults": {
            "mode": "simple",
            "starting_capital": 10000,
            "fees_bps": 10,
            "vip_tier": 0,
            "urgency": 0.5,
            "order_timeout_ms": 60000,
        },
    }


def _result_to_dict(result: SimulationResult) -> dict:
    """Convert SimulationResult to JSON-serializable dict."""
    return {
        "timestamps": result.timestamps,
        "equity": result.equity,
        "drawdown": result.drawdown,
        "total_invested": result.total_invested,
        "metrics": result.metrics,
    }


@router.get("/equity/{symbol}")
async def get_equity(
    request: Request,
    symbol: str,
    bar_type: str = Query(default="time"),
    labeling: str = Query(default="triple_barrier"),
    starting_capital: float = Query(default=10000.0, ge=100),
    fees_bps: float = Query(default=10.0, ge=0),
    simulation_mode: str = Query(default="simple"),
    vip_tier: int = Query(default=0, ge=0, le=9),
    bnb_discount: bool = Query(default=False),
    urgency: float = Query(default=0.5, ge=0.0, le=1.0),
    order_timeout_ms: int = Query(default=60000, ge=1000),
):
    """Simulate equity curve from stored signals and bars.

    Supports three modes via ``simulation_mode``:
    - ``simple``: Flat bps fee (original behaviour)
    - ``realistic``: Decomposed 5-component cost model
    - ``both``: Returns both curves for side-by-side comparison
    """
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    if simulation_mode not in ("simple", "realistic", "both"):
        raise HTTPException(status_code=400, detail=f"Invalid simulation_mode: {simulation_mode}")

    conn = request.app.state.db
    signals_df = load_signals(conn, symbol, bar_type=bar_type,
                              labeling_method=labeling, limit=10000)

    empty_result = {
        "timestamps": [], "equity": [], "drawdown": [],
        "total_invested": [],
        "metrics": {
            "sharpe": 0, "max_dd": 0, "total_return": 0,
            "win_rate": 0, "num_trades": 0,
        },
    }

    if signals_df.empty:
        if simulation_mode == "both":
            return {"simple": empty_result, "realistic": empty_result}
        return empty_result

    # Load only bars overlapping with signal time range
    min_ts = int(signals_df["timestamp"].min())
    max_tb = signals_df["time_barrier"].dropna()
    end_ts = int(max_tb.max()) if not max_tb.empty else int(signals_df["timestamp"].max())
    bars_df = load_bars(conn, symbol, bar_type, start_time=min_ts,
                        end_time=end_ts, limit=10000)

    if bars_df.empty:
        if simulation_mode == "both":
            return {"simple": empty_result, "realistic": empty_result}
        return empty_result

    # Build SimulationConfig for realistic mode
    sim_config = SimulationConfig(
        mode=simulation_mode,
        starting_capital=starting_capital,
        fees_bps=fees_bps,
        vip_tier=vip_tier,
        bnb_discount=bnb_discount,
        order_timeout_ms=order_timeout_ms,
        urgency=urgency,
    )

    result = simulate_equity(
        signals_df, bars_df, labeling,
        starting_capital, fees_bps,
        simulation_mode=simulation_mode,
        sim_config=sim_config,
    )

    if simulation_mode == "both":
        return {
            "simple": _result_to_dict(result["simple"]),
            "realistic": _result_to_dict(result["realistic"]),
        }

    return _result_to_dict(result)
