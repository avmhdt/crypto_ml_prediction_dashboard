from pathlib import Path
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "market_data.duckdb"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
TICK_DATA_DIR = Path(r"D:\Position.One\tick data")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

BAR_TYPES = [
    "time", "tick", "volume", "dollar",
    "tick_imbalance", "volume_imbalance", "dollar_imbalance",
    "tick_run", "volume_run", "dollar_run",
]

LABELING_METHODS = ["triple_barrier", "trend_scanning", "directional_change"]


class BarConfig(BaseModel):
    time_interval: str = "1min"
    tick_count: int = 1000
    volume_threshold: float = 100.0
    dollar_threshold: float = 1_000_000.0
    ewma_span: int = 100


class TripleBarrierConfig(BaseModel):
    sl_multiplier: float = 2.0
    pt_multiplier: float = 2.0
    max_holding_period: int = 50
    volatility_window: int = 20


class TrainingConfig(BaseModel):
    n_splits: int = 5
    embargo_pct: float = 0.01
    optuna_n_trials: int = 100
    optuna_timeout: int = 600
    time_decay_half_life: float = -1.0
    ffd_threshold: float = 1e-5
    ffd_max_d: float = 1.0
    ffd_adf_pvalue: float = 0.05
    feature_window: int = 20
