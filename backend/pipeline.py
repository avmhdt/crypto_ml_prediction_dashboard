"""Live pipeline: ticks → bars → features → inference → signals.

Manages bar generators for each (symbol, bar_type) pair.
When a bar completes, runs model inference (if models are loaded)
to produce trading signals.
"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.bars import BAR_CLASSES
from backend.bars.base import Bar, BaseBarGenerator
from backend.config import (
    BAR_TYPES, LABELING_METHODS, MODELS_DIR, BarConfig, TripleBarrierConfig,
)
from backend.ml.primary_model import PrimaryModel
from backend.ml.meta_labeling import MetaLabelingModel
from backend.ml.bet_sizing import compute_bet_sizes

logger = logging.getLogger(__name__)


class LivePipeline:
    """Manages bar generation and model inference for a single symbol."""

    def __init__(self, symbol: str, bar_config: BarConfig | None = None):
        self.symbol = symbol
        self.bar_config = bar_config or BarConfig()

        # Bar generators: one per bar type
        self._generators: dict[str, BaseBarGenerator] = {}
        self._init_generators()

        # Recent bars buffer per bar_type for feature computation
        self._bar_buffers: dict[str, list[dict]] = {bt: [] for bt in BAR_TYPES}
        self._max_buffer = 100  # Keep last 100 bars for rolling features

        # Models: keyed by (bar_type, labeling_method)
        self._primary_models: dict[tuple[str, str], PrimaryModel] = {}
        self._meta_models: dict[tuple[str, str], MetaLabelingModel] = {}
        self._models_loaded = False

    def _init_generators(self) -> None:
        """Create bar generators for all bar types.

        Each generator class has its own constructor signature — bar_type is
        hardcoded internally via super().__init__.
        """
        cfg = self.bar_config
        # Map bar_type → kwargs matching each class constructor
        bar_params: dict[str, dict] = {
            "time": {"interval": cfg.time_interval},
            "tick": {"tick_count": cfg.tick_count},
            "volume": {"volume_threshold": cfg.volume_threshold},
            "dollar": {"dollar_threshold": cfg.dollar_threshold},
            "tick_imbalance": {"expected_num_ticks_init": cfg.tick_count},
            "volume_imbalance": {"expected_num_ticks_init": cfg.tick_count},
            "dollar_imbalance": {"expected_num_ticks_init": cfg.tick_count},
            "tick_run": {"expected_num_ticks_init": cfg.tick_count},
            "volume_run": {"expected_num_ticks_init": cfg.tick_count},
            "dollar_run": {"expected_num_ticks_init": cfg.tick_count},
        }

        for bar_type, cls in BAR_CLASSES.items():
            params = bar_params.get(bar_type, {})
            try:
                self._generators[bar_type] = cls(symbol=self.symbol, **params)
            except TypeError:
                # Fallback: construct with just symbol
                self._generators[bar_type] = cls(symbol=self.symbol)

    def load_models(self) -> None:
        """Load trained models from disk if available."""
        models_dir = MODELS_DIR
        if not models_dir.exists():
            logger.info(f"No models directory found, inference disabled")
            return

        loaded = 0
        for bar_type in BAR_TYPES:
            for labeling in LABELING_METHODS:
                key = (bar_type, labeling)
                prefix = f"{self.symbol}_{bar_type}_{labeling}"
                primary_path = models_dir / f"{prefix}_primary.joblib"
                scaler_path = models_dir / f"{prefix}_scaler.joblib"
                features_path = models_dir / f"{prefix}_features.json"
                meta_path = models_dir / f"{prefix}_secondary.joblib"

                if primary_path.exists() and scaler_path.exists() and features_path.exists():
                    try:
                        pm = PrimaryModel()
                        pm.load(primary_path, scaler_path, features_path)
                        self._primary_models[key] = pm
                        loaded += 1

                        if meta_path.exists():
                            meta_features_path = models_dir / f"{prefix}_meta_features.json"
                            mm = MetaLabelingModel()
                            mm.load(meta_path, meta_features_path)
                            # Backward compat: use primary feature names if meta has none
                            if not mm.feature_names:
                                mm.feature_names = pm.feature_names
                            self._meta_models[key] = mm
                    except Exception as e:
                        logger.error(f"Failed to load model {key}: {e}")

        self._models_loaded = loaded > 0
        logger.info(f"Loaded {loaded} models for {self.symbol}")

    def process_tick(
        self, price: float, qty: float, time_ms: int, is_buyer_maker: bool
    ) -> list[dict]:
        """Process a single tick through all bar generators.

        Returns list of events to broadcast:
        [{"type": "bar", "data": {...}}, {"type": "signal", "data": {...}}, ...]
        """
        events: list[dict] = []

        for bar_type, gen in self._generators.items():
            completed_bars = gen.process_tick(price, qty, time_ms, is_buyer_maker)

            for bar in completed_bars:
                bar_dict = bar.to_dict()

                # Add to buffer
                self._bar_buffers[bar_type].append(bar_dict)
                if len(self._bar_buffers[bar_type]) > self._max_buffer:
                    self._bar_buffers[bar_type] = self._bar_buffers[bar_type][-self._max_buffer:]

                events.append({"type": "bar", "data": bar_dict})

                # Run inference if models are loaded
                if self._models_loaded:
                    signals = self._run_inference(bar_type, bar_dict)
                    events.extend(signals)

        return events

    def process_ticks_batch(
        self, ticks: list[dict]
    ) -> list[dict]:
        """Process a batch of ticks. Returns all events."""
        events: list[dict] = []
        for tick in ticks:
            tick_events = self.process_tick(
                tick["price"], tick["qty"], tick["time"], tick["is_buyer_maker"]
            )
            events.extend(tick_events)
        return events

    def _run_inference(self, bar_type: str, bar_dict: dict) -> list[dict]:
        """Run model inference on the latest bar for each labeling method."""
        signals: list[dict] = []
        buffer = self._bar_buffers[bar_type]

        # Need enough bars for feature computation
        if len(buffer) < 25:
            return signals

        bars_df = pd.DataFrame(buffer)

        for labeling in LABELING_METHODS:
            key = (bar_type, labeling)
            primary = self._primary_models.get(key)
            if primary is None:
                continue

            try:
                features = self._compute_features_fast(bars_df)
                if features is None or features.empty:
                    continue

                # Get last row features for prediction
                last_features = features.iloc[[-1]]

                # Ensure we have all required feature columns
                missing = set(primary.feature_names) - set(last_features.columns)
                if missing:
                    continue

                # Primary prediction
                side = int(primary.predict(last_features)[0])
                primary_proba = float(primary.predict_proba(last_features)[0])

                # Meta-labeling
                meta = self._meta_models.get(key)
                if meta is not None:
                    meta_proba = float(
                        meta.predict_proba(last_features, np.array([side]))[0]
                    )
                else:
                    meta_proba = primary_proba

                # Skip if meta model says primary is more likely wrong
                if meta_proba < 0.5:
                    continue

                # Bet sizing
                bet_size = float(
                    compute_bet_sizes(np.array([meta_proba]))[0]
                )

                # Skip zero-size bets
                if bet_size <= 0:
                    continue

                entry_price = bar_dict["close"]
                volatility = bars_df["close"].pct_change().rolling(20).std().iloc[-1]
                if pd.isna(volatility) or volatility <= 0:
                    volatility = 0.01

                sl_price = entry_price * (1 - side * 2 * volatility)
                pt_price = entry_price * (1 + side * 2 * volatility)

                signal = {
                    "symbol": self.symbol,
                    "bar_type": bar_type,
                    "labeling_method": labeling,
                    "timestamp": bar_dict["timestamp"],
                    "side": side,
                    "size": bet_size,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "pt_price": pt_price,
                    "time_barrier": bar_dict["timestamp"] + 3_600_000,  # 1h default
                    "meta_probability": meta_proba,
                }
                signals.append({"type": "signal", "data": signal})

            except Exception as e:
                logger.debug(f"Inference failed for {key}: {e}")

        return signals

    def _compute_features_fast(self, bars_df: pd.DataFrame) -> pd.DataFrame | None:
        """Compute features from bar buffer. Lightweight version for live inference."""
        try:
            from backend.features import compute_all_features
            return compute_all_features(bars_df, trades=None, window=20)
        except Exception as e:
            logger.debug(f"Feature computation failed: {e}")
            return None
