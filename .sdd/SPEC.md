# Specification: Crypto ML Prediction Dashboard
**Created:** 2026-03-11T14:35:00-05:00
**Updated:** 2026-03-11T18:00:00-05:00
**Status:** Draft v2
**Version:** 2.0

## 1. Problem Statement

### What
A live cryptocurrency perpetual futures prediction dashboard that ingests real-time market data, generates ML-powered trading signals using Lopez de Prado's *Advances in Financial Machine Learning* (2018) and *Machine Learning for Asset Managers* (2020) methodology, and visualizes predictions on an interactive chart — without executing any live trades.

### Who
Quantitative researchers, crypto traders, and ML practitioners who want to evaluate information-driven bar types, meta-labeled LightGBM models, and ensemble bet sizing on live crypto markets.

### Why
Most crypto trading dashboards either show simple technical indicators or require live capital. This system bridges the gap: it implements a full academic ML pipeline (AFML) and lets users observe model behavior on live data, compare bar types and labeling methods, and assess signal quality before committing capital.

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Next.js 15 Frontend (App Router)                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐      │
│  │ Controls │  │ Candlestick  │  │ Signal   │  │ Metrics     │      │
│  │ Panel    │  │ Chart        │  │ Table    │  │ Panel       │      │
│  │ (symbol, │  │ (lightweight │  │ (live    │  │ (model      │      │
│  │  bar,    │  │  -charts)    │  │  signals)│  │  stats)     │      │
│  │  label)  │  │              │  │          │  │             │      │
│  └──────────┘  └──────────────┘  └──────────┘  └─────────────┘      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ WebSocket (live bars/signals)
                       │ REST (recent data, config)
┌──────────────────────┴───────────────────────────────────────────────┐
│                      FastAPI Backend                                   │
│                                                                        │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Data Layer   │  │ Bar Engine  │  │ ML Pipeline                 │  │
│  │              │  │             │  │                             │  │
│  │ • Live feed  │  │ • 10 bar   │  │ • Feature Engineering       │  │
│  │   (Binance   │  │   types    │  │ • LightGBM Primary (Recall) │  │
│  │    WS)       │  │ • EWMA     │  │ • Meta-Labeling (Precision) │  │
│  │ • DuckDB     │  │   thresholds│  │ • Bet Sizing (avg/discrete)│  │
│  │   (live only)│  │            │  │                             │  │
│  └──────────────┘  └─────────────┘  └─────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ API Layer: REST routes + WebSocket streaming                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────────┐
     │                 │                     │
┌────┴────┐  ┌────────┴────────┐  ┌─────────┴──────────┐
│ DuckDB  │  │ D:\ CSV Files   │  │ Model Artifacts     │
│ (live   │  │ (historical     │  │ (joblib, JSON)      │
│  data)  │  │  training data) │  │                     │
│ • ticks │  │ • aggTrades     │  │ • primary.joblib    │
│ • bars  │  │ • trades        │  │ • secondary.joblib  │
│ • sigs  │  │ (TB-scale,      │  │ • scaler.joblib     │
│         │  │  read directly) │  │ • bar_config.json   │
└─────────┘  └─────────────────┘  └────────────────────┘
```

### Data Architecture Decision
- **DuckDB** stores ONLY recent live data (ticks, bars, signals) for dashboard display. Scoped to ~24h rolling window.
- **Historical CSV files** on `D:\Position.One\tick data` are read directly during offline training — never loaded into DuckDB. This avoids duplicating TB-scale data onto the C: drive.
- **Training pipeline** uses pandas/DuckDB `read_csv_auto` to stream directly from D: drive CSVs.

## 3. Component Design

### 3.1 Data Layer

| Component | Responsibility |
|-----------|---------------|
| `data/database.py` | DuckDB connection management, schema creation for live data only |
| `data/live_feed.py` | Binance perpetual futures WebSocket stream for real-time trade data |
| `data/csv_reader.py` | Direct CSV reading from `D:\Position.One\tick data` for offline training (no DB ingestion) |

**Data formats:**

| Source | Columns |
|--------|---------|
| trades CSV | id, price, qty, quote_qty, time (ms), is_buyer_maker |
| aggTrades CSV | agg_id, price, qty, first_id, last_id, time (ms), is_buyer_maker |
| Live tick (DuckDB) | id, symbol, price, qty, quote_qty, time (ms), is_buyer_maker |

**Live data management:**
- Micro-batch inserts: buffer incoming ticks, flush to DuckDB every N ticks or M seconds
- Rolling window: prune ticks older than 24h to keep DB small
- On startup: DB may be empty — dashboard shows "awaiting data" until ticks accumulate

### 3.2 Bar Engine

| Component | Bar Types | Algorithm Source |
|-----------|-----------|-----------------|
| `bars/time_bars.py` | Time bars | Standard OHLCV aggregation at fixed intervals |
| `bars/information_bars.py` | Tick, Volume, Dollar bars | AFML Ch.2 — sample after threshold reached |
| `bars/imbalance_bars.py` | Tick/Volume/Dollar imbalance bars | AFML Ch.2 — EWMA of expected imbalance |
| `bars/run_bars.py` | Tick/Volume/Dollar run bars | AFML Ch.2 — EWMA of expected runs |

**EWMA threshold management (critical):**
- Imbalance and run bars use EWMA estimation of expected values to set bar boundaries
- Key pitfall: bar explosion / positive feedback loop when EWMA diverges
- Hyperparameters: `expected_num_ticks_init` (initial expected bar size), `num_prev_bars` (EWMA span)
- Bar config (EWMA state) is saved/loaded per symbol+bar_type for continuity across sessions

**Bar output schema:**

| Field | Type | Description |
|-------|------|-------------|
| symbol | VARCHAR | Trading pair |
| bar_type | VARCHAR | One of 10 types |
| timestamp | BIGINT | Bar close time (unix ms) |
| open | DOUBLE | Opening price |
| high | DOUBLE | Highest price |
| low | DOUBLE | Lowest price |
| close | DOUBLE | Closing price |
| volume | DOUBLE | Total volume |
| dollar_volume | DOUBLE | Total dollar volume |
| tick_count | INTEGER | Number of ticks in bar |
| duration_us | BIGINT | Duration in microseconds (non-time bars) |

### 3.3 Labeling Engine

| Component | Method | Source |
|-----------|--------|--------|
| `labeling/triple_barrier.py` | Triple Barrier | AFML Ch.3 — SL/PT from volatility-scaled thresholds |
| `labeling/trend_scanning.py` | Trend Scanning | LdP 2019 — rolling OLS t-value regression |
| `labeling/directional_change.py` | Directional Change (DC) | Intrinsic time — price reversal detection |

**Label output:** side in {-1, 1} (short, long) — ALL labels are binary, no neutral/zero class.

**Triple Barrier specifics:**
- Stop-loss (SL) and profit-target (PT) barriers set from volatility-scaled multipliers
- Time barrier (vertical): when price hits the time exit, label = sign(return at that moment)
  - If return > 0 → label = 1 (long)
  - If return < 0 → label = -1 (short)
  - If return == 0 → label = sign of last non-zero tick direction (never assign 0)

**Trend Scanning specifics:**
- Rolling OLS regression over multiple candidate horizons
- Select horizon with maximum |t-value| of the slope coefficient
- Label = sign(slope) at the selected horizon
- Always produces {-1, 1} — if t-value is 0, use sign of raw return

**Directional Change (DC) specifics:**
- Operates in intrinsic time, orthogonal to both triple barrier and trend scanning
- Detects price reversals of magnitude theta from running extrema
- Multi-scale variant: multiple theta thresholds simultaneously
- DC event → label = direction of the change (-1 for downturn, 1 for upturn)
- Overshoot (OS) extension: measures continuation after DC event

### 3.4 Sample Weights

| Component | Methods |
|-----------|---------|
| `weights/sample_weights.py` | Average uniqueness (AFML Ch.4), Return attribution (AFML Ch.4), Time decay (configurable half-life) |

**Combination:** final_weight = uniqueness * attribution * decay, then normalize to sum=1

### 3.5 Feature Engineering

| Component | Features | Source |
|-----------|----------|--------|
| `features/price_features.py` | FFD prices (fractional differentiation), velocity, acceleration, structural breaks (CUSUM, Chow, SADF, GSADF), entropy (Shannon, plug-in, Lempel-Ziv, Kontoyiannis) | AFML Ch.5, 17, 18 |
| `features/microstructural_features.py` | Order book imbalance, trade flow imbalance, Amihud lambda, Roll spread, Corwin-Schultz spread, variance/skew/kurtosis of raw features | AFML Ch.19 (adapted for crypto perps) |
| `features/volume_features.py` | Dollar volume, duration, velocity, acceleration, variance/skew/kurtosis | Derived |
| `features/volatility_features.py` | Rogers-Satchell (primary), Garman-Klass, Yang-Zhang, realized volatility, bipower variation, velocity, acceleration | Academic literature |
| `features/time_features.py` | Hour (sin/cos), day of week (sin/cos), day of month, month (sin/cos) | Cyclical encoding |

**Total features:** 50+ atomic features

**Microstructural feature decisions (crypto-specific):**
- **NOT using VPIN**: Volume-synchronized PIN is problematic for crypto perpetual futures due to absence of traditional market maker structure and 24/7 continuous trading
- **NOT using Kyle's lambda**: Assumes single informed trader model, breaks down in crypto markets with many competing informed participants and fragmented liquidity
- **Using raw features instead**: Order book imbalance, trade flow imbalance, Amihud lambda (price impact), Roll spread (effective spread), Corwin-Schultz spread (bid-ask from high-low)

**Volatility estimator decisions (24/7 market):**
- **Rogers-Satchell as primary**: Drift-independent, does not require overnight gap (ideal for 24/7 crypto markets). Uses OHLC within bar.
- **Yang-Zhang**: Included but overnight component is inapplicable — adapted to use inter-bar close-to-open only
- **Garman-Klass**: Classical OHLC estimator, simple and fast
- **Realized volatility**: Sum of squared returns at tick level
- **Bipower variation**: Robust to jumps, uses product of adjacent absolute returns

**FFD specifics:**
- Fixed-width window fractional differentiation
- Weights: w_k = -w_{k-1} * (d - k + 1) / k
- Find minimum d such that ADF test rejects unit root at configured p-value
- Preserves memory while achieving stationarity

### 3.6 ML Pipeline

| Component | Responsibility |
|-----------|---------------|
| `ml/purged_cv.py` | Purged K-Fold with embargo (AFML Ch.7) |
| `ml/primary_model.py` | LightGBM classifier — predicts side, optimized for **Recall** |
| `ml/meta_labeling.py` | Secondary LightGBM — predicts hit/miss, optimized for **Precision** |
| `ml/bet_sizing.py` | Ensemble bet sizing: averaging across average bets + size discretization |
| `ml/training.py` | Orchestrates full pipeline: CSV → bars → labels → weights → features → train → meta-label → save |

**Primary model (side prediction):**
- Objective: maximize Recall (catch as many true signals as possible)
- Input: feature matrix from bar data
- Output: side in {-1, 1}
- Rationale: the primary model should not miss real opportunities; false positives are filtered by meta-labeling

**Secondary model (meta-labeling):**
- Objective: maximize Precision (high confidence in signal quality)
- Input: primary model prediction + feature matrix
- Output: meta_label in {0, 1} (0 = primary was wrong, 1 = primary was correct)
- Meta-label construction: meta_label = 1 if primary_prediction == true_label, else 0
- Output probability used for bet sizing

**Bet sizing (AFML + ML4AM):**
- Base size from meta-label probability: size = f(P(meta=1))
- **Averaging across average bets** (ML4AM): adjust size by concurrency — when multiple overlapping bets exist, average the sizes to avoid over-concentration
- **Size discretization** (ML4AM): round continuous sizes to discrete levels {0, 0.25, 0.5, 0.75, 1.0} for practical position management

**Hyperparameter tuning:**
- Optuna with TPE sampler + Hyperband pruner
- Objective: log_loss evaluated via Purged K-Fold CV with embargo
- Tuning: n_estimators, max_depth, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree

**Purged K-Fold CV:**
- Remove training observations whose label spans overlap with test period
- Embargo buffer: configurable percentage of total samples after each test fold
- Prevents information leakage from overlapping labels

**Code provenance:**
- All ML code written from scratch — no mlfinpy dependency
- mlfinpy (MIT, maintained through 2024) used as benchmark for correctness validation
- Custom code needed: run bars, SADF/GSADF, entropy features, DC labeling, bet sizing (~550+ lines beyond what mlfinpy covers)

### 3.7 API Layer

| Component | Responsibility |
|-----------|---------------|
| `api/routes.py` | REST endpoints for bars, signals, config |
| `api/websocket.py` | WebSocket streaming of live bars and signals |

### 3.8 Frontend (Next.js 15)

| Component | Responsibility |
|-----------|---------------|
| `app/layout.tsx` | Root layout with metadata, fonts, global styles |
| `app/page.tsx` | Main dashboard page — Server Component shell |
| `app/dashboard/page.tsx` | Dashboard with client-side interactivity |
| `components/Controls.tsx` | Symbol, bar type, labeling method dropdowns (shadcn/ui Select) |
| `components/Chart.tsx` | TradingView lightweight-charts with signal markers, SL/PT/time barrier lines |
| `components/SignalsTable.tsx` | Live signal table with sortable columns (shadcn/ui Table) |
| `components/MetricsPanel.tsx` | Model stats: accuracy, precision, recall, F1, bet size distribution |
| `components/Header.tsx` | Dashboard header with connection status indicator |
| `hooks/useWebSocket.ts` | WebSocket connection hook with auto-reconnect |
| `lib/types.ts` | Shared TypeScript type definitions |

**Tech stack:**
- Next.js 15 with App Router, React 19, TypeScript
- TailwindCSS for styling
- shadcn/ui for UI components
- lightweight-charts for financial charting
- WebSocket client for real-time data

## 4. Data Models

### 4.1 Database Entities (DuckDB — live data only)

| Entity | Table | Primary Key | Indexes | Retention |
|--------|-------|-------------|---------|-----------|
| Tick | `ticks` | (id, symbol) | (symbol, time) | 24h rolling |
| Bar | `bars` | (symbol, bar_type, timestamp) | (symbol, bar_type, timestamp) | 7d rolling |
| Signal | `signals` | id (auto-increment) | (symbol, timestamp) | 30d rolling |

### 4.2 Signal Schema

| Field | Type | Description |
|-------|------|-------------|
| id | INTEGER | Auto-incrementing PK |
| symbol | VARCHAR | e.g., BTCUSDT |
| bar_type | VARCHAR | e.g., tick_imbalance |
| labeling_method | VARCHAR | e.g., triple_barrier |
| timestamp | BIGINT | Signal generation time (unix ms) |
| side | INTEGER | -1 (short) or 1 (long) — binary only |
| size | DOUBLE | Discretized position size in {0, 0.25, 0.5, 0.75, 1.0} |
| entry_price | DOUBLE | Current price at signal time |
| sl_price | DOUBLE | Stop-loss level (triple barrier) |
| pt_price | DOUBLE | Profit-target level (triple barrier) |
| time_barrier | BIGINT | Time exit timestamp (triple barrier) |
| meta_probability | DOUBLE | Meta-label predicted probability |

### 4.3 Model Artifacts

| Artifact | Path | Format |
|----------|------|--------|
| Primary model | `models/{symbol}_{bar_type}_{label}_primary.joblib` | joblib-serialized LightGBM |
| Secondary model | `models/{symbol}_{bar_type}_{label}_secondary.joblib` | joblib-serialized LightGBM |
| Feature scaler | `models/{symbol}_{bar_type}_{label}_scaler.joblib` | joblib-serialized StandardScaler |
| Bar config | `models/{symbol}_{bar_type}_bar_config.json` | JSON with EWMA state |
| Feature list | `models/{symbol}_{bar_type}_{label}_features.json` | JSON list of feature names in order |

## 5. API Contracts

### REST Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/symbols` | — | `["BTCUSDT", ...]` |
| GET | `/api/bars/{symbol}/{bar_type}` | `?limit=500&start=&end=` | `[{timestamp, o, h, l, c, v, ...}, ...]` |
| GET | `/api/signals/{symbol}` | `?bar_type=&labeling=&limit=100` | `[{timestamp, side, size, entry, sl, pt, ...}, ...]` |
| GET | `/api/config` | — | `{bar_types, labeling_methods, symbols}` |
| POST | `/api/config/barriers` | `{sl_mult, pt_mult, max_hold}` | `{status: "updated"}` |
| GET | `/api/metrics/{symbol}` | `?bar_type=&labeling=` | `{accuracy, precision, recall, f1, sharpe, ...}` |

### WebSocket

| Endpoint | Message Type | Payload |
|----------|-------------|---------|
| `ws://host:8000/ws/{symbol}` | `bar` | `{bar_type, timestamp, o, h, l, c, v, tick_count, duration}` |
| `ws://host:8000/ws/{symbol}` | `signal` | `{bar_type, labeling, side, size, entry, sl, pt, time_barrier, prob}` |
| `ws://host:8000/ws/{symbol}` | `tick` | `{price, qty, time, is_buyer_maker}` |

## 6. Non-Functional Requirements

### Performance

| Metric | Target |
|--------|--------|
| Live bar update latency | < 500ms from tick to chart |
| Signal generation latency | < 2s from bar close to signal display |
| Historical bar query | < 1s for 1000 bars |
| Dashboard initial load | < 3s |
| Model inference | < 100ms per prediction |

### Security

| Constraint | Implementation |
|-----------|---------------|
| No trading permissions | Binance public market data WebSocket only (no API key required) |
| No order placement | Zero exchange interaction beyond market data |
| Read-only API keys | If any API key is used, it must be read-only |

### Scalability
- **DuckDB (live)**: Handles rolling window of recent data efficiently — never grows beyond configured retention
- **CSV training data**: Read directly from D: drive, no duplication to C: drive
- **Live processing**: Single symbol active at a time for live bar generation
- **Model inference**: < 100ms per prediction via pre-loaded joblib models

## 7. Technology Choices

| Choice | Technology | Rationale |
|--------|-----------|-----------|
| Backend | Python 3.12 + FastAPI | ML ecosystem (LightGBM, pandas, numpy, scipy) + async API |
| Frontend | Next.js 15 + React 19 + TypeScript | App Router, Server Components, modern React patterns |
| UI Components | shadcn/ui + TailwindCSS | Accessible, composable, professional styling |
| Charting | TradingView lightweight-charts | Professional financial charts, native marker/line support |
| Database | DuckDB (live data only) | Columnar analytics, fast aggregation, embedded (no server), Windows compatible |
| ML | LightGBM 4.5 | Fast gradient boosting, handles sample weights, categorical features |
| Tuning | Optuna | TPE sampler, Hyperband pruner, parallel trials |
| Live data | python-binance | Maintained WebSocket client for Binance perpetual futures |
| Serialization | joblib | Efficient model persistence |
| Benchmark | mlfinpy (reference only) | MIT-licensed AFML implementation used for correctness validation, NOT as a dependency |

## 8. Deployment Architecture

### Local Development (Primary)
```
Terminal 1: uvicorn backend.main:app --host 0.0.0.0 --port 8000
Terminal 2: cd frontend && npm run dev (Next.js dev server on port 3000)
```

Next.js `next.config.ts` rewrites `/api` and `/ws` to the FastAPI backend on port 8000.

### Offline Training
```
python scripts/train.py --symbol BTCUSDT --bar-type tick_imbalance --labeling triple_barrier --data-dir "D:\Position.One\tick data"
```

Training reads CSV files directly from D: drive. No database initialization step needed. Models saved to `backend/models/`.

## 9. Constraints & Limitations

| Constraint | Impact |
|-----------|--------|
| No live trading | Dashboard is simulation only — signals are theoretical |
| Historical data on D: only | Training requires D: drive mounted with CSV data present |
| C: drive space limited | DuckDB on C: holds only live data (rolling window), not historical |
| Single GPU not required | LightGBM is CPU-based, no GPU acceleration needed |
| Historical data dependency | Model quality depends on data.binance.vision data availability |
| Cross-symbol generalization | Model trained on BTCUSDT may underperform on altcoins |
| 24/7 market | Volatility estimators adapted for no overnight gaps |
| Computational cost | GSADF features are O(n^2) — use windowed computation |
| No mlfinpy dependency | All code from scratch; mlfinpy used only as correctness benchmark |

## 10. Design Decisions (Resolved)

1. **Data volume for training:** Use ALL available aggTrades (2019-2025). More data = better generalization.
2. **Retraining frequency:** One-time offline training is sufficient. Models are trained once and deployed to the dashboard.
3. **Feature selection:** Use all 50+ features. No feature importance filtering — let LightGBM handle feature selection internally via split gains.
4. **Dashboard persistence:** 30-day rolling retention in DuckDB is sufficient for signal history.
5. **DC theta selection:** Derive theta values from rolling volatility (e.g., multiples of Rogers-Satchell vol estimate). Multi-scale: use 3-5 theta levels spanning 0.5x to 3x of recent volatility.
