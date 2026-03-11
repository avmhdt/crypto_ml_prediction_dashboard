# Implementation Plan: Crypto ML Prediction Dashboard
**Spec Version:** 2.0
**Created:** 2026-03-11T18:10:00-05:00
**Status:** Draft

## Implementation Strategy

Build bottom-up: foundation → core engines (parallel) → ML pipeline → API → frontend → integration. The bar engine, labeling engine, sample weights, and feature engineering modules are independent and can be built in parallel by separate agents. The ML pipeline depends on all four. The frontend is independent of backend ML and can be developed in parallel with the API layer.

**Agent delegation:** Use Engineer agents in worktree isolation for parallel Phase B-E work. Each agent gets one module. Phase F (ML) is sequential due to inter-component dependencies. Phase H (Frontend) uses a dedicated frontend-specialist agent.

## Task Breakdown

### Phase A: Foundation (depends on: nothing)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| A1 | Initialize git repo and publish to GitHub | `.gitignore`, repo setup | — | S | Main | No |
| A2 | Create FastAPI application entry point with lifespan | `backend/main.py` | — | S | Engineer | Yes |
| A3 | Create Binance perpetual futures WebSocket live feed | `backend/data/live_feed.py` | — | M | Engineer | Yes |
| A4 | Create shared bar base class and utilities | `backend/bars/__init__.py`, `backend/bars/base.py` | — | S | Engineer | Yes |
| A5 | Update config.py with DC labeling config | `backend/config.py` | — | S | Engineer | Yes |
| A6 | Add __init__.py files for all backend packages | `backend/bars/__init__.py`, `backend/labeling/__init__.py`, `backend/weights/__init__.py`, `backend/features/__init__.py`, `backend/ml/__init__.py`, `backend/api/__init__.py` | — | S | Engineer | Yes |

### Phase B: Bar Engine (depends on: A4, A6)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| B1 | Implement time bars (fixed interval OHLCV) | `backend/bars/time_bars.py` | A4 | S | Engineer-1 | Yes (with B2-B4) |
| B2 | Implement information bars (tick, volume, dollar) | `backend/bars/information_bars.py` | A4 | M | Engineer-2 | Yes (with B1,B3,B4) |
| B3 | Implement imbalance bars (tick/volume/dollar imbalance with EWMA) | `backend/bars/imbalance_bars.py` | A4 | L | Engineer-3 | Yes (with B1,B2,B4) |
| B4 | Implement run bars (tick/volume/dollar run with EWMA) | `backend/bars/run_bars.py` | A4 | L | Engineer-4 | Yes (with B1-B3) |
| B5 | Write bar engine unit tests | `tests/test_bars.py` | B1-B4 | M | Engineer | No |

### Phase C: Labeling Engine (depends on: A6)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| C1 | Implement triple barrier labeling (binary, volatility-scaled SL/PT, time exit = sign(return)) | `backend/labeling/triple_barrier.py` | A6 | M | Engineer-5 | Yes (with C2,C3) |
| C2 | Implement trend scanning labeling (rolling OLS, max t-value, binary) | `backend/labeling/trend_scanning.py` | A6 | M | Engineer-6 | Yes (with C1,C3) |
| C3 | Implement directional change (DC) labeling (intrinsic time, multi-scale theta from vol) | `backend/labeling/directional_change.py` | A6 | L | Engineer-7 | Yes (with C1,C2) |
| C4 | Write labeling engine unit tests | `tests/test_labeling.py` | C1-C3 | M | Engineer | No |

### Phase D: Sample Weights (depends on: A6)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| D1 | Implement average uniqueness weights (AFML Ch.4) | `backend/weights/sample_weights.py` | A6 | M | Engineer-8 | Yes (with B,C,E) |
| D2 | Implement return attribution weights (AFML Ch.4) | `backend/weights/sample_weights.py` | D1 | S | Engineer-8 | No |
| D3 | Implement time decay weights with configurable half-life | `backend/weights/sample_weights.py` | D2 | S | Engineer-8 | No |
| D4 | Implement combined weight: uniqueness × attribution × decay, normalized | `backend/weights/sample_weights.py` | D1-D3 | S | Engineer-8 | No |
| D5 | Write sample weights unit tests | `tests/test_weights.py` | D4 | S | Engineer | No |

### Phase E: Feature Engineering (depends on: A6)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| E1 | Implement FFD (fractional differentiation with ADF test for min d) | `backend/features/price_features.py` | A6 | L | Engineer-9 | Yes (with E2-E5) |
| E2 | Implement structural break features (CUSUM, Chow, SADF, GSADF with windowing) | `backend/features/price_features.py` | E1 | XL | Engineer-9 | No (same file as E1) |
| E3 | Implement entropy features (Shannon, plug-in, Lempel-Ziv, Kontoyiannis) | `backend/features/price_features.py` | E1 | L | Engineer-9 | No (same file as E1) |
| E4 | Implement microstructural features (order book imbalance, trade flow imbalance, Amihud, Roll, Corwin-Schultz) | `backend/features/microstructural_features.py` | A6 | L | Engineer-10 | Yes (with E1) |
| E5 | Implement volume features (dollar volume, duration, velocity, acceleration, moments) | `backend/features/volume_features.py` | A6 | M | Engineer-11 | Yes (with E1,E4) |
| E6 | Implement volatility features (Rogers-Satchell, Garman-Klass, Yang-Zhang adapted, realized vol, bipower variation) | `backend/features/volatility_features.py` | A6 | M | Engineer-12 | Yes (with E1,E4,E5) |
| E7 | Implement time features (cyclical encoding: hour, day of week, day of month, month) | `backend/features/time_features.py` | A6 | S | Engineer-12 | Yes |
| E8 | Create feature pipeline orchestrator (combines all feature modules) | `backend/features/__init__.py` | E1-E7 | M | Engineer | No |
| E9 | Write feature engineering unit tests | `tests/test_features.py` | E8 | L | Engineer | No |

### Phase F: ML Pipeline (depends on: B, C, D, E)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| F1 | Implement Purged K-Fold CV with embargo (AFML Ch.7) | `backend/ml/purged_cv.py` | A6 | L | Engineer | No |
| F2 | Implement primary LightGBM model (side prediction, Recall-optimized) | `backend/ml/primary_model.py` | F1 | M | Engineer | No |
| F3 | Implement meta-labeling secondary model (hit/miss, Precision-optimized) | `backend/ml/meta_labeling.py` | F2 | M | Engineer | No |
| F4 | Implement bet sizing (averaging across average bets + size discretization) | `backend/ml/bet_sizing.py` | F3 | M | Engineer | No |
| F5 | Implement training orchestrator (CSV → bars → labels → weights → features → train → meta → save) | `backend/ml/training.py` | F1-F4, B, C, D, E | XL | Engineer | No |
| F6 | Implement Optuna hyperparameter tuning with purged CV objective | `backend/ml/training.py` | F5 | L | Engineer | No |
| F7 | Create training CLI script | `scripts/train.py` | F5-F6 | M | Engineer | No |
| F8 | Write ML pipeline unit tests | `tests/test_ml.py` | F1-F4 | L | Engineer | No |

### Phase G: API Layer (depends on: A2, A3, B, F)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| G1 | Implement REST API routes (symbols, bars, signals, config, metrics) | `backend/api/routes.py` | A2 | M | Engineer | Yes (with G2) |
| G2 | Implement WebSocket streaming endpoint (tick → bar → signal pipeline) | `backend/api/websocket.py` | A2, A3 | L | Engineer | Yes (with G1) |
| G3 | Wire API routes and WebSocket into FastAPI app | `backend/main.py` | G1, G2 | S | Engineer | No |
| G4 | Write API unit and integration tests | `tests/test_api.py` | G1-G3 | M | Engineer | No |

### Phase H: Frontend (depends on: nothing for setup; G for integration)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| H1 | Initialize Next.js 15 project with TypeScript, TailwindCSS, shadcn/ui | `frontend/` scaffolding | — | M | Frontend-Engineer | Yes (with A-G) |
| H2 | Configure Next.js rewrites to proxy /api and /ws to FastAPI | `frontend/next.config.ts` | H1 | S | Frontend-Engineer | No |
| H3 | Create root layout with metadata and global styles | `frontend/src/app/layout.tsx` | H1 | S | Frontend-Engineer | No |
| H4 | Create TypeScript type definitions for API contracts | `frontend/src/lib/types.ts` | H1 | S | Frontend-Engineer | No |
| H5 | Implement WebSocket connection hook with auto-reconnect | `frontend/src/hooks/useWebSocket.ts` | H4 | M | Frontend-Engineer | No |
| H6 | Implement chart component with lightweight-charts (candlesticks + signal markers + barrier lines) | `frontend/src/components/Chart.tsx` | H4, H5 | XL | Frontend-Engineer | No |
| H7 | Implement controls panel (symbol, bar type, labeling dropdowns) | `frontend/src/components/Controls.tsx` | H4 | M | Frontend-Engineer | Yes (with H6) |
| H8 | Implement signals table with sortable columns | `frontend/src/components/SignalsTable.tsx` | H4 | M | Frontend-Engineer | Yes (with H6) |
| H9 | Implement metrics panel (accuracy, precision, recall, F1, bet size dist) | `frontend/src/components/MetricsPanel.tsx` | H4 | M | Frontend-Engineer | Yes (with H6) |
| H10 | Implement header with connection status indicator | `frontend/src/components/Header.tsx` | H5 | S | Frontend-Engineer | Yes |
| H11 | Compose dashboard page with all components | `frontend/src/app/page.tsx` or `frontend/src/app/dashboard/page.tsx` | H5-H10 | M | Frontend-Engineer | No |

### Phase I: Integration & Polish (depends on: G, H)

| # | Task | Files | Depends On | Size | Agent | Parallel? |
|---|------|-------|------------|------|-------|-----------|
| I1 | End-to-end live pipeline: Binance WS → ticks → bars → features → model inference → signal → WS to frontend | `backend/main.py`, `backend/api/websocket.py` | G, F | L | Engineer | No |
| I2 | Integration test: start backend, connect frontend, verify data flow | `tests/test_integration.py` | I1, H11 | L | Engineer | No |
| I3 | Performance validation: bar latency < 500ms, signal < 2s, inference < 100ms | `tests/test_performance.py` | I1 | M | Engineer | No |
| I4 | Documentation update and final README | `README.md` | I1-I3 | S | Main | No |

## Task Summary

| Phase | Tasks | Parallelizable | Total Size |
|-------|-------|---------------|------------|
| A: Foundation | 6 | 5 of 6 | ~6S = S-M |
| B: Bar Engine | 5 | 4 of 5 | 2S+2M+2L = L |
| C: Labeling | 4 | 3 of 4 | 2M+L+M = L |
| D: Weights | 5 | 0 (sequential) | 3S+M+S = M |
| E: Features | 9 | 5 of 9 | S+XL+2L+2M+S+M+L = XL |
| F: ML Pipeline | 8 | 0 (sequential) | 2L+2M+XL+L+M+L = XL |
| G: API | 4 | 2 of 4 | S+M+L+M = L |
| H: Frontend | 11 | 5 of 11 | 2S+XL+4M+S+M = XL |
| I: Integration | 4 | 0 (sequential) | 2L+M+S = L |
| **Total** | **56 tasks** | **24 parallelizable** | |

## Dependency Graph (Simplified)

```
A (Foundation)
├── B (Bar Engine) ──────────┐
├── C (Labeling) ────────────┤
├── D (Weights) ─────────────┼── F (ML Pipeline) ── G (API) ── I (Integration)
├── E (Features) ────────────┘                                       │
└── A2,A3 ── G (API) ────────────────────────────────────────────────┘
                                                                      │
H (Frontend) ── H1-H11 (independent of backend until integration) ───┘
```

## Agent Delegation Strategy

| Agent Role | Tasks | Model | Rationale |
|-----------|-------|-------|-----------|
| Main (orchestrator) | A1, A5, A6, I4, plan coordination | opus | Coordinates all work, handles git/GitHub |
| Engineer-BarEngine | B1-B5 | sonnet | All 4 bar modules + tests, single cohesive domain |
| Engineer-Labeling | C1-C4 | sonnet | All 3 labeling methods + tests |
| Engineer-Weights | D1-D5 | sonnet | Single module, sequential |
| Engineer-PriceFeatures | E1-E3 | opus | FFD + GSADF + entropy — highest algorithmic complexity |
| Engineer-MicroFeatures | E4 | sonnet | Independent module |
| Engineer-VolFeatures | E5-E7, E8 | sonnet | Volume + volatility + time + orchestrator |
| Engineer-ML | F1-F8 | opus | Sequential pipeline, requires deep AFML knowledge |
| Engineer-API | G1-G4 | sonnet | Standard FastAPI patterns |
| Frontend-Engineer | H1-H11 | sonnet | Next.js + lightweight-charts specialist |
| Engineer-Integration | I1-I3 | sonnet | End-to-end validation |

**Parallelization windows:**
1. **Window 1** (after A): B, C, D, E, H1 — all in parallel (5 agents)
2. **Window 2** (after B+C+D+E): F (sequential), G1-G2 (parallel), H2-H11 (continuing)
3. **Window 3** (after F+G+H): I (sequential)

## Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| R1 | EWMA bar explosion in imbalance/run bars | Bars: millions produced or zero produced | Medium | Clamp EWMA to [0.5x, 2x] of initial expected ticks. Add max_bars_per_window safeguard. Validate in B5 tests. |
| R2 | GSADF O(n^2) on full 2019-2025 dataset | Feature computation takes hours/days | High | Use windowed computation (rolling window of N bars). Configurable window size in TrainingConfig. |
| R3 | DC labeling edge cases produce non-binary output | Labels may be NaN or 0 | Medium | Enforce binary output in DC implementation. Add assertion. Test edge cases in C4. |
| R4 | Training on TB-scale CSV from D: drive OOM | Memory exhaustion during read | Medium | Use chunked reading in csv_reader.py. Process one CSV file at a time. Use DuckDB's out-of-core processing. |
| R5 | Next.js WebSocket proxy to FastAPI fails | Frontend cannot receive live data | Low | Test proxy config early in H2. Fallback: direct WS connection to port 8000 with CORS. |
| R6 | LightGBM Optuna 100 trials with purged CV is slow | Training takes excessively long | Medium | Set timeout (600s default). Use Hyperband pruner to terminate bad trials early. Consider reducing n_trials for initial training. |
| R7 | Rogers-Satchell requires OHLC within single bar | May produce NaN for single-tick bars | Low | Add minimum tick count filter. Use realized vol as fallback for bars with < 4 ticks. |

## Spec Traceability

| Spec Section | Plan Tasks |
|-------------|------------|
| 3.1 Data Layer | A2, A3, A5 (existing: database.py, csv_reader.py) |
| 3.2 Bar Engine | B1, B2, B3, B4, B5 |
| 3.3 Labeling Engine | C1, C2, C3, C4 |
| 3.4 Sample Weights | D1, D2, D3, D4, D5 |
| 3.5 Feature Engineering | E1-E9 |
| 3.6 ML Pipeline | F1-F8 |
| 3.7 API Layer | G1-G4 |
| 3.8 Frontend | H1-H11 |
| 4.1-4.3 Data Models | A2 (schema in database.py), F5 (model artifacts) |
| 5.0 API Contracts | G1, G2, H4 |
| 6.0 Non-Functional Requirements | I3 |
| 10.0 Design Decisions | A5 (config updates) |
