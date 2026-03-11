# Test Report: Crypto ML Prediction Dashboard
**Test Date:** 2026-03-11T21:00:00-05:00
**Iteration:** 1

## Summary
- Total tests: 103
- Passing: 102
- Skipped: 1 (T-M13: requires D: drive CSV data)
- Failing: 0

## Test Results

### Bar Engine (tests/test_bars.py) — 19 passed

| Test ID | Description | Status |
|---------|-------------|--------|
| T-B01 | Time bar aggregates trades into 1-min OHLCV | PASS |
| T-B02 | Time bar handles empty interval | PASS |
| T-B03 | Tick bar emits after N ticks | PASS |
| T-B04 | Volume bar emits after threshold volume | PASS |
| T-B05 | Dollar bar emits after threshold dollar volume | PASS |
| T-B06 | Tick imbalance bar EWMA converges | PASS |
| T-B07 | Volume imbalance bar EWMA converges | PASS |
| T-B08 | Dollar imbalance bar EWMA converges | PASS |
| T-B09 | Tick run bar EWMA converges | PASS |
| T-B10 | Volume run bar emits correctly | PASS |
| T-B11 | Dollar run bar emits correctly | PASS |
| T-B12 | Bar output schema has all required fields (all 10 types) | PASS |
| T-B13 | OHLCV invariant: low <= open,close <= high | PASS |
| T-B14 | Bar config save/load round-trips EWMA state | PASS |

### Labeling Engine (tests/test_labeling.py) — 16 passed

| Test ID | Description | Status |
|---------|-------------|--------|
| T-L01 | Triple barrier: PT hit first → label = 1 | PASS |
| T-L02 | Triple barrier: SL hit first → label = -1 | PASS |
| T-L03 | Triple barrier: time exit positive return → label = 1 | PASS |
| T-L04 | Triple barrier: time exit negative return → label = -1 | PASS |
| T-L05 | Triple barrier: time exit zero return → label ∈ {-1, 1} | PASS |
| T-L06 | Triple barrier: SL/PT from volatility-scaled multipliers | PASS |
| T-L07 | Triple barrier output always binary {-1, 1} (1000 bars) | PASS |
| T-L08 | Trend scanning: uptrend → label = 1 | PASS |
| T-L09 | Trend scanning: downtrend → label = -1 | PASS |
| T-L10 | Trend scanning: max t-value selects best horizon | PASS |
| T-L11 | Trend scanning output always binary {-1, 1} | PASS |
| T-L12 | DC labeling: upturn → label = 1 | PASS |
| T-L13 | DC labeling: downturn → label = -1 | PASS |
| T-L14 | DC labeling: multi-scale theta at all scales | PASS |
| T-L15 | DC labeling: theta derived from volatility | PASS |
| T-L16 | DC labeling output always binary {-1, 1} | PASS |

### Sample Weights (tests/test_weights.py) — 10 passed

| Test ID | Description | Status |
|---------|-------------|--------|
| T-W01 | Non-overlapping labels → uniqueness = 1.0 | PASS |
| T-W02 | Fully overlapping labels → uniqueness < 1.0 | PASS |
| T-W03 | Return attribution weights sum to 1.0, proportional | PASS |
| T-W04 | Time decay with positive half-life decays older | PASS |
| T-W05 | Time decay with half_life ≤ 0 → all = 1.0 | PASS |
| T-W06 | Combined = uniqueness × attribution × decay | PASS |
| T-W07 | Final weights sum to 1.0 | PASS |
| T-W08 | No weight is zero or negative | PASS |

### ML Pipeline (tests/test_ml.py) — 12 passed, 1 skipped

| Test ID | Description | Status |
|---------|-------------|--------|
| T-M01 | Purged K-Fold: test indices not in purged training | PASS |
| T-M02 | Purged K-Fold: embargo buffer applied | PASS |
| T-M03 | Purged K-Fold: correct number of folds | PASS |
| T-M04 | Primary model predicts binary {-1, 1} | PASS |
| T-M05 | Primary model uses sample weights | PASS |
| T-M06 | Meta-label construction: correct=1, incorrect=0 | PASS |
| T-M07 | Meta-labeling probability in [0, 1] | PASS |
| T-M08 | Bet sizing: size increases with probability | PASS |
| T-M09 | Bet sizing: discretized to {0, 0.25, 0.5, 0.75, 1.0} | PASS |
| T-M10 | Bet sizing: averaging reduces concentration | PASS |
| T-M11 | Model save/load round-trips correctly | PASS |
| T-M12 | Optuna trial returns valid log_loss | PASS |
| T-M13 | Training orchestrator produces artifacts | SKIP |

### Feature Engineering (tests/test_features.py) — 20 passed

| Test ID | Description | Status |
|---------|-------------|--------|
| T-F01 | FFD weights correct for d=0.5 | PASS |
| T-F02 | FFD min d small for stationary series | PASS |
| T-F03 | FFD non-stationary requires d > 0 | PASS |
| T-F04 | CUSUM detects level shift | PASS |
| T-F05 | SADF detects explosive behavior | PASS |
| T-F06 | Shannon entropy max for uniform | PASS |
| T-F07 | Lempel-Ziv low for repetitive | PASS |
| T-F08 | Amihud lambda positive | PASS |
| T-F09 | Roll spread non-negative | PASS |
| T-F10 | Corwin-Schultz spread non-negative | PASS |
| T-F11 | Rogers-Satchell matches known vol | PASS |
| T-F12 | Garman-Klass non-negative | PASS |
| T-F13 | Yang-Zhang no NaN (24/7) | PASS |
| T-F14 | Realized vol = sum squared returns | PASS |
| T-F15 | Bipower variation robust to jump | PASS |
| T-F16 | Volume velocity and acceleration | PASS |
| T-F17 | Time features sin/cos cycle | PASS |
| T-F18 | Feature pipeline 50+ columns | PASS |
| T-F19 | No NaN after warm-up | PASS |
| T-F20 | Trade flow imbalance in [-1, 1] | PASS |

### API Layer (tests/test_api.py) — 8 passed

| Test ID | Description | Status |
|---------|-------------|--------|
| T-A01 | GET /api/symbols returns 5 symbols | PASS |
| T-A02 | GET /api/config returns expected keys | PASS |
| T-A03 | GET /api/bars returns array | PASS |
| T-A04 | GET /api/bars respects limit | PASS |
| T-A05 | GET /api/signals returns array | PASS |
| T-A06 | POST /api/config/barriers updates | PASS |
| T-A07 | Invalid symbol → 404 | PASS |
| T-A08 | Invalid bar_type → 400 | PASS |

### Integration (tests/test_pipeline.py) — 17 passed

| Test ID | Description | Status |
|---------|-------------|--------|
| Pipeline init | 10 generators, correct types, no models | PASS (3) |
| Pipeline processing | Ticks produce bars, batch, buffers capped | PASS (6) |
| Database integration | Insert/load bars, signals, pruning | PASS (3) |
| Bar generators | Time, tick, volume, dollar, OHLCV correctness | PASS (5) |

## Fixes Applied

| Fix # | Related To | Description | Files Changed |
|-------|-----------|-------------|---------------|
| 1 | V-1 (Critical) | Model path mismatch between training and inference | backend/pipeline.py |
| 2 | V-2 (Major) | LZ/Kontoyiannis entropy not wired into orchestrator | backend/features/__init__.py |
| 3 | V-3 (Minor) | Time features .dt accessor for pandas Series | backend/features/time_features.py |

## Remaining Issues
None. All 102 tests pass. 1 test skipped by design (requires D: drive data).
