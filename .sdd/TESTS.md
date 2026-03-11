# Test Specifications: Crypto ML Prediction Dashboard
**Spec Version:** 2.0
**Created:** 2026-03-11T18:10:00-05:00

## Unit Tests [unit]

### Bar Engine Tests (`tests/test_bars.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-B01 | Time bar aggregates trades into 1-min OHLCV | 100 trades spanning 3 minutes | 3 bars with correct OHLCV | 3.2 |
| T-B02 | Time bar handles empty interval | No trades in a 1-min window | No bar emitted (skip empty) | 3.2 |
| T-B03 | Tick bar emits after N ticks | 2500 trades, threshold=1000 | 2 bars (2000 ticks consumed), 500 remaining | 3.2 |
| T-B04 | Volume bar emits after threshold volume | Trades with varying qty | Bar emitted when cumulative volume >= threshold | 3.2 |
| T-B05 | Dollar bar emits after threshold dollar volume | Trades with varying price*qty | Bar emitted when cumulative dollar_vol >= threshold | 3.2 |
| T-B06 | Tick imbalance bar EWMA converges | 10000 trades with alternating buy/sell | EWMA expected_imbalance stabilizes, bars emitted at varying intervals | 3.2 |
| T-B07 | Volume imbalance bar EWMA converges | 10000 trades | EWMA expected volume imbalance stabilizes | 3.2 |
| T-B08 | Dollar imbalance bar EWMA converges | 10000 trades | EWMA expected dollar imbalance stabilizes | 3.2 |
| T-B09 | Tick run bar EWMA converges | 10000 trades | EWMA expected tick run stabilizes | 3.2 |
| T-B10 | Volume run bar emits correctly | 10000 trades | Bars emitted when run length condition met | 3.2 |
| T-B11 | Dollar run bar emits correctly | 10000 trades | Bars emitted when dollar run condition met | 3.2 |
| T-B12 | Bar output schema has all required fields | Any bar type | {symbol, bar_type, timestamp, open, high, low, close, volume, dollar_volume, tick_count, duration_us} | 3.2 |
| T-B13 | OHLCV invariant: low <= open,close <= high | Random trades | All bars satisfy low <= min(open,close) and high >= max(open,close) | 3.2 |
| T-B14 | Bar config save/load round-trips EWMA state | Save imbalance bar state, reload | EWMA parameters match after reload | 3.2 |

### Labeling Tests (`tests/test_labeling.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-L01 | Triple barrier: price hits PT first → label = 1 | Bars with upward trend | Label = 1 for bars where close crosses PT | 3.3 |
| T-L02 | Triple barrier: price hits SL first → label = -1 | Bars with downward trend | Label = -1 for bars where close crosses SL | 3.3 |
| T-L03 | Triple barrier: time exit with positive return → label = 1 | Bars with small drift, hits time limit | Label = sign(return at time barrier) = 1 | 3.3 |
| T-L04 | Triple barrier: time exit with negative return → label = -1 | Bars with small negative drift | Label = sign(return at time barrier) = -1 | 3.3 |
| T-L05 | Triple barrier: time exit with zero return → label = last direction | Flat bars hitting time limit | Label ∈ {-1, 1}, never 0 | 3.3 |
| T-L06 | Triple barrier: SL/PT from volatility-scaled multipliers | Bars with known volatility | SL = entry - sl_mult * vol, PT = entry + pt_mult * vol | 3.3 |
| T-L07 | Triple barrier output is always binary {-1, 1} | 1000 random bar sequences | All labels ∈ {-1, 1}, no zeros or NaN | 3.3 |
| T-L08 | Trend scanning: uptrend detected → label = 1 | Linear uptrend bars | Label = 1 (positive slope) | 3.3 |
| T-L09 | Trend scanning: downtrend detected → label = -1 | Linear downtrend bars | Label = -1 (negative slope) | 3.3 |
| T-L10 | Trend scanning: max t-value selects best horizon | Multiple horizons with varying trends | Horizon with max |t-value| selected | 3.3 |
| T-L11 | Trend scanning output is always binary {-1, 1} | 1000 random bars | All labels ∈ {-1, 1} | 3.3 |
| T-L12 | DC labeling: upturn event → label = 1 | Price reversal upward by theta | DC event with label = 1 | 3.3 |
| T-L13 | DC labeling: downturn event → label = -1 | Price reversal downward by theta | DC event with label = -1 | 3.3 |
| T-L14 | DC labeling: multi-scale theta produces events at all scales | Price with multiple reversal magnitudes | Events detected at each theta level | 3.3 |
| T-L15 | DC labeling: theta derived from Rogers-Satchell volatility | Bars with known RS vol | Theta values are multiples of vol estimate | 3.3 |
| T-L16 | DC labeling output is always binary {-1, 1} | 1000 random bars | All labels ∈ {-1, 1}, no zeros or NaN | 3.3 |

### Sample Weights Tests (`tests/test_weights.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-W01 | Average uniqueness: non-overlapping labels → weight = 1.0 | Labels with no time overlap | All uniqueness weights = 1.0 | 3.4 |
| T-W02 | Average uniqueness: fully overlapping labels → weight < 1.0 | Labels with 100% overlap | All uniqueness weights < 1.0 | 3.4 |
| T-W03 | Return attribution weights sum to total return | Bar sequence with known returns | Sum of attributed returns = total return | 3.4 |
| T-W04 | Time decay with positive half-life decays older weights | Labels spanning 1000 bars | Older labels have smaller decay weight | 3.4 |
| T-W05 | Time decay with half_life = -1 means no decay | Any labels | All decay weights = 1.0 | 3.4 |
| T-W06 | Combined weight = uniqueness × attribution × decay | Known component weights | Product matches expected | 3.4 |
| T-W07 | Final weights normalized to sum = 1.0 | Any input | sum(weights) ≈ 1.0 (within float tolerance) | 3.4 |
| T-W08 | No weight is zero or negative | Any valid input | All weights > 0 | 3.4 |

### Feature Engineering Tests (`tests/test_features.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-F01 | FFD: weights computed correctly for d=0.5 | d=0.5, threshold=1e-5 | Weights match analytical formula w_k = -w_{k-1} * (d-k+1)/k | 3.5 |
| T-F02 | FFD: min d found via ADF test | Stationary series | d = 0 (already stationary) | 3.5 |
| T-F03 | FFD: non-stationary series requires d > 0 | Random walk | d > 0 returned, ADF rejects unit root | 3.5 |
| T-F04 | CUSUM structural break: detects level shift | Series with mean shift at midpoint | Break detected near midpoint | 3.5 |
| T-F05 | SADF: detects explosive behavior | Simulated bubble | SADF statistic exceeds critical value | 3.5 |
| T-F06 | Shannon entropy: maximum for uniform distribution | Uniform price returns | Entropy is maximal (log(n_bins)) | 3.5 |
| T-F07 | Lempel-Ziv complexity: low for repetitive sequence | Repeating pattern | LZ complexity is low | 3.5 |
| T-F08 | Amihud lambda: positive for price impact | Trades with consistent impact | Lambda > 0 | 3.5 |
| T-F09 | Roll spread: non-negative effective spread | Bar sequence | Roll spread >= 0 | 3.5 |
| T-F10 | Corwin-Schultz spread: non-negative bid-ask | Bar sequence with high > low | CS spread >= 0 | 3.5 |
| T-F11 | Rogers-Satchell volatility: matches known volatility | GBM simulation with sigma=0.2 | RS vol estimate ≈ 0.2 (within tolerance) | 3.5 |
| T-F12 | Garman-Klass volatility: non-negative | Any OHLCV bars | GK vol >= 0 | 3.5 |
| T-F13 | Yang-Zhang volatility adapted for 24/7: no overnight gap | 24/7 bar sequence | YZ vol computed without NaN | 3.5 |
| T-F14 | Realized volatility: sum of squared returns | Known return sequence | RV matches sum(r^2) | 3.5 |
| T-F15 | Bipower variation: robust to single jump | Series with one large outlier | BV significantly lower than RV | 3.5 |
| T-F16 | Volume velocity and acceleration computed correctly | Linear volume growth | Velocity ≈ constant, acceleration ≈ 0 | 3.5 |
| T-F17 | Time features: hour sin/cos cycle correctly | 24 bars at each hour | sin/cos complete full cycle | 3.5 |
| T-F18 | Feature pipeline produces 50+ columns | Full bar DataFrame | Output has >= 50 feature columns | 3.5 |
| T-F19 | No NaN in feature output (after warm-up period) | 500+ bars | After first 100 bars (warm-up), no NaN in features | 3.5 |
| T-F20 | Trade flow imbalance in [-1, 1] range | Any trade sequence | Imbalance ∈ [-1, 1] | 3.5 |

### ML Pipeline Tests (`tests/test_ml.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-M01 | Purged K-Fold: test indices don't overlap with purged training indices | Overlapping label spans | No training sample has label span overlapping test period | 3.6 |
| T-M02 | Purged K-Fold: embargo buffer applied after test set | 5-fold split, embargo=1% | Gap of ceil(N*0.01) samples between test and next train fold | 3.6 |
| T-M03 | Purged K-Fold: produces correct number of folds | n_splits=5 | 5 (train, test) tuples | 3.6 |
| T-M04 | Primary model predicts binary {-1, 1} | Feature matrix + labels | Predictions ∈ {-1, 1} | 3.6 |
| T-M05 | Primary model uses sample weights in training | Known weighted dataset | Model trained with sample_weight parameter | 3.6 |
| T-M06 | Meta-label construction: correct = 1, incorrect = 0 | Primary predictions + true labels | meta_label[i] = 1 if pred[i] == true[i] else 0 | 3.6 |
| T-M07 | Meta-labeling model outputs probability in [0, 1] | Feature matrix | P(meta=1) ∈ [0, 1] | 3.6 |
| T-M08 | Bet sizing: size = f(meta probability) | Probabilities from 0 to 1 | Sizes monotonically increase with probability | 3.6 |
| T-M09 | Bet sizing: discretized to {0, 0.25, 0.5, 0.75, 1.0} | Continuous probabilities | All outputs ∈ {0, 0.25, 0.5, 0.75, 1.0} | 3.6 |
| T-M10 | Bet sizing: averaging across average bets reduces concentration | Concurrent overlapping bets | Averaged size <= max individual size | 3.6 |
| T-M11 | Model save/load round-trips correctly | Train model, save, load | Loaded model predictions match original | 3.6 |
| T-M12 | Optuna trial objective returns log_loss from purged CV | Feature matrix, labels, weights | Objective value is a valid log_loss float | 3.6 |
| T-M13 | Training orchestrator produces all 4 artifacts | Complete pipeline run | primary.joblib, secondary.joblib, scaler.joblib, bar_config.json all exist | 3.6 |

### API Tests (`tests/test_api.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-A01 | GET /api/symbols returns symbol list | — | ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"] | 5.0 |
| T-A02 | GET /api/config returns bar_types, labeling_methods, symbols | — | 10 bar types, 3 labeling methods, 5 symbols | 5.0 |
| T-A03 | GET /api/bars/{symbol}/{bar_type} returns OHLCV array | symbol=BTCUSDT, bar_type=time | Array of bar objects with timestamp, o, h, l, c, v | 5.0 |
| T-A04 | GET /api/bars respects limit parameter | limit=10 | Exactly 10 bars returned (if available) | 5.0 |
| T-A05 | GET /api/signals/{symbol} returns signal array | symbol=BTCUSDT | Array of signal objects with side, size, entry, etc. | 5.0 |
| T-A06 | POST /api/config/barriers updates config | {sl_mult: 3.0, pt_mult: 3.0} | {status: "updated"} | 5.0 |
| T-A07 | GET /api/bars with invalid symbol returns 404 | symbol=INVALID | HTTP 404 | 5.0 |
| T-A08 | GET /api/bars with invalid bar_type returns 400 | bar_type=invalid | HTTP 400 | 5.0 |

### Database Tests (`tests/test_database.py`)

| Test ID | Description | Input | Expected | Spec Ref |
|---------|-------------|-------|----------|----------|
| T-D01 | init_schema creates ticks, bars, signals tables | Fresh DuckDB | All 3 tables exist | 4.1 |
| T-D02 | insert_ticks_batch inserts N ticks | 100 tick dicts | 100 rows in ticks table | 4.1 |
| T-D03 | prune_old_data removes ticks older than 24h | Ticks at now-25h and now-1h | Only now-1h tick remains | 4.1 |
| T-D04 | prune_old_data removes bars older than 7d | Bars at now-8d and now-1d | Only now-1d bar remains | 4.1 |
| T-D05 | prune_old_data removes signals older than 30d | Signals at now-31d and now-1d | Only now-1d signal remains | 4.1 |
| T-D06 | load_bars returns bars in descending timestamp order | 10 bars | First bar has latest timestamp | 4.1 |
| T-D07 | load_signals filters by bar_type and labeling_method | Mixed signals | Only matching signals returned | 4.2 |

## Integration Tests [integration]

| Test ID | Description | Components | Spec Ref |
|---------|-------------|------------|----------|
| T-I01 | CSV reader loads trades from D: drive and returns valid DataFrame | csv_reader.py + D: CSV files | 3.1 |
| T-I02 | Bar engine produces bars from CSV trade data | csv_reader.py + bar engine | 3.1, 3.2 |
| T-I03 | Full pipeline: CSV → bars → labels → weights → features → model → signals | All backend modules | 3.1-3.6 |
| T-I04 | WebSocket endpoint streams bar updates to connected client | FastAPI + WebSocket + bar engine | 5.0 |
| T-I05 | REST API returns bars computed from live data in DuckDB | FastAPI + database.py + bar engine | 5.0 |
| T-I06 | Frontend connects to WebSocket and displays chart updates | Next.js + FastAPI + WebSocket | 3.8, 5.0 |
| T-I07 | Model training script runs end-to-end and produces artifacts | scripts/train.py + all backend | 3.6, 8.0 |
| T-I08 | Signal generation: bar close → feature compute → model predict → signal stored | ML pipeline + database | 3.6, 4.2 |

## Edge Cases [unit]

| Test ID | Description | Condition | Expected | Spec Ref |
|---------|-------------|-----------|----------|----------|
| T-E01 | EWMA bar explosion: imbalance bar with diverging EWMA | EWMA expected_imbalance → 0 causing infinite bars | Bar emission clamped to max_bars_per_window, EWMA reset | 3.2 |
| T-E02 | EWMA bar starvation: run bar with inflating EWMA | EWMA expected_run → ∞ causing zero bars | EWMA clamped to [0.5x, 2x] of initial, minimum 1 bar per N ticks | 3.2 |
| T-E03 | Single-tick bar: bar with only 1 trade | Single trade in bar | open == high == low == close, volume > 0 | 3.2 |
| T-E04 | Triple barrier with zero volatility | Flat price series | Fall back to minimum barrier width (1 tick) | 3.3 |
| T-E05 | Trend scanning with constant price | All bars same close | Label defaults to sign of micro-noise or last direction, never 0 | 3.3 |
| T-E06 | DC labeling with theta larger than price range | Theta > max(price) - min(price) | No DC events detected (empty output handled gracefully) | 3.3 |
| T-E07 | FFD with d=0 (no differentiation needed) | Already stationary series | Output ≈ input (identity transform) | 3.5 |
| T-E08 | Feature computation on insufficient bars (< warm-up) | 5 bars (warm-up = 20) | NaN for features requiring window > 5, no crash | 3.5 |
| T-E09 | Model prediction with all-NaN features | NaN feature vector | Graceful error or neutral prediction, no crash | 3.6 |
| T-E10 | WebSocket reconnection: client reconnects mid-stream | Disconnect + reconnect | Client receives latest bar state, no stale data | 5.0 |
| T-E11 | Concurrent bet sizing: no overlapping bets | All bets sequential | Averaging has no effect (size = raw size) | 3.6 |
| T-E12 | CSV reader with missing files on D: drive | D: drive unmounted or path doesn't exist | Graceful error with informative message | 3.1 |
| T-E13 | DuckDB empty on first startup | No ticks, no bars, no signals | Dashboard shows "awaiting data", no crash | 4.1 |
| T-E14 | Rogers-Satchell on single-tick bar (O=H=L=C) | Bar with no price range | Vol = 0, handled without log(0) crash | 3.5 |
