# Production-Quality Limit Order Fill Simulation for Crypto Trading Systems

## Comprehensive Research Report

**Date:** 2026-03-13
**Scope:** Market microstructure models, fill simulation, slippage, and optimal execution
**Target:** Binance Perpetual Futures with ML-generated trading signals

---

## Table of Contents

1. [A. Limit Order Fill Simulation](#a-limit-order-fill-simulation)
2. [B. Market Microstructure Models](#b-market-microstructure-models)
3. [C. Binance-Specific Considerations](#c-binance-specific-considerations)
4. [D. Slippage and Market Impact Models](#d-slippage-and-market-impact-models)
5. [E. Optimal Limit Order Placement](#e-optimal-limit-order-placement)
6. [F. Implementation Architecture](#f-implementation-architecture)
7. [G. References](#g-references)

---

## A. Limit Order Fill Simulation

### A.1 How Professional Quant Firms Simulate Limit Order Fills

Professional quantitative firms use a hierarchy of simulation fidelity levels, from simple to sophisticated:

**Level 1 — Naive (Zero Intelligence):**
A limit buy order fills when the market price touches or crosses below the limit price. This is what the current `equity.py` uses. It massively overestimates fill rates because it ignores queue position — in reality, thousands of other orders at that price level may fill before yours.

**Level 2 — Queue Position with FIFO:**
Track estimated queue position and only fill when sufficient volume trades through that price level. This requires modeling how many orders are ahead of you in the queue.

**Level 3 — Probabilistic Queue Models:**
Use probability distributions to model the likelihood that traded volume at your price level actually fills your order, accounting for the fact that cancellations happen both before and after your position in the queue.

**Level 4 — Full LOB Simulation:**
Reconstruct the entire limit order book tick-by-tick and simulate your order's lifecycle. This is the gold standard but requires Level 3 (Market-By-Order) data, which Binance does not provide.

**Level 5 — Agent-Based Models:**
Simulate multiple interacting agents with different strategies. Used for research but too computationally expensive for production backtesting.

For your system (Binance L2 data + aggTrades), **Level 3 (Probabilistic Queue Models)** is the optimal target. It provides realistic simulation without requiring L3 data.

### A.2 State-of-the-Art Fill Probability Models

#### A.2.1 Exponential Decay Model (Baseline)

The simplest useful model assumes fill probability decays exponentially with distance from the best quote:

```
P(fill | delta) = 1 - exp(-lambda * delta)
```

Where:
- `delta` = distance of limit price from best opposing quote (in ticks)
- `lambda` = intensity parameter, calibrated from historical fill data

This is used in the Avellaneda-Stoikov framework (see Section B.2).

#### A.2.2 Survival Analysis / Time-to-Fill Models

Fill probability is modeled as a survival function S(t) = P(fill time > t):

```
S(t | x) = exp(-integral_0^t h(u | x) du)
```

Where h(u | x) is the hazard rate (instantaneous fill probability) conditional on features x:
- Queue position
- Spread width
- Volatility
- Volume imbalance
- Distance from mid-price

**Cox Proportional Hazards Model:**
```
h(t | x) = h_0(t) * exp(beta^T * x)
```

Where h_0(t) is the baseline hazard and beta is the feature coefficient vector.

**Key features for the hazard model:**
- `queue_depth_ahead`: estimated orders ahead in queue
- `spread`: current bid-ask spread in ticks
- `volatility`: recent realized volatility (e.g., 1-minute)
- `ofi`: order flow imbalance
- `depth_ratio`: depth at your level / total visible depth

#### A.2.3 Probabilistic Queue Position Models (hftbacktest)

The most practical approach for L2 data. The key insight: when the total quantity at a price level changes, you do not know whether the change occurred before or after your position in the queue.

**Core probability function:** Given your position `front` (orders ahead of you) and `back` (orders behind you), the probability that a quantity decrease occurred ahead of your position is:

```
P(decrease_before_me) = f(back) / (f(front) + f(back))
```

Where f() is a shaping function. Different choices of f() yield different models:

| Model | f(x) | Characteristics |
|-------|------|----------------|
| **LinearProbQueueModel** | f(x) = x | Equal probability per unit |
| **SquareProbQueueModel** | f(x) = x^2 | Biased toward back of queue |
| **LogProbQueueModel** | f(x) = log(1 + x) | Conservative estimate |
| **PowerProbQueueModel** | f(x) = x^n | Tunable via exponent n |

**Recommended default:** `LogProbQueueModel2` — the default in hftbacktest, it uses `log(1 + x)` and the alternative probability formula:

```
P(decrease_before_me) = f(back) / f(front + back)
```

This gives a conservative but realistic estimate of fills.

#### A.2.4 KANFormer (2024-2025, Cutting Edge)

A deep learning model combining Dilated Causal Convolution + Transformer + Kolmogorov-Arnold Networks (KAN) for predicting time-to-fill. Integrates:
- LOB snapshots (multi-level price/quantity)
- Agent action features (order submission events)
- Queue position features

**Not recommended for production simulation** due to complexity, but useful for understanding what features matter most.

#### A.2.5 The Market Maker's Dilemma (Albers et al., 2024)

Key empirical finding for crypto: **there is a fundamental negative correlation between fill probability and post-fill returns.** Orders that fill easily tend to have adverse returns (the "winner's curse"). Orders that would be profitable tend not to fill.

This means any fill simulation must also model **adverse selection** — the conditional distribution of returns given that a fill occurred is worse than the unconditional distribution.

### A.3 Estimating Queue Position Without L3 Data

With only Level 2 data (Market-By-Price), you cannot observe individual orders. You must estimate your queue position:

**Method 1 — Arrival Time Heuristic:**
When you would place your order, estimate your position as the current depth at that price level (you join at the back of the queue). As trades execute at your level, advance your position by the traded quantity.

```python
class QueuePositionTracker:
    def __init__(self, side: str, price: float, qty: float, initial_depth: float):
        self.side = side
        self.price = price
        self.qty = qty
        self.queue_position = initial_depth  # join at back

    def on_trade(self, trade_price: float, trade_qty: float, is_buyer_maker: bool):
        """Advance queue position when trades occur at our price level."""
        if self.side == "buy" and trade_price <= self.price and is_buyer_maker:
            self.queue_position = max(0, self.queue_position - trade_qty)
        elif self.side == "sell" and trade_price >= self.price and not is_buyer_maker:
            self.queue_position = max(0, self.queue_position - trade_qty)

    def on_depth_change(self, new_depth: float, old_depth: float):
        """Probabilistically update queue position on depth changes."""
        if new_depth < old_depth:
            decrease = old_depth - new_depth
            # Use LogProb model: P(before me) = log(1+back) / log(1+front+back)
            back = max(0, old_depth - self.queue_position)
            front = self.queue_position
            total = front + back
            if total > 0:
                p_before = math.log(1 + back) / math.log(1 + total)
                expected_advance = decrease * p_before
                self.queue_position = max(0, self.queue_position - expected_advance)

    @property
    def is_filled(self) -> bool:
        return self.queue_position <= 0
```

**Method 2 — Volume Clock:**
Track cumulative volume traded at your price level since order submission. Fill when cumulative volume exceeds `queue_position_at_submission + your_order_size`.

**Method 3 — Hybrid (Recommended):**
Combine trade-through detection (fills when price crosses your limit) with probabilistic queue modeling (fills at-the-touch with probability based on volume and queue position).

### A.4 Best Open-Source Implementations

| Library | Language | License | Key Features |
|---------|----------|---------|-------------|
| **[hftbacktest](https://github.com/nkaz001/hftbacktest)** | Rust/Python | MIT | Queue position models, L2/L3 support, Binance/Bybit examples, latency modeling |
| **[ABIDES](https://github.com/jpmorganchase/abides-jpmc-public)** | Python | MIT (JPMorgan) | Agent-based LOB simulation, configurable agents |
| **[QLib](https://github.com/microsoft/qlib)** | Python | MIT (Microsoft) | ML-focused quant platform with order execution module |
| **[Hummingbot](https://github.com/hummingbot/hummingbot)** | Python | Apache 2.0 | Avellaneda-Stoikov implementation, Binance integration |

**hftbacktest** is the most directly relevant. It:
- Has built-in Binance data feed support
- Implements 5+ queue position probability models
- Accounts for latency (feed + order)
- Provides both `NoPartialFillExchange` and `PartialFillExchange` modes
- Written in Rust with Python bindings for performance

---

## B. Market Microstructure Models

### B.1 Almgren-Chriss Model for Optimal Execution

The Almgren-Chriss (2000) model is the foundational framework for optimal trade execution. It balances market impact costs against timing risk.

#### B.1.1 Price Dynamics

The stock price evolves with two impact components:

```
S_k = S_{k-1} + sigma * sqrt(tau) * xi_k - g(v_k)
```

Where:
- `S_k` = price at time step k
- `sigma` = volatility
- `tau` = time between trades (T/N for N steps)
- `xi_k` ~ N(0,1) random price shock
- `g(v_k)` = permanent market impact of trading rate v_k = n_k / tau

The actual execution price includes temporary impact:

```
S_tilde_k = S_{k-1} - h(v_k)
```

#### B.1.2 Impact Functions

**Permanent impact (linear model):**
```
g(v) = gamma * v
```
Where gamma has units $/share^2. This permanently shifts the equilibrium price.

**Temporary impact (linear model):**
```
h(v) = epsilon * sign(v) + eta * v
```
Where:
- `epsilon` = fixed cost per trade (half the bid-ask spread)
- `eta` = linear temporary impact coefficient

#### B.1.3 Cost Function (Implementation Shortfall)

For a liquidation of X shares over N time periods:

```
E[Cost] = 0.5 * gamma * X^2 + epsilon * sum(|n_k|) + eta_tilde * sum(n_k^2 / tau)
```

Where `eta_tilde = eta - 0.5 * gamma * tau` (adjusted permanent impact).

Variance of cost:
```
Var[Cost] = sigma^2 * tau * sum(x_k^2)
```

#### B.1.4 Optimal Execution Trajectory

The trader minimizes `E[Cost] + lambda * Var[Cost]` where lambda is risk aversion.

**Optimal holdings at time t_j:**
```
x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X
```

**Optimal trade list (shares to trade at step j):**
```
n_j = (2 * sinh(kappa * tau / 2)) / sinh(kappa * T) * cosh(kappa * (T - t_{j-1/2})) * X
```

**Key derived parameter kappa:**
```
kappa = (1/tau) * arccosh(0.5 * kappa_tilde^2 * tau^2 + 1)
```

Where:
```
kappa_tilde^2 = lambda * sigma^2 / eta_tilde
```

**Interpretation:** kappa controls the "urgency" of execution. Higher risk aversion lambda or higher volatility sigma means faster execution (front-loaded). Lower impact eta_tilde means more patient execution (linear).

#### B.1.5 Efficient Frontier

The model produces a cost-variance efficient frontier:
```
E[Cost] = 0.5 * gamma * X^2 + epsilon * X + f(lambda)
```

Where f(lambda) traces out the tradeoff between expected cost and variance.

**Application to your system:** Use Almgren-Chriss to determine the **optimal execution schedule** for converting an ML signal into a limit order strategy. Given signal urgency (decay rate of alpha), set lambda accordingly.

### B.2 Avellaneda-Stoikov Market Making Model

The Avellaneda-Stoikov (2008) model provides the optimal bid/ask quote placement for a market maker under inventory risk.

#### B.2.1 Price Dynamics
```
dS(t) = sigma * dW(t)
```
(Arithmetic Brownian motion for mid-price.)

#### B.2.2 Reservation Price

The market maker's indifference price, adjusted for inventory risk:

```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

Where:
- `s` = current mid-price
- `q` = current inventory (positive = long, negative = short)
- `gamma` = risk aversion parameter
- `sigma` = volatility
- `T - t` = time remaining until session end

**Key insight:** When long (q > 0), the reservation price drops below mid (favoring sells to reduce inventory). When short (q < 0), it rises above mid (favoring buys).

#### B.2.3 Optimal Spread

```
delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
```

Where:
- `delta_a` = ask spread from reservation price
- `delta_b` = bid spread from reservation price
- `kappa` = order book liquidity parameter (higher = denser book = tighter spreads)

#### B.2.4 Optimal Bid/Ask Quotes

```
ask(t) = r(t) + r_spread / 2
bid(t) = r(t) - r_spread / 2
```

Where `r_spread = (2/gamma) * ln(1 + gamma/kappa)`.

#### B.2.5 Order Arrival Rate (Fill Intensity)

Market orders arrive at your limit prices as a Poisson process with intensity:

```
lambda_a = A * exp(-kappa * (ask - s))
lambda_b = A * exp(-kappa * (s - bid))
```

Where:
- `A` = base arrival intensity
- `kappa` = sensitivity parameter (how quickly fill probability decays with distance)

**Fill probability within time dt:**
```
P(ask fill in dt) = 1 - exp(-lambda_a * dt)
P(bid fill in dt) = 1 - exp(-lambda_b * dt)
```

#### B.2.6 Practical Parameter Estimation

```
kappa: Fit exponential decay to empirical fill rate vs. distance from BBO
A: Total market order arrival rate (from aggTrade stream)
sigma: Realized volatility from recent price data
gamma: Risk aversion — tune via backtest (typical: 0.01-0.5)
```

**Application to your system:** Use A-S to model the **fill probability** of your limit orders. Given a signal with entry_price, calculate the expected fill rate and adverse selection cost. This directly feeds into the fill simulation engine.

### B.3 Guilbaud-Pham Framework

Guilbaud-Pham (2011) extends Avellaneda-Stoikov by jointly optimizing **limit orders AND market orders**.

#### B.3.1 Key Extensions

The bid-ask spread is modeled as a **Markov chain** with states that are multiples of the tick size, subordinated by a Poisson tick-time clock:

```
Spread states: {1*tick, 2*tick, 3*tick, ...}
Transition matrix: P[s_{k+1} | s_k] (calibrated from data)
```

The agent submits:
- **Limit orders** at each tick (continuous control)
- **Market orders** at discrete times (impulse control)

#### B.3.2 Fill Rate Model

Limit orders fill via Cox processes with intensities depending on spread and limit prices:

```
lambda_bid(delta, s) = f(delta, s)  where delta = distance from BBO, s = spread state
lambda_ask(delta, s) = g(delta, s)
```

These are calibrated from empirical data using maximum likelihood.

#### B.3.3 Application

Use the Guilbaud-Pham transition matrix to model **spread dynamics** — this improves fill probability estimation by conditioning on spread state.

### B.4 Order Flow Imbalance (OFI) for Fill Probability Prediction

OFI quantifies the net buying/selling pressure from order book changes.

#### B.4.1 OFI Formula

At each book update n:
```
e_n = I(P_n^B >= P_{n-1}^B) * q_n^B
    - I(P_n^B <= P_{n-1}^B) * q_{n-1}^B
    - I(P_n^A <= P_{n-1}^A) * q_n^A
    + I(P_n^A >= P_{n-1}^A) * q_{n-1}^A
```

Where:
- `P^B, P^A` = best bid/ask prices
- `q^B, q^A` = quantities at best bid/ask
- `I()` = indicator function

**Aggregated OFI** over a time window:
```
OFI_t = sum(e_n for n in window)
```

#### B.4.2 Multi-Level OFI (MLOFI)

Extend to multiple book levels for stronger predictive power:
```
MLOFI_t = sum_{l=1}^{L} w_l * OFI_t^{(l)}
```

Where `w_l` are weights (can be equal or decaying with level depth). Empirical research shows multi-level OFI explains 65-87% of short-term price variance.

#### B.4.3 OFI-Price Relationship

```
Delta_P_t = alpha + beta * OFI_t + epsilon_t
```

Where beta scales inversely with contemporaneous depth. Linear for small imbalances, sublinear (concave) for extreme imbalances.

#### B.4.4 Application to Fill Probability

High positive OFI (buying pressure) implies:
- **Buy limit orders** have higher fill probability (more aggressive sellers trading through)
- **Sell limit orders** have lower fill probability (fewer sellers reaching your ask)

Use OFI as a **feature in the survival model** (Section A.2.2) or as a **modifier on the Avellaneda-Stoikov arrival intensity**:

```
lambda_adjusted = lambda_base * exp(beta_ofi * OFI_normalized)
```

### B.5 Kyle's Lambda and Market Impact Decomposition

#### B.5.1 Kyle's Lambda

Kyle (1985) defines the price adjustment rule:

```
Delta_P = lambda * (net_order_flow)
```

Where:
- `lambda` = Kyle's lambda (market impact coefficient, $/share)
- `net_order_flow` = signed volume (buyer-initiated minus seller-initiated)

Higher lambda = less liquid market = more price impact per unit of flow.

#### B.5.2 Estimation via Regression

```
r_t = alpha + lambda * V_t + epsilon_t
```

Where `r_t` is the return and `V_t` is signed volume. Lambda is estimated via OLS.

#### B.5.3 Hasbrouck Decomposition

Hasbrouck extends Kyle by decomposing price movements into:

**Permanent component** (information-driven):
```
m_t = m_{t-1} + theta * x_t + w_t
```

**Temporary component** (liquidity-driven):
```
s_t = m_t + c * x_t + u_t
```

Where `x_t` is trade sign, `theta` captures permanent impact, and `c` captures temporary impact.

#### B.5.4 Application

Estimate Kyle's lambda from your aggTrade data to quantify:
1. How much your order will move the price (market impact)
2. Whether the market is in a high/low liquidity regime
3. When to use limit vs. market orders

---

## C. Binance-Specific Considerations

### C.1 Fee Structure

#### C.1.1 USDS-M Perpetual Futures Fees (Current as of 2026)

| VIP Tier | 30-Day Volume (USD) | BNB Required | Maker Fee | Taker Fee |
|----------|---------------------|--------------|-----------|-----------|
| Regular (VIP0) | < $15M | 0 | 0.0200% | 0.0500% |
| VIP 1 | >= $15M | >= 25 BNB | 0.0160% | 0.0400% |
| VIP 2 | >= $50M | >= 100 BNB | 0.0140% | 0.0350% |
| VIP 3 | >= $100M | >= 250 BNB | 0.0120% | 0.0320% |
| VIP 4 | >= $250M | >= 500 BNB | 0.0100% | 0.0300% |
| VIP 5 | >= $500M | >= 1000 BNB | 0.0080% | 0.0270% |
| VIP 6 | >= $750M | >= 1750 BNB | 0.0060% | 0.0250% |
| VIP 7 | >= $1B | >= 3000 BNB | 0.0040% | 0.0220% |
| VIP 8 | >= $1.5B | >= 4500 BNB | 0.0020% | 0.0200% |
| VIP 9 | >= $2B | >= 6000 BNB | 0.0000% | 0.0170% |

**Additional:** 10% discount when paying fees with BNB.

**Critical note:** The user's original specification mentioned maker -0.025% (rebate) and taker 0.075%. Binance has adjusted fees since then. Current VIP0 is 0.0200% maker / 0.0500% taker. Some promotional periods offered negative maker fees (rebates) but these are time-limited. **Always check the live fee schedule.**

#### C.1.2 Fee Calculation Formula

```
Fee = Notional Value * Fee Rate
Notional Value = Quantity * Entry Price

# Example: Buy 1 BTC at $50,000
# Maker: 1 * 50,000 * 0.0002 = $10.00
# Taker: 1 * 50,000 * 0.0005 = $25.00
```

#### C.1.3 Maker vs. Taker Classification

- **Maker:** Your order adds liquidity (sits on the book, does not immediately match)
- **Taker:** Your order removes liquidity (immediately matches against resting orders)

For simulation: If your limit order fills by being matched against an incoming market order, **you pay maker fees**. If the price gaps through your level and you need to cross the spread, **you pay taker fees**.

### C.2 WebSocket Data Streams

#### C.2.1 Aggregate Trade Stream (aggTrade)

**Subscription:** `wss://fstream.binance.com/ws/btcusdt@aggTrade`

**Payload:**
```json
{
    "e": "aggTrade",     // Event type
    "E": 1672515782136,  // Event time (ms)
    "s": "BTCUSDT",      // Symbol
    "a": 5933014,        // Aggregate trade ID
    "p": "50000.00",     // Price (string)
    "q": "0.100",        // Quantity (string)
    "nq": "0.100",       // Quantity excl. RPI orders
    "f": 100,            // First trade ID
    "l": 105,            // Last trade ID
    "T": 1672515782000,  // Trade time (ms)
    "m": true            // Is buyer maker? (true = sell aggressor)
}
```

**Update frequency:** 100ms aggregation window.

**Key field:** `m` (is_buyer_maker). When `true`, the buyer was the maker (passive), meaning a SELL market order was the aggressor. This is essential for classifying trade direction (Lee-Ready algorithm simplified).

#### C.2.2 Diff Book Depth Stream

**Subscription:** `wss://fstream.binance.com/ws/btcusdt@depth@100ms`

Available speeds: `@depth` (250ms default), `@depth@500ms`, `@depth@100ms`

**Payload:**
```json
{
    "e": "depthUpdate",
    "E": 1672515782136,  // Event time
    "T": 1672515782000,  // Transaction time
    "s": "BTCUSDT",
    "U": 157,            // First update ID
    "u": 160,            // Last update ID
    "pu": 156,           // Previous final update ID
    "b": [               // Bids [price, qty]
        ["50000.00", "1.500"],
        ["49999.50", "2.300"]
    ],
    "a": [               // Asks [price, qty]
        ["50000.50", "0.800"],
        ["50001.00", "1.200"]
    ]
}
```

**Local order book maintenance protocol:**
1. Buffer depth stream events
2. GET REST snapshot (`/fapi/v1/depth?symbol=BTCUSDT&limit=1000`)
3. Discard buffered events where `u <= lastUpdateId` from snapshot
4. First buffered event must satisfy: `U <= lastUpdateId+1 <= u`
5. Apply subsequent events: `pu == previous_event.u`
6. Set price level to new quantity; remove level if quantity = 0

#### C.2.3 Individual Symbol Book Ticker Stream (BBO)

**Subscription:** `wss://fstream.binance.com/ws/btcusdt@bookTicker`

**Payload:**
```json
{
    "e": "bookTicker",
    "u": 400900217,      // Order book update ID
    "E": 1568014460893,  // Event time
    "T": 1568014460891,  // Transaction time
    "s": "BTCUSDT",
    "b": "50000.00",     // Best bid price
    "B": "1.500",        // Best bid qty
    "a": "50000.50",     // Best ask price
    "A": "0.800"         // Best ask qty
}
```

**Update frequency:** Real-time for individual symbol streams (every BBO change).

**Note:** The all-symbols `!bookTicker` stream was throttled to 5-second updates since Dec 2023. Use per-symbol streams for real-time BBO.

#### C.2.4 Recommended Multi-Stream Subscription

For production fill simulation, subscribe to all three streams:

```python
streams = [
    f"{symbol}@aggTrade",      # Trade flow for queue advancement
    f"{symbol}@depth@100ms",   # Order book for depth tracking
    f"{symbol}@bookTicker",    # BBO for spread monitoring
]
url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
```

### C.3 Funding Rate Impact

#### C.3.1 Funding Rate Mechanics

Perpetual futures use a funding rate mechanism to keep the futures price anchored to the spot index:

```
Funding Fee = Position Notional * Funding Rate
Position Notional = Mark Price * Position Size
```

**Timing:**
- Default: Every 8 hours (00:00, 08:00, 16:00 UTC)
- Some contracts: Every 4 hours (since Oct 2023)
- You only pay/receive if you hold a position at the funding timestamp

**Direction:**
- Funding Rate > 0: Longs pay shorts
- Funding Rate < 0: Shorts pay longs

#### C.3.2 Funding Rate Calculation

```
Funding Rate = Premium Index + clamp(Interest Rate - Premium Index, -0.05%, +0.05%)
```

Where:
- `Interest Rate` = 0.01% per interval (0.03% daily)
- `Premium Index` = (Impact Bid + Impact Ask) / 2 - Spot Index Price

#### C.3.3 Impact on Simulation

For positions held across funding intervals, the total cost model becomes:

```
Total Cost = Entry Fee + Exit Fee + sum(Funding Fees) + Slippage
```

For your system's typical hold times:
- **Triple barrier (minutes to hours):** Funding usually hits 0-1 times. Typical impact: 0-3 bps.
- **Trend scanning (hours to days):** Multiple funding events. Can add up to 5-20+ bps.

**Implementation:** Check if position overlaps a funding timestamp. If so, add:
```python
funding_cost = position_notional * funding_rate * num_funding_events
```

### C.4 Inferring Order Book State from aggTrade

When full depth data is unavailable (e.g., historical replay), you can infer approximate order book state from aggTrades:

**Trade Imbalance (Rolling Window):**
```python
def compute_trade_imbalance(trades: list, window_ms: int = 60000) -> float:
    buy_volume = sum(t['qty'] for t in trades if not t['is_buyer_maker'])
    sell_volume = sum(t['qty'] for t in trades if t['is_buyer_maker'])
    total = buy_volume + sell_volume
    return (buy_volume - sell_volume) / total if total > 0 else 0.0
```

**Implied Spread Estimation:**
```python
def estimate_spread(trades: list, window_ms: int = 1000) -> float:
    """Estimate spread from consecutive buy/sell trades."""
    buy_prices = [t['price'] for t in trades if not t['is_buyer_maker']]
    sell_prices = [t['price'] for t in trades if t['is_buyer_maker']]
    if buy_prices and sell_prices:
        return min(buy_prices) - max(sell_prices)
    return 0.0
```

**Volume Profile at Price Levels:**
Aggregate trade volumes at each price to estimate where liquidity rests.

---

## D. Slippage and Market Impact Models

### D.1 The Square-Root Law of Market Impact

The most universal empirical law in market microstructure:

```
I(Q) = Y * sigma * sqrt(Q / V)
```

Where:
- `I(Q)` = expected price impact of metaorder of size Q
- `Y` = calibration constant (~1.0, "order unity")
- `sigma` = daily volatility
- `Q` = total order size (shares or contracts)
- `V` = daily traded volume (ADV)

**Key properties:**
1. The exponent is **universally ~0.5** across equities, futures, options, and crypto (empirical error bars < 0.01)
2. The constant Y is roughly 1.0 across markets
3. Impact is **approximately independent** of execution schedule (number of child orders, execution time)
4. Impact is **concave** — doubling order size does NOT double impact; it increases by sqrt(2) ~ 1.41x

**Crypto calibration:** The square-root law has been **empirically confirmed on Bitcoin** across four decades of volume, including in the quasi-absence of HFT market-making (early Bitcoin markets).

### D.2 Talos Market Impact Model (Crypto-Specific)

The first empirically calibrated crypto market impact model, decomposing execution cost into three components:

```
Total_Impact = Spread_Cost + Physical_Impact + Time_Risk
```

#### D.2.1 Spread Cost

```
Spread_Cost = c_1 * S / 2
```

Where S is the bid-ask spread. This is the immediate cost of crossing the spread.

#### D.2.2 Physical Impact

```
Physical_Impact = c_2 * sigma * phi(pi) * pi^{phi_p(pi)}
```

Where:
- `pi = Q / V_T` = participation rate (your volume / market volume over horizon T)
- `sigma` = intraday volatility
- `phi_p(pi)` = sigmoid function adjusting the exponent based on participation rate
- For `pi` in the middle range (~1-10%): `phi_p ~ 0.5` (square-root law holds)
- For `pi < 0.5%`: Impact is approximately linear (higher than square-root would predict)
- For `pi > 20%`: Impact grows faster than square-root

#### D.2.3 Time Risk

Market risk exposure during execution:
```
Time_Risk = c_3 * sigma * sqrt(T)
```

Where T is the execution horizon.

**Key finding:** Traditional square-root models **underestimate impact by ~4 basis points** in the 0-0.5% participation rate range for crypto.

### D.3 Temporary vs. Permanent Market Impact

**Temporary impact** is the immediate price concession that reverts after execution:

```
Delta_P_temp(t) = eta * v(t)
```

Where v(t) is the instantaneous trading rate. This reverts with half-life determined by market resilience (Obizhaeva-Wang model):

```
Delta_P_temp(t) = Delta_P_0 * exp(-rho * t)
```

Where rho is the resilience rate.

**Permanent impact** is the lasting equilibrium shift:

```
Delta_P_perm = gamma * Q
```

This does NOT revert. It represents the information content of the trade.

**For your simulation:**
- Temporary impact affects your fill price (slippage on entry/exit)
- Permanent impact affects the market price AFTER your trade (relevant for computing realistic PnL)

### D.4 Empirical Slippage Model for Crypto

A practical slippage model combining the above:

```python
def estimate_slippage_bps(
    order_size_usd: float,
    adv_usd: float,
    spread_bps: float,
    volatility_daily: float,
    execution_time_minutes: float = 1.0,
) -> float:
    """Estimate expected slippage in basis points for a crypto futures order.

    Based on the square-root law with crypto-specific calibration.
    """
    # Participation rate
    minutes_in_day = 24 * 60
    volume_in_window = adv_usd * (execution_time_minutes / minutes_in_day)
    participation_rate = order_size_usd / volume_in_window if volume_in_window > 0 else 1.0

    # Square-root impact (Y ~ 1.0 for crypto)
    Y = 1.0
    impact_bps = Y * volatility_daily * 10000 * math.sqrt(order_size_usd / adv_usd)

    # Spread cost (half-spread for limit, full spread for market)
    spread_cost = spread_bps / 2  # limit order: half-spread

    # Time risk
    sigma_minute = volatility_daily / math.sqrt(minutes_in_day)
    time_risk = sigma_minute * math.sqrt(execution_time_minutes) * 10000

    # Total expected slippage
    return spread_cost + impact_bps + time_risk
```

### D.5 Order Size Relative to Book Depth

When your order is a significant fraction of visible book depth, slippage increases non-linearly:

```
Estimated_Slippage = sum_{i=1}^{N} (P_i - P_mid) * min(Q_remaining, D_i) / Q_total
```

Where:
- Walk through order book levels i = 1, 2, ...
- `P_i` = price at level i
- `D_i` = depth (quantity) at level i
- `Q_remaining` = unfilled portion of your order

**Rule of thumb for crypto perpetuals:**
- Order < 1% of best level depth: Minimal slippage (~0.5-1 bps)
- Order 1-10% of best level depth: Moderate slippage (~1-5 bps)
- Order 10-50% of best level depth: Significant slippage (~5-20 bps)
- Order > 50% of best level depth: Use limit orders or execution algo

---

## E. Optimal Limit Order Placement

### E.1 Maximizing Fill Probability While Minimizing Adverse Selection

The fundamental tension in limit order placement:

**Aggressive placement** (closer to or crossing the spread):
- Higher fill probability
- Higher adverse selection risk (fills correlate with adverse price moves)
- Higher fees (taker if crossing spread)

**Passive placement** (deeper in the book):
- Lower fill probability
- Lower adverse selection risk
- Earns maker rebate
- Longer time-to-fill

#### E.1.1 Optimal Distance from Mid-Price

Using the Avellaneda-Stoikov framework, the optimal distance `delta*` from mid-price for a limit order balances fill probability against adverse selection:

```
delta* = (1/kappa) * ln(1 + gamma/kappa) + (q * gamma * sigma^2 * (T - t))
```

The first term is the base spread component, the second is the inventory skew.

#### E.1.2 Signal-Adjusted Placement

For an ML signal with predicted direction and confidence:

```python
def compute_limit_price(
    mid_price: float,
    signal_side: int,       # -1 or 1
    signal_size: float,     # [0, 1] confidence
    spread: float,
    volatility: float,
    gamma: float = 0.1,
    kappa: float = 1.5,
    urgency: float = 0.5,   # 0=patient, 1=aggressive
) -> float:
    """Compute optimal limit order price given ML signal."""

    # Base offset from A-S model
    base_offset = (1 / kappa) * math.log(1 + gamma / kappa)

    # Urgency adjustment: high urgency = closer to mid
    # Low urgency = deeper in book (more passive)
    urgency_factor = 1.0 - 0.8 * urgency  # Scale offset

    # Confidence adjustment: high confidence = more aggressive
    confidence_factor = 1.0 - 0.5 * signal_size

    adjusted_offset = base_offset * urgency_factor * confidence_factor

    # Apply to correct side
    if signal_side == 1:  # Buy signal
        limit_price = mid_price - adjusted_offset * spread
    else:  # Sell signal
        limit_price = mid_price + adjusted_offset * spread

    return limit_price
```

#### E.1.3 Adverse Selection Quantification

The **expected adverse selection cost** of a filled limit order:

```
AS_cost = E[Delta_P | fill] - E[Delta_P]
```

Where `E[Delta_P | fill]` is the expected price change **conditional on your order filling**. Empirically in crypto markets:
- Limit buys that fill see an average subsequent price decline
- Limit sells that fill see an average subsequent price increase

This is the "winner's curse" — fills are conditionally bad news.

**Mitigation strategies:**
1. Place orders deeper when OFI indicates flow against your direction
2. Use wider spreads during high volatility
3. Cancel orders when adverse flow is detected (short-cancellation timeout)
4. Size inversely to adverse selection risk

### E.2 TWAP/VWAP Adapted for Single-Order Simulation

For your use case (single limit order from an ML signal), TWAP/VWAP concepts apply to determining **when and how aggressively** to place the order:

#### E.2.1 TWAP (Time-Weighted Average Price)

Slice the order into equal-sized child orders spaced evenly over an execution window:

```
child_size = total_size / N
child_interval = execution_window / N
```

**For single-order simulation:** If the ML signal has a known alpha decay rate, set the execution window to match the alpha half-life. Front-load execution if alpha decays quickly.

#### E.2.2 VWAP (Volume-Weighted Average Price)

Size child orders proportional to expected volume in each time interval:

```
child_size_i = total_size * (V_i / sum(V_j))
```

Where V_i is the expected volume in interval i (from historical volume profile).

**For single-order simulation:** Use the intraday volume curve to determine the best time to submit your limit order. Higher volume periods have higher fill probability and lower market impact.

#### E.2.3 Implementation Shortfall (IS) Strategy

The Almgren-Chriss optimal trajectory is essentially an IS strategy. For your ML signal:

```python
def compute_execution_schedule(
    signal_alpha: float,     # Expected alpha in bps
    alpha_halflife_min: float,  # Half-life of signal alpha
    total_size: float,
    volatility: float,
    temp_impact_coeff: float,
    num_steps: int = 10,
) -> list[float]:
    """Compute optimal execution schedule balancing alpha capture vs impact."""

    # Risk aversion derived from alpha decay rate
    # Faster decay = higher urgency = higher risk aversion
    lambda_risk = math.log(2) / alpha_halflife_min

    tau = alpha_halflife_min / num_steps
    eta_tilde = temp_impact_coeff
    kappa_tilde_sq = lambda_risk * volatility**2 / eta_tilde
    kappa = math.acosh(0.5 * kappa_tilde_sq * tau**2 + 1) / tau

    T = alpha_halflife_min
    schedule = []
    for j in range(num_steps):
        t_j = j * tau
        x_j = math.sinh(kappa * (T - t_j)) / math.sinh(kappa * T) * total_size
        schedule.append(x_j)

    # Convert holdings to trade list
    trades = [schedule[i] - schedule[i+1] for i in range(len(schedule)-1)]
    trades.append(schedule[-1])  # Final trade

    return trades
```

### E.3 Spread and Volatility for Optimal Price Offset

#### E.3.1 Dynamic Spread-Based Offset

```
optimal_offset = alpha_0 + alpha_1 * spread + alpha_2 * volatility + alpha_3 * ofi
```

Calibrate alpha coefficients from historical data by regressing fill quality (fill probability * post-fill return) against these features.

#### E.3.2 Volatility Regimes

| Regime | Volatility (Ann.) | Spread (bps) | Recommended Offset |
|--------|-------------------|--------------|-------------------|
| Low vol | < 30% | 1-2 bps | 0.5-1 tick passive |
| Medium vol | 30-80% | 2-5 bps | At best bid/ask |
| High vol | > 80% | 5-20+ bps | 1-2 ticks aggressive |
| Extreme vol | > 150% | 20+ bps | Market order or deep limit |

#### E.3.3 Practical Decision Framework

```python
def decide_order_type(
    signal_urgency: float,  # 0-1
    spread_bps: float,
    maker_fee_bps: float,   # e.g., 2.0 for VIP0
    taker_fee_bps: float,   # e.g., 5.0 for VIP0
    fill_prob_limit: float, # estimated fill probability for limit order
) -> str:
    """Decide between limit and market order based on expected cost."""

    # Expected cost of limit order (if it fills)
    limit_cost = maker_fee_bps  # maker fee only
    # Expected cost including fill uncertainty
    # If limit doesn't fill, we may need to cross at taker + spread
    limit_expected = (fill_prob_limit * limit_cost +
                      (1 - fill_prob_limit) * (taker_fee_bps + spread_bps))

    # Expected cost of market order
    market_cost = taker_fee_bps + spread_bps / 2  # taker fee + half spread

    if limit_expected < market_cost:
        return "LIMIT"
    else:
        return "MARKET"
```

---

## F. Implementation Architecture

### F.1 Proposed Fill Simulation Engine

Combining all the above research, here is the recommended architecture for upgrading the current `equity.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                   OrderFillSimulator                             │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ QueuePosition│  │ FillProb     │  │ SlippageModel      │    │
│  │ Tracker      │  │ Estimator    │  │                    │    │
│  │              │  │              │  │ • SpreadCost       │    │
│  │ • LogProb    │  │ • A-S Arrival│  │ • SqrtImpact       │    │
│  │   QueueModel │  │   Rate Model │  │ • TimeRisk         │    │
│  │ • Trade-thru │  │ • OFI Adj.   │  │ • AdverseSelect.   │    │
│  │   Detection  │  │ • Survival   │  │                    │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ FeeModel     │  │ FundingRate  │  │ MarketState        │    │
│  │              │  │ Calculator   │  │                    │    │
│  │ • Maker/Taker│  │              │  │ • BBO Tracker      │    │
│  │ • VIP Tiers  │  │ • 4h/8h     │  │ • Depth Tracker    │    │
│  │ • BNB Disc.  │  │   intervals │  │ • Vol Estimator    │    │
│  └──────────────┘  └──────────────┘  │ • OFI Calculator   │    │
│                                       └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### F.2 Key Data Flow

```
ML Signal → OrderFillSimulator.submit_order(signal)
    │
    ├── 1. Compute limit price (E.1.2 signal-adjusted placement)
    ├── 2. Estimate queue position (A.3 arrival time heuristic)
    ├── 3. On each tick:
    │       ├── Update queue position (A.3 LogProb model)
    │       ├── Update fill probability (B.2 A-S arrival rate)
    │       ├── Check trade-through (price crosses limit)
    │       └── Apply adverse selection filter (E.1.3)
    ├── 4. On fill:
    │       ├── Compute slippage (D.4 empirical model)
    │       ├── Apply fees (C.1 maker/taker classification)
    │       └── Check funding rate exposure (C.3)
    └── 5. Return SimulatedFill(price, fees, slippage, timestamp)
```

### F.3 Minimum Viable Implementation Priority

1. **Phase 1 (Must Have):** Maker/taker fee model with proper classification + spread-based slippage
2. **Phase 2 (High Value):** Queue position tracking with LogProb model + fill probability estimation
3. **Phase 3 (Production):** OFI-based fill probability adjustment + adverse selection modeling
4. **Phase 4 (Advanced):** Full Avellaneda-Stoikov fill rate model + optimal placement engine + funding rate integration

---

## G. References

### G.1 Foundational Papers

1. **Almgren, R. & Chriss, N. (2000).** "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5-39.
   - [PDF](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)

2. **Avellaneda, M. & Stoikov, S. (2008).** "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217-224.
   - [PDF](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)

3. **Guilbaud, F. & Pham, H. (2013).** "Optimal high-frequency trading with limit and market orders." *Quantitative Finance*, 13(1), 79-94.
   - [arXiv](https://arxiv.org/abs/1106.5040)

4. **Kyle, A.S. (1985).** "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315-1335.

5. **Obizhaeva, A. & Wang, J. (2013).** "Optimal trading strategy and supply/demand dynamics." *Journal of Financial Markets*, 16(1), 1-32.

### G.2 Fill Probability and Queue Position

6. **Moallemi, C.C. & Yuan, K. (2017).** "A Model for Queue Position Valuation in a Limit Order Book."
   - [PDF](https://moallemi.com/ciamac/papers/queue-value-2016.pdf)

7. **Cont, R. & de Larrard, A. (2024).** "Fill Probabilities in a Limit Order Book."
   - [arXiv](https://arxiv.org/pdf/2403.02572)

8. **Briola, A. et al. (2022).** "A deep learning approach to estimating fill probabilities in a limit order book." *Quantitative Finance*, 22(11).
   - [Columbia](https://business.columbia.edu/sites/default/files-efs/citation_file_upload/deep-lob-2021.pdf)

9. **KANFormer (2025).** "KANFormer for Predicting Fill Probabilities via Survival Analysis."
   - [HAL](https://hal.science/hal-05399393v1/document)

### G.3 Market Impact

10. **Bouchaud, J.P. et al. (2018).** "Trades, Quotes and Prices: Financial Markets Under the Microscope." Cambridge University Press.

11. **Gatheral, J., Schied, A., & Slynko, A. (2012).** "Transient linear price impact and Fredholm integral equations." *Mathematical Finance*.
    - [Imperial](https://mfe.baruch.cuny.edu/wp-content/uploads/2017/05/Chicago2016OptimalExecution.pdf)

12. **Talos Research.** "Understanding Market Impact in Crypto Trading."
    - [Talos](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs)

13. **Donier, J. & Bonart, J. (2014).** "A Million Metaorder Analysis of Market Impact on the Bitcoin."
    - [ResearchGate](https://www.researchgate.net/publication/269636386_A_Million_Metaorder_Analysis_of_Market_Impact_on_the_Bitcoin)

### G.4 Adverse Selection and Market Making

14. **Albers, J. et al. (2024).** "The Market Maker's Dilemma: Navigating the Fill Probability vs. Post-Fill Returns Trade-Off."
    - [arXiv](https://arxiv.org/html/2502.18625)

15. **Cartea, A., Jaimungal, S., & Penalva, J. (2015).** "Algorithmic and High-Frequency Trading." Cambridge University Press.

16. **Lehalle, C.A. (2025).** "Limit Order Strategic Placement with Adverse Selection Risk and the Role of Latency."
    - [arXiv](https://arxiv.org/abs/1610.00261)

### G.5 Order Flow Imbalance

17. **Cont, R., Kukanov, A., & Stoikov, S. (2014).** "The Price Impact of Order Book Events."
    - [ResearchGate](https://www.researchgate.net/publication/47860140_The_Price_Impact_of_Order_Book_Events)

18. **Markwick, D. (2022).** "Order Flow Imbalance - A High Frequency Trading Signal."
    - [Blog](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html)

### G.6 Open-Source Implementations

19. **hftbacktest** — HFT backtesting with queue position models, Binance/Bybit support.
    - [GitHub](https://github.com/nkaz001/hftbacktest)
    - [Docs - Queue Models](https://hftbacktest.readthedocs.io/en/latest/tutorials/Probability%20Queue%20Models.html)
    - [Docs - Order Fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html)

20. **Hummingbot** — Open-source market making bot with Avellaneda-Stoikov strategy.
    - [GitHub](https://github.com/hummingbot/hummingbot)
    - [A-S Guide](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/)

21. **Almgren-Chriss Python Implementation**
    - [GitHub](https://github.com/joshuapjacob/almgren-chriss-optimal-execution)

22. **Avellaneda-Stoikov Python Implementation**
    - [DeepWiki](https://deepwiki.com/fedecaccia/avellaneda-stoikov/2-avellaneda-stoikov-model)

### G.7 Binance API Documentation

23. **Binance USDS-M Futures WebSocket Streams**
    - [Aggregate Trades](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams)
    - [Diff Book Depth](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Diff-Book-Depth-Streams)
    - [Book Ticker](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/All-Book-Tickers-Stream)
    - [Fee Structure](https://www.binance.com/en/fee/futureFee)
    - [Funding Rates](https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031)

---

## Appendix: Quick Reference Formulas

### Fill Probability (Avellaneda-Stoikov)
```
lambda(delta) = A * exp(-kappa * delta)
P(fill in dt) = 1 - exp(-lambda * dt)
```

### Reservation Price
```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

### Optimal Spread
```
spread = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
```

### Square-Root Market Impact
```
I(Q) = Y * sigma * sqrt(Q / V)     [Y ~ 1.0]
```

### Queue Position Probability (LogProb)
```
P(decrease_before_me) = log(1 + back) / log(1 + front + back)
```

### Order Flow Imbalance
```
OFI = sum_n [ I(Bn>=Bn-1)*qBn - I(Bn<=Bn-1)*qBn-1 - I(An<=An-1)*qAn + I(An>=An-1)*qAn-1 ]
```

### Almgren-Chriss Optimal Trajectory
```
x_j = sinh(kappa*(T-t_j)) / sinh(kappa*T) * X
kappa = arccosh(0.5*kappa_tilde^2*tau^2 + 1) / tau
kappa_tilde^2 = lambda*sigma^2 / eta_tilde
```

### Fee Calculation
```
Fee = Position_Notional * Fee_Rate
Maker (VIP0): 0.0200%
Taker (VIP0): 0.0500%
Funding: Position_Notional * Funding_Rate (every 4h or 8h)
```
