# Crypto ML Prediction Dashboard

Live cryptocurrency perpetual futures prediction dashboard powered by machine learning models implementing Lopez de Prado's *Advances in Financial Machine Learning* (2018) and *Machine Learning for Asset Managers* (2020) methodology.

## Overview

This system provides real-time ML-generated trading signals for crypto perpetual futures **without executing any live trades**. It is a simulation/visualization tool that demonstrates the predictive capabilities of information-driven bars, advanced feature engineering, and meta-labeled LightGBM models.

### Supported Symbols
- BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT

### Key Features
- **10 Bar Types**: Time, tick, volume, dollar, tick/volume/dollar imbalance, tick/volume/dollar run bars
- **3 Labeling Methods**: Triple barrier, trend scanning, directional change (DC)
- **50+ Engineered Features**: FFD prices, structural breaks, entropy, microstructural features, volatility estimators
- **Meta-Labeled LightGBM**: Primary model (Recall-optimized side prediction) + secondary model (Precision-optimized meta-labeling) with ensemble bet sizing
- **Live Dashboard**: Next.js 15 with interactive charts, signal markers, barrier visualization, and real-time signal table

## Architecture

```
┌──────────────────────────────────────────────────┐
│            Next.js 15 Dashboard                   │
│  (lightweight-charts + shadcn/ui + WebSocket)     │
└──────────────────────┬───────────────────────────┘
                       │ WebSocket + REST
┌──────────────────────┴───────────────────────────┐
│                FastAPI Backend                     │
│  ┌─────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │ Live     │ │ Bar      │ │ ML Pipeline       │  │
│  │ Feed     │ │ Engine   │ │ (LightGBM +       │  │
│  │ (Binance)│ │ (10 types│ │  Meta-labeling)   │  │
│  └─────────┘ └──────────┘ └───────────────────┘  │
└──────────┬────────────────────────┬──────────────┘
           │                        │
  ┌────────┴────────┐    ┌─────────┴──────────┐
  │    DuckDB       │    │  D:\ CSV Files      │
  │ (live data      │    │  (training data,     │
  │  rolling window)│    │   read directly)     │
  └─────────────────┘    └────────────────────┘
```

## Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- Historical tick data from data.binance.vision on `D:\Position.One\tick data`

### Backend
```bash
pip install -r requirements.txt
```

### Train Models
```bash
python scripts/train.py --symbol BTCUSDT --bar-type tick_imbalance --labeling triple_barrier --data-dir "D:\Position.One\tick data"
```

Training reads CSV files directly from D: drive — no database initialization needed.

### Start Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 to view the dashboard.

## ML Pipeline

Based on Lopez de Prado's methodology:

1. **Data** → Read tick/trade CSVs directly from D: drive (training) or live Binance WebSocket (dashboard)
2. **Bar Generation** → Information-driven bars (tick/volume/dollar imbalance & run bars) with EWMA thresholds
3. **Labeling** → Triple barrier (binary) / trend scanning / directional change (DC)
4. **Sample Weights** → Uniqueness x return attribution x time decay
5. **Feature Engineering** → 50+ features (FFD, structural breaks, entropy, microstructural, Rogers-Satchell volatility)
6. **Primary Model** → LightGBM classifier optimized for **Recall** (side prediction) with Purged K-Fold CV + Optuna
7. **Meta-Labeling** → Secondary LightGBM optimized for **Precision** (hit/miss prediction)
8. **Bet Sizing** → Averaging across average bets + size discretization {0, 0.25, 0.5, 0.75, 1.0}

## Disclaimer

This is a research and simulation tool. **No live trades are executed.** Signals are theoretical predictions for educational and research purposes only. Past performance of ML models does not guarantee future results.

## License

MIT
