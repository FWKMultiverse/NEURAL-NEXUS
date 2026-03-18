# Neural-Nexus — XAUUSD AI Trading System

Automated trading engine for XAUUSD (Gold) using a Spatio-Temporal Graph Neural Network.  
Connects directly to MetaTrader 5 for live order execution.

> **Disclaimer:** This is a research project. Live trading involves real financial risk. Always test on a Demo account first. Past performance does not guarantee future results. This is not financial advice.

---

## What This Is

Neural-Nexus is a machine-learning system that trades XAUUSD autonomously. It does not use rule-based signals or hardcoded indicators. Instead, it learns market patterns from historical price data and news, then applies that knowledge in real time.

The core idea: most trading bots look at one timeframe at a time. This system looks at five timeframes simultaneously as a graph — so it understands not just "what price is doing" but "how different time horizons relate to each other right now."

---

## Architecture

Four components process market data at the same time — not one after another:

| Component | Role |
|---|---|
| **GATv2** — Graph Attention Network | Learns relationships between timeframes as a unified graph |
| **GIN** — Graph Isomorphism Network | Captures structural patterns within the graph |
| **LSTM + Attention** | Extracts temporal patterns from price history |
| **NLP Encoder** | Encodes news sentiment as a real-time feature |

All four feed into a **CrossModalAttention** layer and a **Multi-Modal Fusion gate** — the model sees the complete picture, not isolated signals averaged together at the end.

**Timeframes processed simultaneously:** M1 · M5 · M15 · H1 · H4

**Prediction output:** `BUY_STRONG` · `BUY` · `HOLD` · `SELL` · `SELL_STRONG`

The graph is constructed automatically at runtime from raw OHLCV data — no manual edge definitions required.

---

## Risk Management

Multiple independent protection layers — no single layer can be bypassed by another:

- **Emergency Exit** — closes all positions immediately if account loss exceeds threshold
- **Circuit Breaker** — halts trading when volatility spikes beyond 3× normal ATR
- **Position Protection** — pauses after consecutive losses
- **RSI Reversal Bias** — suppresses SELL when market is deeply oversold; suppresses BUY when overbought
- **News Pre-Close** — exits positions before scheduled high-impact news
- **Trailing Stop** — adjusts stop-loss dynamically as a position moves in profit
- **Regime Detection** — recognizes bull / bear / range / volatile / crisis states and adjusts behavior accordingly

The guiding principle: **preservation over profit.** A system that avoids large losses survives long enough to learn.

---

## Signal Filtering

Before any order is sent, the signal passes through multiple filters:

- Model confidence must exceed minimum threshold
- Top-two predicted outcomes must be clearly separated (low ambiguity)
- Minimum number of internal agents must agree on the direction
- Spread must be within acceptable range
- RSI must not contradict the signal direction

If any filter fails → **HOLD**. Knowing when not to trade is as important as knowing when to trade.

---

## Requirements

- Windows (required for MetaTrader 5)
- Python 3.10+
- NVIDIA GPU with CUDA (RTX 3060 or better recommended)
- MetaTrader 5 with AutoTrading enabled
- 16GB RAM recommended

---

## Project Structure

```
Neural-Nexus/
├── neural_nexus_v3.2-ULTRA.py   # Main system
├── .env                          # Credentials — never share this file
├── Data/
│   ├── XAUUSDm/                  # OHLCV CSV files (M1, M5, M15, H1, H4)
│   └── News/                     # News JSON files
├── models/                       # Trained model checkpoints
└── logs/                         # System and trade logs
```

---

## Training

The system trains on historical OHLCV data across all five timeframes plus news sentiment. Labels are derived from actual trade outcomes — whether TP or SL would have been hit — not raw price direction.

Training uses streaming mode: samples from available data until target count is reached, then begins immediately. Checkpoints are saved automatically. If interrupted, training resumes from where it left off.

---

## Monitoring

The system logs attribution every epoch showing how much each component (graph, temporal, news) contributed to predictions. This verifies that all modalities are actually being used.

Live logs show per-step: signal, confidence, market regime, lot size, and account balance.

---

## Notes

- Under active development. Results vary depending on training data, market conditions, and configuration.
- Designed specifically for XAUUSD. Not tested on other instruments.
- Demo testing strongly recommended before any real account usage.
- Credentials in `.env` must never be shared or committed to version control.
- Source code and implementation details are proprietary. This document describes behavior only.
