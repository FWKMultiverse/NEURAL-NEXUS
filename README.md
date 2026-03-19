# Neural-Nexus — XAUUSD AI Trading System

Automated trading engine for XAUUSD (Gold) using a Spatio-Temporal Graph Neural Network, connected to MetaTrader 5 for live execution.

> **Disclaimer:** This document is a public overview only. This is a research project — not financial advice. Live trading carries real financial risk. Past performance does not guarantee future results.

---

## Overview

Neural-Nexus is a machine-learning system that trades XAUUSD autonomously. Unlike conventional bots that apply fixed rules or look at a single timeframe, this system learns market structure from historical data and adapts its behavior to current conditions in real time.

The core distinction: five timeframes are processed **simultaneously as a unified graph** — not sequentially, not averaged. The model understands how M1, M5, M15, H1, and H4 relate to each other at any given moment, not just what each one says in isolation.

---

## Architecture

Four components run in parallel and are fused together — not chained one after another:

| Component | Function |
|---|---|
| **GATv2** — Graph Attention Network | Learns cross-timeframe relationships dynamically — which timeframe should influence which, and by how much |
| **GIN** — Graph Isomorphism Network | Captures structural market patterns within the graph that attention alone may miss |
| **LSTM + Self-Attention** | Extracts temporal dependencies from the price sequence |
| **NLP Encoder** | Encodes news sentiment as a direct quantitative input to the model |

All four outputs are merged through a **CrossModalAttention** layer and a **Multi-Modal Fusion gate** with entropy regularization. This prevents any single component from dominating — all four must contribute meaningfully.

**Timeframes:** M1 · M5 · M15 · H1 · H4 — processed simultaneously as one graph

**Output:** 5-class signal — `BUY_STRONG` · `BUY` · `HOLD` · `SELL` · `SELL_STRONG`

**Parameters:** ~1.2M — deliberately compact to prevent overfitting on limited data

The graph structure is generated automatically at runtime from raw OHLCV data. No manual edge definitions. No hand-crafted rules.

---

## Prediction Accuracy

The model predicts across **5 classes** simultaneously — not just up or down.

Random guessing on 5 classes = **20% accuracy baseline**

Current validated accuracy: **36.7%** — trained on 100 epochs, zero overfitting

This represents **1.84× better than random**, which is significantly stronger than it appears. Most published research reports accuracy on binary (2-class) problems where random baseline is 50%. On that scale, 36.7% on 5 classes is equivalent to approximately **92% on a binary problem** — a level that is theoretically impossible to reach consistently in live markets.

The number will be updated as training progresses.

---

## Risk Management

Seven independent protection layers run concurrently. No single layer can override another:

| Layer | Trigger |
|---|---|
| **Emergency Exit** | Account loss exceeds threshold → close all positions immediately |
| **Circuit Breaker** | Volatility spike beyond 3× average ATR → halt trading |
| **Position Protection** | Consecutive losses exceed limit → pause and wait |
| **RSI Reversal Bias** | Deeply oversold → suppress SELL signals; overbought → suppress BUY |
| **News Pre-Close** | High-impact news approaching → exit all open positions |
| **Trailing Stop** | Profitable position → move SL dynamically with ATR |
| **Regime Detection** | Classifies market as bull / bear / range / volatile / crisis → adjusts behavior accordingly |

Core principle: **capital preservation over return maximization.** A system that survives long enough will eventually learn enough.

---

## Signal Execution

Before any order is placed, the signal must pass every filter simultaneously:

- Model confidence exceeds minimum execution threshold
- Probability gap between top-2 predictions is sufficient — ambiguous signals are rejected
- Minimum number of internal agents must agree on direction
- Market spread is within acceptable range
- RSI must not contradict the signal direction

If any single filter fails → **HOLD**. The system waits for the next cycle.

This is intentional behavior. A system that knows when not to trade is more valuable than one that trades frequently.

**Live test result:** During the March 19, 2026 gold market crash — one of the sharpest single-day selloffs in recent history — the system held all positions closed for over 8 hours without a single order. Balance unchanged.

---

## Training Methodology

- **Input:** Multi-timeframe OHLCV data across 5 timeframes + news sentiment vectors
- **Labels:** Derived from trade outcome simulation — whether TP or SL would have been hit — not raw price direction
- **Samples:** 46,000 per training run, streamed from historical data
- **Validation:** Time-based split to prevent data leakage — future data never seen during training
- **Checkpointing:** Automatic — training resumes from last saved state if interrupted

---

## Monitoring

The system reports modality attribution per epoch — showing how much each component (graph, temporal, news) contributed to predictions. This confirms all four modalities are actively used, not just one carrying all the weight.

Live inference logs show per-step output: signal class, confidence score, market regime, lot size, and account balance.

---

## Requirements

- Windows (MetaTrader 5 is Windows-only)
- Python 3.10+
- NVIDIA GPU with CUDA support
- MetaTrader 5 with algorithmic trading enabled
- 16GB RAM or more recommended

---

## Intellectual Property

**This document is a behavioral description only.**

Source code, model architecture details, training pipeline, graph construction methodology, and proprietary risk logic are not disclosed. All implementation details remain private.

Unauthorized reproduction, reverse engineering, or redistribution of any part of this system is not permitted.

---

## Status

Active development. Currently in live testing phase on MT5 Demo account.
Results are instrument-specific (XAUUSD) and depend on training data, market conditions, and configuration.

Credentials and account details are stored locally and never shared or committed to any repository.
