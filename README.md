# Neural-Nexus — XAUUSD AI Trading System

Automated trading engine for XAUUSD (Gold) using a Spatio-Temporal Graph Neural Network architecture, connected to MetaTrader 5 for live execution.

> **Disclaimer:** This document is a public overview only. This is a research project — not financial advice. Live trading carries real financial risk. Past performance does not guarantee future results.

---

## Overview

Neural-Nexus is a machine-learning system that trades XAUUSD autonomously. Unlike conventional bots that apply fixed rules or look at a single timeframe, this system learns market structure from historical data and adapts its behavior to current conditions.

The key distinction: five timeframes are processed **simultaneously as a graph**, allowing the model to understand how different time horizons relate to each other in real time — not just what each one says in isolation.

---

## Architecture

Four components run in parallel, not sequentially:

| Component | Function |
|---|---|
| **GATv2** — Graph Attention Network | Models cross-timeframe relationships as a dynamic graph |
| **GIN** — Graph Isomorphism Network | Captures structural market patterns within the graph |
| **LSTM + Self-Attention** | Extracts temporal dependencies from price sequences |
| **NLP Encoder** | Encodes news sentiment as a quantitative signal |

The outputs of all four components are merged through a **CrossModalAttention** layer followed by a **Multi-Modal Fusion gate** with entropy regularization — ensuring all modalities contribute meaningfully rather than one dominating.

**Timeframes:** M1 · M5 · M15 · H1 · H4 — processed simultaneously

**Output classes:** `BUY_STRONG` · `BUY` · `HOLD` · `SELL` · `SELL_STRONG`

The graph structure is generated automatically at runtime from raw OHLCV data. No manual edge definitions or hand-crafted rules.

---

## Risk Management

Seven independent protection layers operate concurrently. No single layer can override another:

| Layer | Trigger |
|---|---|
| **Emergency Exit** | Account loss exceeds threshold → close all positions immediately |
| **Circuit Breaker** | Volatility spike beyond 3× average ATR → halt trading |
| **Position Protection** | Consecutive losses exceed limit → pause and wait |
| **RSI Reversal Bias** | Deeply oversold → suppress SELL; overbought → suppress BUY |
| **News Pre-Close** | High-impact news approaching → exit all open positions |
| **Trailing Stop** | Profitable position → move SL dynamically with ATR |
| **Regime Detection** | Classifies market as bull / bear / range / volatile / crisis → adjusts risk accordingly |

Core principle: **capital preservation over return maximization.** A system that controls drawdown survives long enough to improve.

---

## Signal Execution

Every signal passes through multiple filters before an order is placed:

- Model confidence exceeds minimum execution threshold
- Probability margin between top-2 predictions is sufficient (low ambiguity)
- Minimum number of internal sub-agents agree on direction
- Market spread is within acceptable range
- RSI does not contradict the predicted direction

If any single filter rejects the signal → the system outputs **HOLD** and waits for the next cycle. This behavior is intentional. Knowing when not to act is as important as knowing when to act.

---

## Training Methodology

- **Input:** Multi-timeframe OHLCV data + news sentiment vectors
- **Labels:** Derived from actual trade simulation — whether TP or SL would have been reached — not raw price movement
- **Sampling:** Streaming mode — draws samples continuously until target count is reached, then training begins immediately without waiting for a full data scan
- **Checkpointing:** Automatic — training resumes from last checkpoint if interrupted
- **Validation:** Time-based split to prevent data leakage between training and evaluation periods

---

## Monitoring

The system reports modality attribution per epoch — showing the relative contribution of each component (graph, temporal, news) to predictions. This confirms that all components are actively used rather than one pathway carrying all the weight.

Live inference logs include per-step output: signal class, confidence score, market regime, position size, and account balance.

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

The source code, model architecture details, training pipeline, graph construction methodology, and proprietary risk logic are not disclosed. All implementation details remain private.

Unauthorized reproduction, reverse engineering, or redistribution of any part of this system is not permitted.

---

## Status

Active development. Currently in live testing phase on demo account.  
Results are instrument-specific (XAUUSD) and depend on training data, market conditions, and configuration.

Credentials and account details are stored locally and never shared or committed to any repository.
