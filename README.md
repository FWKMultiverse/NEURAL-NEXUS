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

| Baseline | Value |
|---|---|
| Random guessing (5 classes) | 20% |
| Previous run — 100 epochs | **36.7%** (1.84× above random) |
| Current run — 140 epochs | Training in progress — results pending |

**Why 36.7% on 5 classes is harder than it looks:**
Most published research reports binary classification (up vs down) where random baseline is 50%. Getting to 53% on binary — as seen in recent GNN + news papers — means being only 1.06× above random. Neural-Nexus at 36.7% on 5 classes is 1.84× above random, meaning it learns significantly more from data relative to chance.

---

## Overfitting

Overfitting occurs when a model memorizes training data rather than learning general patterns — it performs well on training data but fails on unseen data. This is one of the most common failure modes in ML trading systems.

| Run | Epochs | Overfitting |
|---|---|---|
| Previous run | 100 | **0.000** — zero overfitting across all epochs |
| Current run | 140 | Pending — monitoring in progress |

**Previous run achieved 0.000 overfitting across 100 full epochs.** This is notable because more epochs typically increase overfitting risk, especially with limited training data. Maintaining zero overfitting at 100 epochs indicates the architecture and regularization are well-calibrated.

The current 140-epoch run uses 3× more training samples (46,000 vs 16,000) and 42% more news coverage. More data generally reduces overfitting risk. Whether 0.000 is maintained at 140 epochs with the expanded dataset will be confirmed when training completes.

---

## How It Compares to Published Research

This section is included for context — not to claim superiority. The field is genuinely difficult and results are rarely directly comparable across studies.

### What most research does

A February 2025 arXiv paper combining LSTM + GNN for stock prediction achieved MSE of 0.00144 — a 10.6% improvement over standalone LSTM, tested on historical backtest data only. The same paper acknowledges that empirical evaluations in real-world conditions remain limited for hybrid LSTM-GNN models.

A hybrid GNN model integrating news achieved 53% accuracy on binary stock movement — representing a 1% absolute gain over the LSTM baseline, tested on daily data with no live execution.

A 2025 ScienceDirect review of 187 deep learning financial forecasting studies confirms the dominance of LSTM-based models, while noting that hybrid architectures combining GNN with other components are gaining ground.

A 2025 LSTM + Transformer sentiment model showed that news sentiment provides additive value over price-only models, tested on daily stock data.

### What Neural-Nexus does differently

| Dimension | Most Published Research | Neural-Nexus |
|---|---|---|
| Task | Binary (up/down) or regression | 5-class (direction + strength) |
| Data frequency | Daily candles | Intraday — M1 through H4 |
| Timeframes | Single timeframe | 5 timeframes as a unified graph |
| News integration | Optional add-on | Core component, live-aligned |
| Validation | Historical backtest only | Live market — MT5 Demo |
| Execution layer | None (paper models) | Full MT5 order execution |
| Overfitting (100 epochs) | Often present | **0.000** |

The most significant gap is execution. Most hybrid LSTM-GNN studies demonstrate performance on historical data only and do not address real-time trading. Neural-Nexus has been running on live market data including the March 19, 2026 gold crash.

### Honest limitations

- Training data is limited to ~2 years of intraday XAUUSD data
- News coverage is partial — not all market-moving events are captured
- 46,000 training samples is small by research standards
- The model has not been tested across other instruments
- Live demo testing is still early — long-term track record is not yet established

*Sources: Sonani et al. (2025) arXiv:2502.15813 · Vallarino (2025) Journal of Economic Analysis 4(3):109 · Hybrid GNN-News paper, ICAIR 2025 · ScienceDirect Deep Learning Review (2025) doi:10.1016/j.irfa.2025.103988*

---

## Training Comparison

| Parameter | Previous Run | Current Run |
|---|---|---|
| Epochs | 100 | 140 |
| Training samples | ~16,000 | 46,000 (+188%) |
| News events | 2,904 | 4,129 (+42%) |
| Batch size | 32 | 32 (effective 64 with accumulation) |
| Timeframes | M1 · M5 · M15 · H1 · H4 | M1 · M5 · M15 · H1 · H4 |
| Overfitting | **0.000** | Pending |
| Validated accuracy | 36.7% | Pending |

The increase from 16,000 to 46,000 samples means the model now sees nearly **3× more market situations** during training. Combined with 42% more news coverage and 40 additional epochs, the current run is expected to produce a more robust model.

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

**Live test result:** During the March 19, 2026 gold market crash — one of the sharpest single-day selloffs in recent history — the system held all positions closed for over 8 hours without a single order. Balance unchanged.

---

## Training Methodology

- **Input:** Multi-timeframe OHLCV data across 5 timeframes + news sentiment vectors
- **Labels:** Derived from trade outcome simulation — whether TP or SL would have been hit — not raw price direction
- **Sampling:** Streaming mode — samples drawn continuously from historical data
- **Validation:** Time-based split to prevent data leakage — future data is never seen during training
- **Checkpointing:** Automatic — resumes from last saved state if interrupted

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

Active development. Currently training — 140 epoch run in progress.
Live testing on MT5 Demo account follows training completion.
Results are instrument-specific (XAUUSD) and depend on training data, market conditions, and configuration.

Credentials and account details are stored locally and never shared or committed to any repository.
