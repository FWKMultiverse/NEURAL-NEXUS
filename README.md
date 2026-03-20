# Neural-Nexus — XAUUSD AI Trading System

Automated trading engine for XAUUSD (Gold) using a Spatio-Temporal Graph Neural Network, connected to MetaTrader 5 for live execution.

> **Disclaimer:** This document is a public overview only. This is a research project — not financial advice. Live trading carries real financial risk. Past performance does not guarantee future results.

---

## Overview

Neural-Nexus is a machine-learning system that trades XAUUSD autonomously. Unlike conventional bots that apply fixed rules or look at a single timeframe, this system learns market structure from historical data and adapts its behavior to current conditions in real time.

The core distinction: five timeframes are processed **simultaneously as a unified graph** — not sequentially, not averaged. The model understands how M1, M5, M15, H1, and H4 relate to each other at any given moment, not just what each one says in isolation. This allows it to detect whether a short-term move is supported or contradicted by longer-term structure — something a single-timeframe model cannot see.

---

## Architecture

Four components run in parallel and are fused together — not chained one after another:

| Component | Function |
|---|---|
| **GATv2** — Graph Attention Network | Learns cross-timeframe relationships dynamically — which timeframe should influence which, and by how much |
| **GIN** — Graph Isomorphism Network | Captures structural market patterns within the graph that attention alone may miss |
| **LSTM + Self-Attention** | Extracts temporal dependencies from the price sequence |
| **NLP Encoder** | Encodes news sentiment as a direct quantitative input to the model |

All four outputs are merged through a **CrossModalAttention** layer and a **Multi-Modal Fusion gate** with entropy regularization. This prevents any single component from dominating — all four must contribute meaningfully to every prediction.

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
| Current run — 120 epochs | **41.8%** (2.09× above random) — confirmed |

**Why 41.8% on 5 classes matters:**
Most published research reports binary classification (up vs down) where random baseline is 50%. Getting to 53% on binary — as seen in recent GNN + news papers — means being only 1.06× above random. Neural-Nexus at 41.8% on 5 classes is 2.09× above random — meaningfully stronger relative to chance than most published results on simpler tasks, and a clear improvement over the previous 1.84×.

The improvement from 36.7% to 41.8% between runs came from three changes: 3× more training samples (16,000 → 46,000), 42% more news coverage, and 20 additional training epochs with better stability controls.

---

## Training Stability

The current version includes a multi-layer stability system built directly into the training loop. This was developed in response to real instability observed during training on live hardware — not added preemptively.

### NaN Guard — 4 layers

Numerical instability (NaN/Inf loss) can silently corrupt a training run if undetected. The system intercepts it at four independent points:

**Layer 1 — Model outputs:** Forward pass output is checked before loss computation. If any value is non-finite, the batch is skipped immediately.

**Layer 2 — Loss value:** Loss is checked before backward pass. Non-finite loss triggers immediate batch skip with gradient clear.

**Layer 3 — Gradients:** All parameter gradients are verified before the optimizer step. Non-finite gradients trigger skip without updating weights.

**Layer 4 — Parameters:** Model weights are verified after each optimizer step. If parameters become non-finite, the step is rolled back.

### Automatic epoch recovery

If invalid-loss batches exceed 25% of an epoch, the system triggers automatic recovery: learning rate is reduced, AMP (mixed precision) is disabled if needed, and the best checkpoint is restored before continuing to the next epoch.

### Sample quarantine

Batches that produce NaN loss have their source sample indices flagged. These samples are excluded from subsequent epochs. A maximum quarantine ratio of 30% prevents over-pruning the dataset.

### Checkpoint integrity

When resuming from a saved checkpoint, all model parameters are verified for finite values before loading. A corrupted checkpoint is detected and discarded — training restarts from scratch rather than propagating the corruption.

### Overfitting auto-control

The system monitors overfitting per epoch and applies graduated responses — learning rate decay and dropout increase when overfitting exceeds the soft threshold, with escalated response at the hard threshold. Target band: 0.000–0.001 throughout training.

---

## Overfitting Resistance

Overfitting occurs when a model memorizes training data rather than learning general patterns — it performs well on seen data but fails on unseen data. This is one of the most common failure modes in ML trading systems.

| Run | Epochs | Overfitting at 0.000 | Overfitting at 0.001 | Peak |
|---|---|---|---|---|
| Previous run | 100 | ~95 epochs | ~5 epochs | 0.001 |
| Current run | 120 | 109 epochs | 11 epochs | 0.001 |

Both runs stayed within the 0.000–0.001 band throughout. Epochs touching 0.001 were scattered with no accumulation trend — always returning to 0.000 on the following epoch. Neither run ever crossed above 0.001.

### Why overfitting stays low

**Multi-modal diversity** — Four separate components each learn from a different view of the same market. For overfitting to occur, all four would need to memorize the same patterns simultaneously. If one component drifts, the others remain stable and pull the output back toward generalization.

**Fusion gate with entropy regularization** — The gate combining all four components is actively penalized during training if it collapses too much weight onto any single component. The model cannot overfit by routing everything through one pathway.

**Agent diversity term** — Four internal agents produce independent predictions. The training loss penalizes the agents for agreeing too strongly — encouraging diverse perspectives rather than converging to a single overfit solution.

**Epoch bagging** — Each epoch trains on a random 85% subset of available samples, reducing the model's ability to memorize any specific example.

**Label smoothing** — The model trains on softened targets rather than hard labels. This prevents overconfidence on any specific training example.

**Return-weighted loss** — Samples with larger actual market moves receive higher weight, de-emphasizing borderline or noisy samples that would otherwise encourage memorization.

**EWC — Elastic Weight Consolidation** — Preserves knowledge from earlier in training by penalizing large changes to parameters that were previously important. Prevents the model from overwriting general patterns with late-stage specifics.

**Dropout, weight decay, gradient clipping** — Standard regularization applied throughout all components.

**Expanding validation** — Validation set grows over time rather than remaining fixed, testing the model against an increasing proportion of unseen data as training progresses.

---

## How It Compares to Published Research

This section is included for context — not to claim superiority. The field is genuinely difficult and results are rarely directly comparable across studies.

### What most research does

A February 2025 arXiv paper combining LSTM + GNN for stock prediction achieved a 10.6% improvement in MSE over standalone LSTM — tested on historical backtest data only. The same paper acknowledges that empirical evaluation in real-world conditions remains limited for hybrid LSTM-GNN models.

A hybrid GNN model integrating news achieved 53% accuracy on binary stock movement — representing a 1% absolute gain over the LSTM baseline, tested on daily data with no live execution.

A 2025 ScienceDirect review of 187 deep learning financial forecasting studies confirms the dominance of LSTM-based models, while noting that hybrid architectures combining GNN with other components are gaining ground.

### What Neural-Nexus does differently

| Dimension | Most Published Research | Neural-Nexus |
|---|---|---|
| Task | Binary (up/down) or regression | 5-class (direction + strength) |
| Data frequency | Daily candles | Intraday — M1 through H4 |
| Timeframes | Single timeframe | 5 timeframes as a unified graph |
| News integration | Optional add-on | Core component, live-aligned |
| Validation | Historical backtest only | Live market — MT5 Demo |
| Execution layer | None (paper models) | Full MT5 order execution |
| Overfitting (120 epochs) | Often present | 0.000–0.001 — never exceeded |
| Training stability | Not addressed | 4-layer NaN guard + auto recovery |

The most significant difference is execution. Most hybrid LSTM-GNN studies remain paper models. Neural-Nexus has been running on live market data, including the March 19, 2026 gold crash.

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
| Epochs | 100 | 120 |
| Training samples | ~16,000 | 46,000 (+188%) |
| News events | 2,904 | 4,129 (+42%) |
| Batch size | 32 | 32 (effective 64 with accumulation) |
| Timeframes | M1 · M5 · M15 · H1 · H4 | M1 · M5 · M15 · H1 · H4 |
| Validation mode | Fixed split | Expanding split |
| NaN guard | Basic | 4-layer + auto recovery + quarantine |
| Overfitting control | Passive | Active auto-control with target band |
| Peak overfitting | 0.001 | 0.001 |
| Validated accuracy | 36.7% | **41.8%** (+5.1pp) |

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

If any single filter fails → **HOLD**. The system waits for the next cycle. Knowing when not to act is as important as knowing when to act.

**Live test result:** During the March 19, 2026 gold market crash — one of the sharpest single-day selloffs in recent history — the system held all positions closed for over 8 hours without a single order. Balance unchanged.

---

## Training Methodology

- **Input:** Multi-timeframe OHLCV data across 5 timeframes + news sentiment vectors
- **Labels:** Derived from trade outcome simulation — whether TP or SL would have been hit — not raw price direction
- **Sampling:** Streaming mode — samples drawn continuously from historical data; training begins immediately without waiting for a full data scan
- **Validation:** Expanding time-based split — validation set grows as training progresses; future data is never seen during training
- **Checkpointing:** Full state saved every epoch — model, optimizer, scheduler, epoch number. Resumes from last valid checkpoint if interrupted. Corrupted checkpoints are detected and discarded automatically.

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

120-epoch training run complete — March 20, 2026.
Validated accuracy: **41.8%** (2.09× above random on 5-class prediction).
Peak overfitting: **0.001** across 120 epochs.
Live testing on MT5 Demo account in progress.

Results are instrument-specific (XAUUSD) and depend on training data, market conditions, and configuration. Credentials and account details are stored locally and never shared or committed to any repository.
