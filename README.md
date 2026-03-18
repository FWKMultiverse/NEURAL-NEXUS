# Neural-Nexus — XAUUSD AI Trading System

Automated trading engine for XAUUSD (Gold) using a Spatio-Temporal Graph Neural Network.  
Connects directly to MetaTrader 5 for live order execution.

> **Disclaimer:** This is a research and personal-use project. Live trading involves real financial risk. Always test on a Demo account first. Past model performance does not guarantee future results.

---

## Architecture Overview

The model combines four components that process market data simultaneously — not sequentially:

| Component | Role |
|---|---|
| **GATv2** (Graph Attention Network) | Learns relationships between timeframes as a unified graph |
| **GIN** (Graph Isomorphism Network) | Captures structural patterns within the graph |
| **LSTM + Attention** | Extracts temporal patterns from price sequences |
| **NLP Encoder** | Encodes news sentiment as a feature |

All four are fused through **CrossModalAttention** and a **Multi-Modal Fusion gate**, so the model sees the full picture — not isolated signals merged at the end.

**Timeframes used simultaneously:** M1 · M5 · M15 · H1 · H4

**Output:** 5-class signal — `BUY_STRONG` · `BUY` · `HOLD` · `SELL` · `SELL_STRONG`

---

## Requirements

- Windows (MT5 requires Windows)
- Python 3.10+
- NVIDIA GPU with CUDA (recommended: RTX 3060+)
- MetaTrader 5 with AutoTrading enabled
- 16GB RAM recommended

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pandas numpy python-dotenv MetaTrader5 rich
```

---

## Project Structure

```
Neural-Nexus/
├── neural_nexus_v3.2-ULTRA.py   # Main executable
├── .env                          # Credentials (do not share)
├── Data/
│   ├── XAUUSDm/                  # OHLCV CSV files (M1, M5, M15, H1, H4)
│   └── News/                     # News JSON files
├── models/                       # Saved checkpoints
└── logs/                         # Training and trade logs
```

---

## Setup

**1. Enable AutoTrading in MT5**

`Tools → Options → Expert Advisors → Allow Algo Trading`  
Make sure the AutoTrading button is green.

**2. Configure `.env`**

```env
MT5_DEMO_LOGIN=your_login
MT5_DEMO_PASSWORD=your_password
MT5_DEMO_SERVER=your_server
MT5_DEMO_PATH=C:\path\to\terminal64.exe
```

**3. Place data files**

- OHLCV CSVs → `Data/XAUUSDm/`
- News JSONs → `Data/News/`

---

## Usage

```bash
# Train then ask whether to go live
python neural_nexus_v3.2-ULTRA.py --demo

# Train then go live automatically
python neural_nexus_v3.2-ULTRA.py --demo --trade --interval 15

# Live only (no retraining, loads existing model)
python neural_nexus_v3.2-ULTRA.py --live --demo --interval 15

# Check open positions
python neural_nexus_v3.2-ULTRA.py --status

# Adjust batch and accumulation for your hardware
python neural_nexus_v3.2-ULTRA.py --demo --batch 24 --accum 3
```

**Effective batch size = `--batch` × `--accum`**  
Default: `BATCH=32, ACCUM=2` → effective batch 64

---

## Risk Management Layers

The system uses multiple independent protection mechanisms:

- **Emergency Exit** — closes position immediately if loss exceeds 2% of account
- **Circuit Breaker** — pauses trading when ATR spikes above 3× average
- **Position Protection** — pauses after 3 consecutive losses
- **RSI Reversal Bias** — suppresses SELL signals when RSI < 28 (oversold), suppresses BUY when RSI > 72
- **News Pre-Close** — closes all positions 3 minutes before high-impact news
- **Trailing Stop** — moves SL dynamically based on ATR

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `BATCH` | 32 | Real batch size per GPU step |
| `ACCUM_STEPS` | 2 | Gradient accumulation steps |
| `TARGET_TRAIN_SAMPLES` | 42,000 | Samples streamed per training run |
| `MAX_EP` | 100 | Training epochs |
| `MIN_CONF_EXEC` | 0.42 | Minimum confidence to send an order |
| `RISK_PCT` | 2% | Risk per trade |
| `MAX_DAILY` | 5% | Maximum daily drawdown |
| `SL_M` | 1.5× ATR | Stop-loss multiplier |
| `TP_M` | 2.5× ATR | Take-profit multiplier |

Override `MIN_CONF_EXEC` without touching the code:

```env
NN_MIN_CONF_EXEC=0.35
```

---

## Why HOLD

If the system outputs `HOLD` repeatedly, it means one or more filters rejected the signal:

- **Probability margin too low** — top-2 class probabilities too close → uncertain
- **Agent disagreement** — fewer than 2 of 4 agents agree → uncertain
- **Confidence below threshold** — `conf < MIN_CONF_EXEC`
- **RSI reversal bias active** — market momentum contradicts signal direction
- **Market regime** — `crisis` or `volatile` regime reduces allowed risk

This is intentional behavior. A system that knows when *not* to trade is more important than one that trades frequently.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `retcode=10027` | AutoTrading disabled in MT5 | Enable AlgoTrading in MT5 settings |
| Always HOLD | Signal filters rejecting trades | Check `MIN_CONF_EXEC` or reduce `MIN_PROB_MARGIN` |
| `conf` value identical every step | Running old model | Retrain or restart `--live` after training |
| RAM spike during dataset build | Too many samples | Reduce `TARGET_TRAIN_SAMPLES` or use `--batch 16` |
| GPU util stays low | CPU bottleneck (graph building) | Already handled via `TRAIN_PREFETCH=True` |

---

## Notes

- Model checkpoint: `models/nn31_best.pt`
- The system logs attribution per epoch showing which modality (GAT/GIN/LSTM/NLP) contributed most
- Retraining automatically loads the previous checkpoint and continues from where it left off
- News files are loaded once at startup (`NEWS_RELOAD_EVERY_STEPS=0`)
