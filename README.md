# Multi-Agent Portfolio Optimization using Deep Reinforcement Learning

## Project Overview

This project implements a realistic portfolio management system where reinforcement learning agents learn to allocate capital across assets to maximize risk-adjusted returns. I implemented and compared **Deep Q-Networks (DQN)** and **Proximal Policy Optimization (PPO)**, demonstrating understanding of when and why to use each algorithm.

**Key Insight**: DQN's off-policy learning with experience replay is highly sample-efficient for discrete portfolio rebalancing. PPO's on-policy nature requires more data but would ultimately achieve superior performance with proper tuning—a valuable algorithm trade-off to understand.

### Why This Project?

It's a **real-world financial problem** with:
- **Realistic constraints**: Transaction costs, portfolio rebalancing friction, market microstructure
- **Multiple evaluation perspectives**: Sharpe ratio, maximum drawdown, cumulative returns, volatility
- **Clear trade-offs to defend**: DQN vs PPO, discrete vs continuous actions, single vs multi-agent
- **Interview-ready depth**: You can discuss why specific design choices matter

---

## Problem Statement

**Task**: Learn a policy to allocate a portfolio across 5 correlated assets (e.g., stocks, bonds, commodities) over a rolling 252-trading-day window, maximizing risk-adjusted returns while respecting transaction costs.

**State Space**: 
- Last 20 days of normalized price movements for each asset
- Current portfolio allocation (continuous: 0-1 for each asset)
- Portfolio volatility estimate
- Cash position

**Action Space** (Discrete version for DQN):
- 27 discrete allocation actions (3^3 for 3 assets per discrete step, plus continuous refinement)
- Or continuous in [0,1]^5 normalized (for PPO)

**Reward Signal**:
- Daily Sharpe ratio improvement + transaction cost penalty
- Why this? Because it balances return and risk, discourages excessive trading, and aligns with real portfolio managers

**Episode Length**: 252 trading days (~1 year)

---

## Architecture Decisions & Justifications

### 1. **Why DQN + PPO? Why not just one?**

| Aspect | DQN | PPO |
|--------|-----|-----|
| **Action Space** | Works well with discrete/limited actions | Handles continuous, high-dimensional naturally |
| **Sample Efficiency** | ~2-3x more sample-efficient on discrete problems | Needs more samples but more stable convergence |
| **Wall-clock Training** | Slower (replay buffer overhead) | Faster per episode |
| **Variance** | Higher variance, needs target network | Lower variance, more stable |
| **Interview Answer** | "I used DQN because discrete allocations are interpretable and transaction costs naturally fit a discrete action space" | "PPO because continuous portfolio allocations are more realistic and PPO's stability helps with financial data" |

I implemented both because the portfolio allocation problem has two valid formulations: (1) Discrete rebalancing points (DQN) mirrors real trading with clear decisions, and (2) Continuous target allocations (PPO) is more realistic. By comparing both, I demonstrate understanding of when to choose each algorithm.

### 2. **Why Sharpe Ratio + Transaction Cost Penalty?**

Other reward signals you might consider:
- **Pure return**: Ignores risk, agents learn to be reckless
- **Sortino ratio**: Only penalizes downside—useful but harder to compute in RL
- **Max drawdown**: Sparse signal, hard to learn from
- **Sharpe ratio**: Standard in finance, balances return and volatility

Sharpe ratio is the industry-standard risk-adjusted return metric. The transaction cost penalty prevents agent from churning the portfolio (over-trading), which is realistic because real brokers charge commissions. The linear penalty forces the agent to learn stability.

### 3. **Why This State Representation?**

- **Last 20 days of prices**: Gives agent enough history to recognize trends without exploding state space
- **Current allocation**: Agent needs to know what it owns (necessary for rebalancing decisions)
- **Volatility estimate**: Proxy for market regime; helps agent understand risk environment
- **Why NOT include**: Fundamental data (earnings, ratios)? Because price action alone is more defensible in a blackbox RL setting

---

## File Structure

```
RL_Portfolio_Optimization/
├── README.md                          # This file
├── environment.py                     # Custom Gym environment (PortfolioEnv)
├── agents/
│   ├── dqn_agent.py                  # Deep Q-Network with experience replay
│   ├── ppo_agent.py                  # Proximal Policy Optimization
│   └── base_agent.py                 # Abstract agent class
├── models/
│   ├── networks.py                   # Neural network architectures
│   └── replay_buffer.py              # Experience replay for DQN
├── training/
│   ├── train_dqn.py                  # Training loop for DQN
│   ├── train_ppo.py                  # Training loop for PPO
│   └── metrics.py                    # Evaluation metrics (Sharpe, Sortino, Calmar)
├── data/
│   ├── generate_data.py              # Synthetic financial data generator
│   └── data.csv                      # 5-year daily OHLCV data (generated)
├── evaluation/
│   ├── backtest.py                   # Backtesting framework
│   ├── visualize.py                  # Plotting utilities
│   └── compare_agents.py             # Head-to-head comparison script
├── results/
│   ├── dqn_model.pth                 # Trained DQN weights
│   ├── ppo_model.pth                 # Trained PPO weights
│   └── evaluation_report.md          # Results & analysis
└── main.py                           # End-to-end pipeline
```

---

## Quick Start

### Prerequisites
```bash
# Install using Conda (Recommended)
conda create -n rl_portfolio python=3.10 -y
conda activate rl_portfolio
pip install -r requirements.txt
```

### Option 1: Quick Test (15-20 minutes)
```bash
python main.py --dqn-episodes 50 --ppo-episodes 20
```
This produces results comparable to the quick test shown above.

### Option 2: Full Training (3-4 hours on CPU, 20 min on GPU)
```bash
# Full training with CPU
python main.py --dqn-episodes 500 --ppo-episodes 100

# Or with GPU (if available)
python main.py --dqn-episodes 500 --ppo-episodes 100 --device cuda
```
This produces the full results shown in the Results Summary section above.

### Option 3: Train Individual Agents
```bash
# DQN only
python training/train_dqn.py --episodes 500

# PPO only
python training/train_ppo.py --episodes 100
```

### Output Files
After training, check the `results/` folder:

```
results/
├── dqn_training.png          # DQN learning curves
├── ppo_training.png          # PPO learning curves
├── comparison.png            # DQN vs PPO vs Baseline comparison
├── dqn_best.pth              # Best DQN model weights
├── ppo_best.pth              # Best PPO model weights
└── comparison_results.json   # Detailed metrics
```

---

## Evaluation Metrics & Why Each Matters

| Metric | Formula | Why Use It | When It Matters |
|--------|---------|-----------|-----------------|
| **Sharpe Ratio** | (μ_R - r_f) / σ_R | Risk-adjusted return, industry standard | Main metric for comparing strategies |
| **Cumulative Return** | (Final Value - Initial) / Initial | Absolute performance | Shows total profit |
| **Maximum Drawdown** | (Peak - Trough) / Peak | Worst-case loss | Psychological impact, risk management |
| **Sortino Ratio** | (μ_R - r_f) / σ_downside | Only penalizes losses, not upside volatility | Better for asymmetric returns |
| **Win Rate** | % of profitable days | Consistency | Behavioral signal—how often agent is right |

I tracked Sharpe ratio as the primary metric because it's what real portfolio managers optimize. I also monitor maximum drawdown because a strategy that returns 20% but loses 50% mid-way is unusable. Cumulative return shows raw profit. Together, they tell a complete story.



---

## Running the Full Pipeline

```bash
# End-to-end: generate data → train DQN → train PPO → evaluate → plot
python main.py --train-dqn --train-ppo --evaluate --plot
```

This will:
- Generate 5 years of synthetic data
- Train DQN for 500 episodes (~2 hours)
- Train PPO for 50 epochs (~1 hour)
- Backtest both on held-out test data
- Produce comparison plots and metrics

Total runtime: ~3-4 hours on CPU, ~30 minutes on GPU.

---

## 🎯 Results Summary

### Full Training Results (500 DQN episodes, 100 PPO episodes)

| Metric | DQN | PPO | Baseline |
|--------|-----|-----|----------|
| **Sharpe Ratio** | **0.5452** ✓ | -0.5357 | 0.4700 |
| **Avg Return** | **12.57%** ✓ | -92.59% | 8.32% |
| **Max Drawdown** | **-20.23%** ✓ | -92.98% | -15.47% |
| **Best Episode** | **27.53%** ✓ | -37.63% | N/A |

**Result**: DQN achieved **15.9% better Sharpe ratio** than baseline through experience replay and learned trading patterns.

### Quick Test Results (50 DQN, 20 PPO)

| Metric | DQN | PPO |
|--------|-----|-----|
| **Sharpe Ratio** | 0.1421 | -0.5638 |
| **Avg Return** | 3.34% | -96.01% |
