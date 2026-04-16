# Multi-Agent Portfolio Optimization using Deep Reinforcement Learning

## Project Overview

This project implements a realistic portfolio management system where reinforcement learning agents learn to allocate capital across assets to maximize risk-adjusted returns. You'll implement and compare **Deep Q-Networks (DQN)** and **Proximal Policy Optimization (PPO)**, understanding when and why to use each.

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

I engineered the state to be Markovian—containing all information needed to make optimal future decisions. I limited history to 20 days because (1) longer sequences blow up memory, (2) portfolio decision horizons are typically 1-4 weeks, and (3) more history doesn't meaningfully improve performance on this dataset.

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

### 1. Install Dependencies
```bash
pip install numpy pandas torch gymnasium scikit-learn matplotlib seaborn
```

### 2. Generate Data
```bash
python data/generate_data.py
```
Creates realistic synthetic OHLCV data with correlations mimicking real markets.

### 3. Train DQN
```bash
python training/train_dqn.py --episodes 500 --batch-size 32 --gamma 0.99
```

### 4. Train PPO
```bash
python training/train_ppo.py --epochs 50 --batch-size 64 --lr 3e-4
```

### 5. Evaluate & Compare
```bash
python evaluation/compare_agents.py
```

Generates comparison plots and metrics table.

---

## Evaluation Metrics & Why Each Matters

| Metric | Formula | Why Use It | When It Matters |
|--------|---------|-----------|-----------------|
| **Sharpe Ratio** | (μ_R - r_f) / σ_R | Risk-adjusted return, industry standard | Main metric for comparing strategies |
| **Cumulative Return** | (Final Value - Initial) / Initial | Absolute performance | Shows total profit |
| **Maximum Drawdown** | (Peak - Trough) / Peak | Worst-case loss | Psychological impact, risk management |
| **Sortino Ratio** | (μ_R - r_f) / σ_downside | Only penalizes losses, not upside volatility | Better for asymmetric returns |
| **Win Rate** | % of profitable days | Consistency | Behavioral signal—how often agent is right |



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
