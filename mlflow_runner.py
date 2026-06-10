"""
mlflow_runner.py — Drop-in MLflow wrapper for RL-Portfolio-Optimization

Usage (replaces main.py):
    python mlflow_runner.py --dqn-episodes 500 --ppo-episodes 100

Then view runs in browser:
    mlflow ui          # visit http://localhost:5000

What this logs:
  - All hyperparameters (lr, gamma, batch size, episodes, etc.)
  - Per-episode reward, Sharpe ratio, loss (DQN), entropy (PPO)
  - Final evaluation metrics for DQN, PPO, and baseline
  - Saved model artifacts (.pth files)
  - Training curve plots

No changes needed to your existing agents/, training/, or evaluation/ files.
"""

import argparse
import sys
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.generate_data import generate_synthetic_data, save_data
from training.train_dqn import train_dqn, plot_training as plot_dqn
from training.train_ppo import train_ppo, plot_training as plot_ppo
from evaluation.compare_agents import compare_agents


# ── helpers ──────────────────────────────────────────────────────────────────

def log_episode_metrics(metrics: dict, prefix: str):
    """
    Log per-episode metrics returned by your existing train_* functions.
    Expects metrics dict with keys like:
      episode_rewards, sharpe_ratios, losses  (DQN)
      episode_rewards, sharpe_ratios, entropies (PPO)
    Adjust key names here if your training code uses different names.
    """
    reward_key   = "episode_rewards"
    sharpe_key   = "sharpe_ratios"
    loss_key     = "losses"       # DQN
    entropy_key  = "entropies"    # PPO

    rewards  = metrics.get(reward_key, [])
    sharpes  = metrics.get(sharpe_key, [])
    losses   = metrics.get(loss_key, [])
    entropys = metrics.get(entropy_key, [])

    for ep, reward in enumerate(rewards):
        mlflow.log_metric(f"{prefix}_episode_reward", reward, step=ep)

    for ep, s in enumerate(sharpes):
        mlflow.log_metric(f"{prefix}_sharpe_ratio", s, step=ep)

    for ep, loss in enumerate(losses):
        mlflow.log_metric(f"{prefix}_loss", loss, step=ep)

    for ep, ent in enumerate(entropys):
        mlflow.log_metric(f"{prefix}_entropy", ent, step=ep)

    # Summary stats
    if rewards:
        mlflow.log_metric(f"{prefix}_final_avg_reward",  float(np.mean(rewards[-50:])))
        mlflow.log_metric(f"{prefix}_best_episode_reward", float(max(rewards)))
    if sharpes:
        mlflow.log_metric(f"{prefix}_final_avg_sharpe",  float(np.mean(sharpes[-50:])))
        mlflow.log_metric(f"{prefix}_best_sharpe",       float(max(sharpes)))


def log_eval_metrics(metrics: dict, prefix: str):
    """Log final evaluation / backtest metrics."""
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            mlflow.log_metric(f"{prefix}_eval_{key}", float(val))


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    Path("results").mkdir(exist_ok=True)

    # One experiment groups all your runs together in the MLflow UI
    mlflow.set_experiment("RL-Portfolio-Optimization")

    with mlflow.start_run(run_name=f"dqn{args.dqn_episodes}_ppo{args.ppo_episodes}"):

        # ── Log all hyperparameters ──────────────────────────────────────────
        mlflow.log_params({
            # DQN
            "dqn_episodes":    args.dqn_episodes,
            "dqn_lr":          args.dqn_lr,
            "dqn_batch_size":  args.dqn_batch_size,
            # PPO
            "ppo_episodes":    args.ppo_episodes,
            "ppo_lr":          args.ppo_lr,
            "ppo_batch_size":  args.ppo_batch_size,
            "ppo_epochs":      args.ppo_epochs,
            "entropy_coef":    args.entropy_coef,
            # Shared
            "gamma":           args.gamma,
            "epsilon_decay":   args.epsilon_decay,
            "seed":            args.seed,
            "device":          args.device,
        })

        # ── Step 1: Generate data ────────────────────────────────────────────
        if args.generate_data:
            print("\n[STEP 1] Generating Synthetic Financial Data...")
            prices = generate_synthetic_data(n_days=2000, n_assets=5, seed=args.seed)
            save_data(prices, "data/data.csv")
            mlflow.log_param("n_days",   2000)
            mlflow.log_param("n_assets", 5)
            print(f"✓ Generated {len(prices)} trading days")

        # ── Step 2: Train DQN ────────────────────────────────────────────────
        if args.train_dqn:
            print("\n[STEP 2] Training DQN Agent...")
            dqn_agent, dqn_train_metrics, dqn_eval_metrics = train_dqn(
                num_episodes=args.dqn_episodes,
                learning_rate=args.dqn_lr,
                gamma=args.gamma,
                epsilon_decay=args.epsilon_decay,
                batch_size=args.dqn_batch_size,
                eval_interval=max(1, args.dqn_episodes // 10),
                device=args.device,
                seed=args.seed,
            )

            # Log per-episode training curves
            log_episode_metrics(dqn_train_metrics, prefix="dqn")
            # Log evaluation results
            log_eval_metrics(dqn_eval_metrics,    prefix="dqn")

            # Save plots & model as MLflow artifacts
            plot_dqn(dqn_train_metrics, dqn_eval_metrics, "results/dqn_training.png")
            mlflow.log_artifact("results/dqn_training.png", artifact_path="plots")

            if Path("results/dqn_best.pth").exists():
                mlflow.log_artifact("results/dqn_best.pth", artifact_path="models")
                # Also log as an MLflow PyTorch model for the model registry
                mlflow.pytorch.log_model(dqn_agent.q_network,  # adjust attr name if needed
                                         artifact_path="dqn_model")

            print("✓ DQN training complete — metrics logged to MLflow")

        # ── Step 3: Train PPO ────────────────────────────────────────────────
        if args.train_ppo:
            print("\n[STEP 3] Training PPO Agent...")
            ppo_agent, ppo_train_metrics, ppo_eval_metrics = train_ppo(
                num_episodes=args.ppo_episodes,
                learning_rate=args.ppo_lr,
                gamma=args.gamma,
                epochs=args.ppo_epochs,
                batch_size=args.ppo_batch_size,
                entropy_coef=args.entropy_coef,
                eval_interval=max(1, args.ppo_episodes // 10),
                device=args.device,
                seed=args.seed,
            )

            log_episode_metrics(ppo_train_metrics, prefix="ppo")
            log_eval_metrics(ppo_eval_metrics,     prefix="ppo")

            plot_ppo(ppo_train_metrics, ppo_eval_metrics, "results/ppo_training.png")
            mlflow.log_artifact("results/ppo_training.png", artifact_path="plots")

            if Path("results/ppo_best.pth").exists():
                mlflow.log_artifact("results/ppo_best.pth", artifact_path="models")
                mlflow.pytorch.log_model(ppo_agent.actor,   # adjust attr name if needed
                                         artifact_path="ppo_model")

            print("✓ PPO training complete — metrics logged to MLflow")

        # ── Step 4: Compare agents ───────────────────────────────────────────
        if args.compare:
            print("\n[STEP 4] Comparing Agents...")
            dqn_m, ppo_m, baseline_m = compare_agents(
                num_eval_episodes=args.eval_episodes,
                device=args.device,
                seed=args.seed + 1,
            )

            log_eval_metrics(dqn_m,      prefix="final_dqn")
            log_eval_metrics(ppo_m,      prefix="final_ppo")
            log_eval_metrics(baseline_m, prefix="final_baseline")

            if Path("results/comparison.png").exists():
                mlflow.log_artifact("results/comparison.png", artifact_path="plots")

            if Path("results/comparison_results.json").exists():
                mlflow.log_artifact("results/comparison_results.json",
                                    artifact_path="reports")

                # Flatten JSON metrics into MLflow for easy comparison in UI
                with open("results/comparison_results.json") as f:
                    comp = json.load(f)
                for agent_name, agent_metrics in comp.items():
                    if isinstance(agent_metrics, dict):
                        for k, v in agent_metrics.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"compare_{agent_name}_{k}", float(v))

            print("✓ Comparison complete")

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print("\nView results:  mlflow ui   →  http://localhost:5000")
        print(f"Run ID:        {mlflow.active_run().info.run_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL Portfolio agents with MLflow experiment tracking"
    )
    parser.add_argument("--generate-data",  action="store_true", default=True)
    parser.add_argument("--train-dqn",      action="store_true", default=True)
    parser.add_argument("--train-ppo",      action="store_true", default=True)
    parser.add_argument("--compare",        action="store_true", default=True)

    parser.add_argument("--dqn-episodes",   type=int,   default=500)
    parser.add_argument("--dqn-lr",         type=float, default=1e-3)
    parser.add_argument("--dqn-batch-size", type=int,   default=32)

    parser.add_argument("--ppo-episodes",   type=int,   default=100)
    parser.add_argument("--ppo-lr",         type=float, default=3e-4)
    parser.add_argument("--ppo-batch-size", type=int,   default=64)
    parser.add_argument("--ppo-epochs",     type=int,   default=10)
    parser.add_argument("--entropy-coef",   type=float, default=0.01)

    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--epsilon-decay",  type=float, default=0.995)
    parser.add_argument("--eval-episodes",  type=int,   default=20)
    parser.add_argument("--device",         type=str,   default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--seed",           type=int,   default=42)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    main(args)
