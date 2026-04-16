"""
Main Pipeline: End-to-End RL Portfolio Optimization

Run this script to:
1. Generate synthetic financial data
2. Train DQN agent (discrete actions)
3. Train PPO agent (continuous actions)
4. Evaluate both on test data
5. Compare performance
6. Generate final report

Expected runtime:
- DQN training: ~2 hours (CPU) / ~10 minutes (GPU)
- PPO training: ~1 hour (CPU) / ~5 minutes (GPU)
- Evaluation & comparison: ~5 minutes
- Total: ~3-4 hours (CPU) or ~20 minutes (GPU)
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.generate_data import generate_synthetic_data, save_data
from training.train_dqn import train_dqn, plot_training as plot_dqn
from training.train_ppo import train_ppo, plot_training as plot_ppo
from evaluation.compare_agents import compare_agents


def main(args):
    """Run full pipeline."""
    
    print("\n" + "="*100)
    print("MULTI-AGENT PORTFOLIO OPTIMIZATION USING DEEP REINFORCEMENT LEARNING")
    print("="*100)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # === STEP 1: Generate Data ===
    if args.generate_data:
        print("\n[STEP 1] Generating Synthetic Financial Data...")
        prices = generate_synthetic_data(n_days=2000, n_assets=5, seed=args.seed)
        save_data(prices, 'data/data.csv')
        print(f"✓ Generated {len(prices)} trading days of data")
    
    # === STEP 2: Train DQN ===
    if args.train_dqn:
        print("\n[STEP 2] Training DQN Agent (Discrete Actions)...")
        print("This may take 1-2 hours on CPU, 10 minutes on GPU")
        
        dqn_agent, dqn_train_metrics, dqn_eval_metrics = train_dqn(
            num_episodes=args.dqn_episodes,
            learning_rate=args.dqn_lr,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.dqn_batch_size,
            eval_interval=max(1, args.dqn_episodes // 10),
            device=args.device,
            seed=args.seed
        )
        
        # Plot DQN training
        plot_dqn(dqn_train_metrics, dqn_eval_metrics, 'results/dqn_training.png')
        print("✓ DQN training complete")
    
    # === STEP 3: Train PPO ===
    if args.train_ppo:
        print("\n[STEP 3] Training PPO Agent (Continuous Actions)...")
        print("This is faster than DQN: 30 minutes to 1 hour")
        
        ppo_agent, ppo_train_metrics, ppo_eval_metrics = train_ppo(
            num_episodes=args.ppo_episodes,
            learning_rate=args.ppo_lr,
            gamma=args.gamma,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            entropy_coef=args.entropy_coef,
            eval_interval=max(1, args.ppo_episodes // 10),
            device=args.device,
            seed=args.seed
        )
        
        # Plot PPO training
        plot_ppo(ppo_train_metrics, ppo_eval_metrics, 'results/ppo_training.png')
        print("✓ PPO training complete")
    
    # === STEP 4: Compare Agents ===
    if args.compare:
        print("\n[STEP 4] Comparing Agents on Test Data...")
        dqn_metrics, ppo_metrics, baseline_metrics = compare_agents(
            num_eval_episodes=args.eval_episodes,
            device=args.device,
            seed=args.seed + 1
        )
        print("✓ Comparison complete")
    
    print("\n" + "="*100)
    print("PIPELINE COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print("  ✓ results/dqn_training.png - DQN training curves")
    print("  ✓ results/ppo_training.png - PPO training curves")
    print("  ✓ results/comparison.png - Agent comparison plots")
    print("  ✓ results/dqn_best.pth - Best DQN model")
    print("  ✓ results/ppo_best.pth - Best PPO model")
    print("  ✓ results/comparison_results.json - Detailed comparison metrics")
    print("\nNext steps:")
    print("  1. Review the plots in results/")
    print("  2. Analyze the comparison results")
    print("  3. Try modifying hyperparameters and re-running")
    print("  4. Experiment with different reward signals and state representations")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL agents for portfolio optimization"
    )
    
    # Pipeline control
    parser.add_argument('--generate-data', action='store_true', default=True,
                       help='Generate synthetic data')
    parser.add_argument('--train-dqn', action='store_true', default=True,
                       help='Train DQN agent')
    parser.add_argument('--train-ppo', action='store_true', default=True,
                       help='Train PPO agent')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Compare agents')
    
    # DQN hyperparameters
    parser.add_argument('--dqn-episodes', type=int, default=500,
                       help='DQN training episodes')
    parser.add_argument('--dqn-lr', type=float, default=1e-3,
                       help='DQN learning rate')
    parser.add_argument('--dqn-batch-size', type=int, default=32,
                       help='DQN batch size')
    
    # PPO hyperparameters
    parser.add_argument('--ppo-episodes', type=int, default=100,
                       help='PPO training episodes')
    parser.add_argument('--ppo-lr', type=float, default=3e-4,
                       help='PPO learning rate')
    parser.add_argument('--ppo-batch-size', type=int, default=64,
                       help='PPO batch size')
    parser.add_argument('--ppo-epochs', type=int, default=10,
                       help='PPO update epochs per batch')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient for PPO')
    
    # Common hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='DQN epsilon decay')
    
    # Evaluation
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    
    # General
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='torch device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    main(args)
