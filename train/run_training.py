"""
IOS Risk Intelligence Core — Training CLI Entry Point
Execute via: python -m train.run_training [--smoke-test]
"""

import argparse
from train.config import TrainingConfig
from train.trainer import run_training_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="IOS Risk Brain #1 Fine-Tuning Pipeline"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2,000-sample trial test to verify GPU setup and loss convergence.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Manually cap dataset samples (e.g., 20000 for a single Kaggle session).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom directory to save trained LoRA adapters.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override default learning rate (e.g., 2e-4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig()

    # Apply CLI overrides if provided
    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE ACTIVATED: Setting max_samples=2000 <<<")
        config.max_samples = 2000
        config.save_steps = 50
        config.logging_steps = 5
        config.wandb_run_name = "llama3-8b-smoke-test"
    elif args.max_samples is not None:
        config.max_samples = args.max_samples

    if args.output_dir is not None:
        config.output_dir = args.output_dir

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    run_training_pipeline(config)


if __name__ == "__main__":
    main()
