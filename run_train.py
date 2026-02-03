"""
Simple script to start training with recommended settings for 1B parameter model.
"""

import subprocess
import sys

# Configuration for different model sizes
CONFIGS = {
    "tiny": {
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 2,
        "desc": "~100M parameters"
    },
    "small": {
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 4,
        "desc": "~400M parameters"
    },
    "base": {
        "dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 4,
        "desc": "~800M parameters"
    },
    "large": {
        "dim": 1280,
        "n_layers": 20,
        "n_heads": 20,
        "n_kv_heads": 5,
        "desc": "~1.5B parameters"
    },
}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Mini LLM")
    parser.add_argument("--config", type=str, default="small", choices=list(CONFIGS.keys()),
                        help="Model configuration")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    config = CONFIGS[args.config]
    print(f"Training with config: {args.config} ({config['desc']})")
    print(f"  dim: {config['dim']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  n_kv_heads: {config['n_kv_heads']}")
    
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--dim", str(config["dim"]),
        "--n_layers", str(config["n_layers"]),
        "--n_heads", str(config["n_heads"]),
        "--n_kv_heads", str(config["n_kv_heads"]),
        "--max_seq_len", str(args.max_seq_len),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
