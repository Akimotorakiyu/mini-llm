"""
Training script for Mini LLM on Daily Dialog dataset.
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from mini_llm.model import MiniLLM, MiniLLMConfig, create_mini_llm
from mini_llm.data import create_dataloader, get_vocab_size


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_model_and_optimizer(
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    learning_rate: float,
    weight_decay: float,
):
    """Setup model and optimizer."""
    # Create model
    model = create_mini_llm(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
    )
    
    device = get_device()
    model = model.to(device)
    
    print(f"Model configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden dim: {dim}")
    print(f"  Num layers: {n_layers}")
    print(f"  Num heads: {n_heads}")
    print(f"  Num KV heads: {n_kv_heads}")
    print(f"  Max seq len: {max_seq_len}")
    
    info = model.estimate_memory()
    print(f"  Parameters: {info['parameters']:,} ({info['parameters']/1e6:.1f}M)")
    print(f"  Estimated memory: {info['memory_gb']:.2f} GB")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    
    return model, optimizer, device


def setup_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Setup learning rate scheduler with warmup and cosine decay."""
    base_lr = optimizer.defaults["lr"]
    min_lr = base_lr * min_lr_ratio
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return 0.01 + (1.0 - 0.01) * (current_step / warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * (0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: MiniLLM,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits, loss = model(inputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })
        
        # Log periodically
        if step % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            pbar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model: MiniLLM, dataloader, device):
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for inputs, targets in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save a checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": model.config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load a checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def main():
    parser = argparse.ArgumentParser(description="Train Mini LLM")
    
    # Model arguments
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=4, help="Number of KV heads (GQA)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Other arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval (epochs)")
    parser.add_argument("--eval_interval", type=int, default=1, help="Eval interval (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get vocabulary size
    vocab_size = get_vocab_size(args.tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Setup model and optimizer
    model, optimizer, device = setup_model_and_optimizer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dataloader = create_dataloader(
        split="train",
        tokenizer_name=args.tokenizer,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    valid_dataloader = create_dataloader(
        split="validation",
        tokenizer_name=args.tokenizer,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    
    # Setup scheduler
    scheduler = setup_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1
    
    # Training loop
    print("\nStarting training...")
    best_valid_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_interval == 0:
            valid_loss, valid_ppl = evaluate(model, valid_dataloader, device)
            print(f"Valid loss: {valid_loss:.4f}, Perplexity: {valid_ppl:.2f}")
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, valid_loss,
                    os.path.join(args.output_dir, "best_model.pt")
                )
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            )
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
