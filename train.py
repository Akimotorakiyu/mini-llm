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


def estimate_max_batch_size(
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    target_memory_gb: float = None,
) -> int:
    """
    估算给定显存下的最大 batch_size（保守估计，避免 OOM）。
    
    显存占用 ≈ 模型参数 + 优化器状态 + 激活值 + 梯度 + 临时缓冲区
    """
    if not torch.cuda.is_available():
        return 4  # CPU 模式下默认 batch_size
    
    # 获取可用显存（留 25% 余量防止 OOM，因为激活值和临时内存难以精确计算）
    if target_memory_gb is None:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        available = (total_memory - allocated) * 0.75  # 留 25% 安全余量
        target_memory_gb = available / (1024 ** 3)
    
    target_memory_bytes = target_memory_gb * (1024 ** 3)
    
    # 计算模型参数数量（与 MiniLLM.estimate_memory() 保持一致）
    head_dim = dim // n_heads
    # 词嵌入
    embed_params = vocab_size * dim
    # 各层参数
    q_proj = dim * dim
    k_proj = dim * n_kv_heads * head_dim
    v_proj = dim * n_kv_heads * head_dim
    o_proj = dim * dim
    # FFN: SwiGLU - w1, w2, w3
    ffn_hidden = 256 * ((int(4 * dim * 2 / 3) + 256 - 1) // 256)  # 保持与模型一致
    ffn_params = dim * ffn_hidden * 3
    # 每层参数
    layer_params = q_proj + k_proj + v_proj + o_proj + ffn_params
    # 总参数
    total_params = embed_params + n_layers * layer_params + vocab_size * dim + dim  # + lm_head + final norm
    
    # 参数内存（混合精度训练）
    # FP16 参数: 2 bytes/param
    # FP16 梯度: 2 bytes/param
    # FP32 优化器状态 (AdamW): 4 bytes/param (copy) + 4 bytes (momentum) + 4 bytes (variance) = 12 bytes/param
    param_memory = total_params * 4  # FP16 params + gradients
    optimizer_memory = total_params * 12  # FP32 copy + momentum + variance
    
    # 固定开销（CUDA context, cuDNN workspace等）
    fixed_overhead = 1.0 * (1024 ** 3)  # 1GB
    
    # 留给激活值的显存
    activation_budget = target_memory_bytes - param_memory - optimizer_memory - fixed_overhead
    
    if activation_budget <= 0:
        print(f"Warning: Model itself may exceed available memory!")
        print(f"  Target: {target_memory_gb:.2f} GB")
        print(f"  Model params: {param_memory / (1024**3):.2f} GB")
        print(f"  Optimizer state: {optimizer_memory / (1024**3):.2f} GB")
        return 1
    
    # 更准确地估算每个样本的激活值内存
    # 在 Transformer 中，激活值主要包括：
    # 1. 输入嵌入: seq_len * dim * 2 bytes
    # 2. 每层:
    #    - Attention: Q, K, V 投影输出 + attention scores + attention output
    #    - FFN: 两个线性层的中间结果（SwiGLU 有 2 个投影）
    #    - LayerNorm 输入（用于反向传播）
    
    head_dim = dim // n_heads
    
    # 嵌入层
    embed_activation = max_seq_len * dim * 2
    
    # 每层激活值
    # Attention: Q, K, V (seq_len, dim) + attention scores (seq_len, seq_len, n_heads) + output (seq_len, dim)
    # 加上一些中间计算的缓冲
    attn_activation = max_seq_len * dim * 3 * 2  # Q, K, V
    attn_activation += max_seq_len * max_seq_len * n_heads * 2  # attention scores
    attn_activation += max_seq_len * dim * 2  # attention output
    
    # FFN (SwiGLU): up_proj + gate_proj + down_proj 输入
    ffn_hidden = int(8 * dim / 3)
    ffn_activation = max_seq_len * ffn_hidden * 3 * 2  # gate, up, and intermediate
    
    # LayerNorm 保存的输入
    norm_activation = max_seq_len * dim * 2 * 2  # 2 layernorms per layer
    
    layer_activation = attn_activation + ffn_activation + norm_activation
    
    # 所有层 + 嵌入 + 最终输出
    total_activation_per_sample = embed_activation + (layer_activation * n_layers) + (max_seq_len * dim * 2)
    
    # 反向传播时梯度计算需要额外的临时内存（约为前向的 1.5-2 倍）
    # 再加上一些 CUDA kernel 的 workspace
    bytes_per_sample = int(total_activation_per_sample * 2.5)
    
    max_batch = int(activation_budget / bytes_per_sample)
    
    # 进一步限制上限，避免极端值
    max_batch = max(1, min(max_batch, 64))
    
    return max_batch


def auto_batch_config(
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    target_batch_size: int = None,
    max_accumulation: int = 16,
) -> tuple:
    """
    自动计算 batch_size 和 gradient_accumulation_steps。
    
    目标：找到最佳组合，使得：
    1. batch_size 尽可能大（减少 accumulation，提高速度）
    2. effective_batch_size ≥ target_batch_size（如果指定）
    
    Returns:
        (batch_size, gradient_accumulation_steps)
    """
    max_batch = estimate_max_batch_size(
        vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len
    )
    
    print(f"Estimated max batch_size for available memory: {max_batch}")
    
    if target_batch_size is None:
        # 没有目标，就用最大可能的 batch_size，不累积
        return max_batch, 1
    
    # 需要达到 target_batch_size 的有效 batch size
    if max_batch >= target_batch_size:
        # 显存足够，直接用目标 batch size
        return target_batch_size, 1
    
    # 需要梯度累积
    # 优先使用较大的 batch_size，较小的 accumulation_steps
    for accum in range(1, max_accumulation + 1):
        effective = max_batch * accum
        if effective >= target_batch_size:
            return max_batch, accum
    
    # 即使最大累积也达不到目标，返回最大配置
    return max_batch, max_accumulation


def find_batch_size_by_trial(
    model_fn,
    vocab_size: int,
    max_seq_len: int,
    start_batch: int = 1,
) -> int:
    """
    通过实际试错找到最大 batch_size（更准确但较慢）。
    
    Args:
        model_fn: 创建模型的函数
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
        start_batch: 起始 batch_size
    
    Returns:
        能跑的最大 batch_size（留一点余量）
    """
    if not torch.cuda.is_available():
        return start_batch
    
    device = torch.device("cuda")
    batch_size = start_batch
    
    print("Finding max batch_size by trial...")
    
    while True:
        try:
            # 清理显存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 创建模型和 dummy 输入
            model = model_fn().to(device)
            dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
            dummy_target = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
            
            # 尝试前向 + 反向
            logits, loss = model(dummy_input, dummy_target)
            loss.backward()
            
            # 检查显存使用
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            print(f"  batch_size={batch_size}: {max_memory:.2f}GB / {total_memory:.2f}GB")
            
            # 如果还有 20% 以上余量，继续增大
            if max_memory < total_memory * 0.80:
                batch_size *= 2
            else:
                # 留 10% 余量
                safe_batch = max(1, int(batch_size * 0.9))
                print(f"  Safe batch_size: {safe_batch}")
                del model, dummy_input, dummy_target, loss
                torch.cuda.empty_cache()
                return safe_batch
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  batch_size={batch_size}: OOM")
                # 清理显存
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise RuntimeError("Cannot even run with batch_size=1! Model is too large.")
                # 返回上一次的 safe 值
                safe_batch = max(1, batch_size // 2)
                print(f"  Safe batch_size: {safe_batch}")
                return safe_batch
            else:
                raise


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
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits, loss = model(inputs, targets)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"\nWarning: NaN loss detected at step {step}")
            continue
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
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
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Other arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval (epochs)")
    parser.add_argument("--eval_interval", type=int, default=1, help="Eval interval (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples for testing (None = use all)")
    
    # Auto batch configuration
    parser.add_argument("--auto_batch", action="store_true", help="Automatically determine batch_size and gradient_accumulation_steps")
    parser.add_argument("--target_batch_size", type=int, default=None, help="Target effective batch size (used with --auto_batch)")
    parser.add_argument("--use_trial", action="store_true", help="Use trial-and-error to find max batch_size (more accurate but slower)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get vocabulary size
    vocab_size = get_vocab_size(args.tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Auto batch configuration
    if args.auto_batch:
        print("\n" + "="*50)
        print("Auto batch configuration enabled")
        print("="*50)
        
        if args.use_trial:
            # 试错法（更准确）
            from mini_llm.model import create_mini_llm
            
            def model_fn():
                return create_mini_llm(
                    vocab_size=vocab_size,
                    dim=args.dim,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    max_seq_len=args.max_seq_len,
                )
            
            max_batch = find_batch_size_by_trial(
                model_fn, vocab_size, args.max_seq_len, start_batch=1
            )
            
            if args.target_batch_size:
                accum = max(1, (args.target_batch_size + max_batch - 1) // max_batch)
                args.batch_size = max_batch
                args.gradient_accumulation_steps = min(accum, 16)
            else:
                args.batch_size = max_batch
                args.gradient_accumulation_steps = 1
        else:
            # 估算法（更快）
            batch_size, accum_steps = auto_batch_config(
                vocab_size=vocab_size,
                dim=args.dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                max_seq_len=args.max_seq_len,
                target_batch_size=args.target_batch_size,
                max_accumulation=16,
            )
            args.batch_size = batch_size
            args.gradient_accumulation_steps = accum_steps
        
        effective_batch = args.batch_size * args.gradient_accumulation_steps
        print(f"\nAuto configuration result:")
        print(f"  batch_size: {args.batch_size}")
        print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
        print(f"  effective_batch_size: {effective_batch}")
        print("="*50 + "\n")
    
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
        max_examples=args.max_examples,
    )
    
    valid_dataloader = create_dataloader(
        split="validation",
        tokenizer_name=args.tokenizer,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_examples=args.max_examples if args.max_examples else None,
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
