# Training Guide

## Quick Start

### 1. Small Model (~400M parameters)

```bash
uv run python run_train.py --config small --epochs 3
```

This is equivalent to:
```bash
uv run python train.py \
    --dim 768 \
    --n_layers 12 \
    --n_heads 12 \
    --n_kv_heads 4 \
    --max_seq_len 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --learning_rate 3e-4
```

### 2. Base Model (~800M parameters, under 1B)

```bash
uv run python run_train.py --config base --epochs 3
```

Equivalent to:
```bash
uv run python train.py \
    --dim 1024 \
    --n_layers 16 \
    --n_heads 16 \
    --n_kv_heads 4 \
    --max_seq_len 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --learning_rate 3e-4
```

## Auto Batch Size Configuration

训练脚本支持自动计算最优的 `batch_size` 和 `gradient_accumulation_steps`，以最大化显存利用率并提高训练效率。

### 快速估算模式（推荐）

根据可用显存自动计算最佳 batch size：

```bash
# 自动计算，不指定目标（尽可能使用大的 batch_size）
uv run python train.py \
    --dim 768 \
    --n_layers 12 \
    --max_seq_len 512 \
    --auto_batch \
    --epochs 3

# 指定目标有效 batch size，自动分配最优组合
uv run python train.py \
    --dim 768 \
    --n_layers 12 \
    --max_seq_len 512 \
    --auto_batch \
    --target_batch_size 32 \
    --epochs 3
```

### 试错法模式（更准确）

实际运行测试找到最大 batch_size（更精确但启动较慢）：

```bash
uv run python train.py \
    --dim 768 \
    --n_layers 12 \
    --max_seq_len 512 \
    --auto_batch \
    --use_trial \
    --epochs 3
```

### 自动配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--auto_batch` | 启用自动 batch size 计算 | False |
| `--target_batch_size` | 目标有效 batch size（batch_size × gradient_accumulation_steps）| None |
| `--use_trial` | 使用试错法（实际运行测试，更准确但较慢）| False |

### 工作原理

1. **快速估算模式**：根据模型参数、序列长度和可用显存进行理论计算，快速得出结果
2. **试错法模式**：逐步增大 batch_size 实际运行测试，直到触发 OOM，返回安全值

### 示例输出

```
==================================================
Auto batch configuration enabled
==================================================
Estimated max batch_size for available memory: 12

Auto configuration result:
  batch_size: 12
  gradient_accumulation_steps: 1
  effective_batch_size: 12
==================================================
```

## Full Command Options

```bash
uv run python train.py \
    --dim 1024 \              # Hidden dimension
    --n_layers 16 \           # Number of transformer layers
    --n_heads 16 \            # Number of attention heads
    --n_kv_heads 4 \          # Number of KV heads (GQA ratio = 4:1)
    --max_seq_len 512 \       # Maximum sequence length
    --batch_size 2 \          # Batch size per device
    --gradient_accumulation_steps 8 \  # Effective batch = 2 * 8 = 16
    --epochs 3 \              # Number of training epochs
    --learning_rate 3e-4 \    # Peak learning rate
    --weight_decay 0.1 \      # Weight decay for regularization
    --warmup_steps 1000 \     # Learning rate warmup steps
    --max_grad_norm 1.0 \     # Gradient clipping
    --output_dir checkpoints \ # Checkpoint output directory
    --save_interval 1 \       # Save checkpoint every N epochs
    --eval_interval 1         # Evaluate every N epochs
```

## Resume Training

```bash
uv run python train.py \
    --checkpoint checkpoints/checkpoint_epoch_1.pt \
    --resume checkpoints/checkpoint_epoch_1.pt
```

## Generate Text

```bash
uv run python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Hello, how are you?" \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_k 50
```

## Memory Requirements

| Model | Params | VRAM (fp32) | VRAM (fp16/bf16) |
|-------|--------|-------------|------------------|
| tiny  | ~100M  | ~2GB        | ~1GB             |
| small | ~400M  | ~4GB        | ~2GB             |
| base  | ~800M  | ~8GB        | ~4GB             |
| large | ~1.5B  | ~16GB       | ~8GB             |

Note: PyTorch's SDPA automatically uses efficient attention implementations which reduce memory usage.

## Training Time Estimates

On a single RTX 4090 (24GB):
- small config: ~2-3 hours per epoch
- base config: ~4-6 hours per epoch

## Monitoring Training

Checkpoints are saved to `checkpoints/`:
- `best_model.pt`: Best model based on validation loss
- `checkpoint_epoch_N.pt`: Model at epoch N

Training logs show:
- Loss: Cross-entropy loss
- LR: Current learning rate
- Perplexity: exp(loss) on validation set
