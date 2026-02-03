"""
Mini LLM Model with GQA and SPDA.

Architecture:
- RMSNorm (using torch.nn.RMSNorm)
- RoPE (Rotary Position Embedding)
- GQA (Grouped Query Attention with F.scaled_dot_product_attention)
- SwiGLU FFN
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniLLMConfig:
    """Configuration for MiniLLM."""
    vocab_size: int = 32000
    dim: int = 1024          # Hidden dimension
    n_layers: int = 16       # Number of transformer layers
    n_heads: int = 16        # Number of attention heads (for queries)
    n_kv_heads: int = 4      # Number of key/value heads (GQA)
    max_seq_len: int = 2048  # Maximum sequence length
    ffn_dim_multiplier: float = 4.0  # FFN hidden dim multiplier
    multiple_of: int = 256   # Make FFN dim multiple of this
    dropout: float = 0.0
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0  # RoPE base frequency

    def __post_init__(self):
        # Ensure n_heads is divisible by n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequency tensors.
    
    Returns:
        cos, sin tensors of shape (max_seq_len, dim // 2)
    """
    # Compute frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Create position indices
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    
    # Outer product: (max_seq_len, dim // 2)
    freqs = torch.outer(t, freqs)
    
    # Complex numbers in polar form: cos + i*sin
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin


def apply_rotary_embedding(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embedding to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
        freqs_cos, freqs_sin: Precomputed frequency tensors of shape (seq_len, head_dim // 2)
    
    Returns:
        Rotated tensor of same shape as x
    """
    # Split x into even and odd indices
    x_even = x[..., 0::2]  # (batch, seq_len, n_heads, head_dim // 2)
    x_odd = x[..., 1::2]   # (batch, seq_len, n_heads, head_dim // 2)
    
    # Expand freqs to match batch and n_heads dimensions
    # freqs_cos/freqs_sin: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation
    # x' = [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
    x_rotated_even = x_even * freqs_cos - x_odd * freqs_sin
    x_rotated_odd = x_even * freqs_sin + x_odd * freqs_cos
    
    # Interleave back
    x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_out = x_out.flatten(-2)  # (batch, seq_len, n_heads, head_dim)
    
    return x_out


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention using PyTorch's SDPA."""
    
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # Repetition factor
        
        # Q, K, V projections
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.dropout = config.dropout
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            freqs_cos, freqs_sin: Precomputed RoPE frequencies
            mask: Attention mask (optional)
        
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        xq = self.wq(x)  # (batch, seq_len, n_heads * head_dim)
        xk = self.wk(x)  # (batch, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)  # (batch, seq_len, n_kv_heads * head_dim)
        
        # Reshape to (batch, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        xq = apply_rotary_embedding(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_embedding(xk, freqs_cos, freqs_sin)
        
        # Repeat K and V for GQA: (batch, seq_len, n_kv_heads, head_dim) -> (batch, seq_len, n_heads, head_dim)
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose for SDPA: (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Use PyTorch's SDPA (efficient attention implementation)
        # This automatically uses Flash Attention when available
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,  # Use causal mask if no custom mask provided
        )
        
        # Reshape back: (batch, seq_len, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.wo(output)
        
        return output


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        
        # Calculate hidden dim (similar to LLaMA)
        hidden_dim = int(config.ffn_dim_multiplier * config.dim)
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU uses 2/3
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # Down projection
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # Up projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: swish(x @ w1) * (x @ w3) @ w2"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        
        # Use PyTorch's native RMSNorm
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-normalization + Attention with residual
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        
        # Pre-normalization + FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


class MiniLLM(nn.Module):
    """Mini LLM with GQA and SPDA."""
    
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output normalization
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        
        # Output projection (shared with input embedding for efficiency)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Share weights between input embedding and output projection
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies (buffer so they move with model.to(device))
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            tokens: Input token IDs (batch, seq_len)
            targets: Target token IDs for training (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        batch_size, seq_len = tokens.shape
        
        # Token embedding
        h = self.tok_embeddings(tokens)
        
        # Get RoPE frequencies for this sequence length
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        
        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        
        # Final normalization
        h = self.norm(h)
        
        # Output projection
        logits = self.output(h)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens using the model.
        
        Args:
            tokens: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
        
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            tokens_cond = tokens if tokens.size(1) <= self.config.max_seq_len else tokens[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(tokens_cond)
            
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_memory(self) -> dict:
        """Estimate memory usage."""
        param_bytes = self.count_parameters() * 4  # Assuming float32
        param_mb = param_bytes / (1024 ** 2)
        param_gb = param_bytes / (1024 ** 3)
        
        return {
            "parameters": self.count_parameters(),
            "memory_mb": param_mb,
            "memory_gb": param_gb,
        }


def create_mini_llm(
    vocab_size: int = 32000,
    dim: int = 1024,
    n_layers: int = 16,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    max_seq_len: int = 2048,
) -> MiniLLM:
    """
    Create a MiniLLM model with specified configuration.
    
    Default config is ~0.4B parameters.
    """
    config = MiniLLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
    )
    return MiniLLM(config)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_mini_llm(
        vocab_size=10000,
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
    ).to(device)
    
    # Print model info
    info = model.estimate_memory()
    print(f"Parameters: {info['parameters']:,}")
    print(f"Estimated memory: {info['memory_mb']:.2f} MB")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    
    logits, loss = model(tokens, tokens)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'None'}")
    
    # Test generation
    prompt = torch.randint(0, 10000, (1, 10), device=device)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
