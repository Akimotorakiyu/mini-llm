"""
Generate text using trained Mini LLM.
"""

import argparse

import torch
from transformers import AutoTokenizer

from mini_llm.model import MiniLLM


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mini LLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set BOS token if available
    if tokenizer.bos_token_id is None and hasattr(tokenizer, 'eos_token_id'):
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer info:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    print(f"Model config:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden dim: {config.dim}")
    print(f"  Num layers: {config.n_layers}")
    print(f"  Num heads: {config.n_heads}")
    print(f"  Num KV heads: {config.n_kv_heads}")
    
    # Create and load model
    model = MiniLLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {prompt_tokens[0].tolist()}")
    
    # Generate
    print(f"\nGenerating (temperature={args.temperature}, top_k={args.top_k})...")
    print("(This may take a while...)")
    with torch.no_grad():
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    
    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated text:")
    print("=" * 50)
    print(output_text)
    print("=" * 50)
    
    # Also show token-by-token for debugging
    if len(generated[0]) < 50:
        print(f"\nGenerated tokens: {generated[0].tolist()}")


if __name__ == "__main__":
    main()
