"""
Data loading and preprocessing for Baike dataset.
"""

from typing import Iterator, List, Optional
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class BaikeDataset(Dataset):
    """Baike dataset for language modeling."""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
    ):
        super().__init__()
        self.split = split
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        print(f"Loading Baike dataset ({split} split)...")
        self.dataset = load_dataset("Dialogue-Model-Research-Group/baike", split=split)
        
        # Process articles into text
        self.examples = self._process_dataset()
        print(f"Loaded {len(self.examples)} examples")
    
    def _process_dataset(self) -> List[str]:
        """Process dataset into text examples."""
        examples = []
        for item in self.dataset:
            # For Baike dataset with context and response fields
            if "context" in item and "response" in item:
                # Combine context and response as Q&A pair
                context = item["context"]
                response = item["response"]
                if context and response:
                    text = f"问：{context}\n答：{response}"
                    examples.append(text)
                    continue
            
            # Try common field names for text content
            text = None
            for key in ["text", "content", "article", "passage", "document", "body"]:
                if key in item and item[key]:
                    text = item[key]
                    break
            
            # If no recognized field, use the first string field
            if text is None:
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        text = value
                        break
            
            if text and isinstance(text, str):
                examples.append(text)
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.examples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
        )
        
        return torch.tensor(tokens, dtype=torch.long)


class ConversationDataset(Dataset):
    """
    Dataset that packs multiple conversations into fixed-length sequences.
    More efficient for training than padding individual examples.
    Includes caching mechanism to avoid reprocessing data.
    """
    
    # Cache directory
    CACHE_DIR = Path(__file__).parent.parent / "cache"
    
    @staticmethod
    def _get_cache_path(
        split: str,
        tokenizer_name: str,
        max_length: int,
        max_examples: Optional[int] = None,
    ) -> Path:
        """Generate cache file path based on parameters."""
        cache_key = f"{split}_{tokenizer_name.replace('/', '-')}_{max_length}"
        if max_examples is not None:
            cache_key += f"_{max_examples}"
        cache_file = f"dataset_{cache_key}.pt"
        return ConversationDataset.CACHE_DIR / cache_file
    
    @classmethod
    def clear_cache(cls, split: Optional[str] = None) -> None:
        """
        Clear cached datasets.
        
        Args:
            split: If specified, only clear cache for that split. If None, clear all cache.
        """
        if not cls.CACHE_DIR.exists():
            print("Cache directory does not exist")
            return
        
        if split is None:
            # Clear all cache
            import shutil
            shutil.rmtree(cls.CACHE_DIR)
            print(f"Cleared all cache in {cls.CACHE_DIR}")
        else:
            # Clear specific split caches
            for cache_file in cls.CACHE_DIR.glob(f"dataset_{split}_*.pt"):
                cache_file.unlink()
                print(f"Removed cache: {cache_file.name}")
    
    @classmethod
    def get_cache_size(cls) -> dict:
        """Get the size of cached datasets."""
        if not cls.CACHE_DIR.exists():
            return {}
        
        cache_info = {}
        for cache_file in cls.CACHE_DIR.glob("dataset_*.pt"):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            cache_info[cache_file.name] = f"{size_mb:.2f} MB"
        return cache_info
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        max_examples: Optional[int] = None,
    ):
        super().__init__()
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check cache first
        cache_path = self._get_cache_path(split, tokenizer_name, max_length, max_examples)
        if cache_path.exists():
            print(f"Loading cached dataset from {cache_path.name}...", flush=True)
            cache_data = torch.load(cache_path, weights_only=False)
            self.sequences = cache_data
            print(f"Loaded {len(self.sequences)} cached sequences", flush=True)
            return
        
        # Create cache directory if needed
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print(f"Loading Baike dataset ({split} split)...", flush=True)
        
        # Load dataset - handle case where dataset only has 'train' split
        try:
            dataset = load_dataset("Dialogue-Model-Research-Group/baike", split=split, streaming=False)
        except ValueError:
            # If requested split doesn't exist, use 'train' split and slice it
            if split == "validation":
                print(f"  '{split}' split not found, using last 5% of 'train' split as validation set", flush=True)
                dataset = load_dataset("Dialogue-Model-Research-Group/baike", split="train", streaming=False)
                # Use last 5% for validation
                total_len = len(dataset)
                val_start = int(total_len * 0.95)
                dataset = dataset.select(range(val_start, total_len))
            elif split == "train":
                print(f"  Using first 95% of 'train' split as training set", flush=True)
                dataset = load_dataset("Dialogue-Model-Research-Group/baike", split="train", streaming=False)
                total_len = len(dataset)
                train_end = int(total_len * 0.95)
                dataset = dataset.select(range(0, train_end))
            else:
                raise
        
        print(f"Dataset size: {len(dataset)}", flush=True)
        
        # Limit examples if specified
        if max_examples is not None and len(dataset) > max_examples:
            print(f"Limiting to first {max_examples} examples", flush=True)
            dataset = dataset.select(range(max_examples))
        
        # Tokenize all articles with progress bar
        print(f"Tokenizing {len(dataset)} examples...", flush=True)
        all_tokens = []
        
        # Process in batches for better progress reporting
        batch_size = 1000
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            print(f"  Processing examples {batch_start}-{batch_end}/{len(dataset)}...", flush=True)
            
            for idx in range(batch_start, batch_end):
                item = dataset[idx]
                
                # For Baike dataset with context and response fields
                text_content = None
                if "context" in item and "response" in item:
                    # Combine context and response as Q&A pair
                    context = item["context"]
                    response = item["response"]
                    if context and response:
                        text_content = f"问：{context}\n答：{response}"
                
                # Try common field names for text content
                if text_content is None:
                    for key in ["text", "content", "article", "passage", "document", "body"]:
                        if key in item and item[key]:
                            text_content = item[key]
                            break
                
                # If no recognized field, use the first string field
                if text_content is None:
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 10:
                            text_content = value
                            break
                
                if text_content and isinstance(text_content, str):
                    # Add BOS token at start and EOS token at end
                    text = self.tokenizer.bos_token + text_content + self.tokenizer.eos_token
                    tokens = self.tokenizer.encode(
                        text, 
                        add_special_tokens=False,
                        truncation=True,
                        max_length=2048  # Truncate very long texts
                    )
                    all_tokens.extend(tokens)
        
        # Pack into sequences of max_length
        self.sequences = []
        for i in range(0, len(all_tokens) - max_length, max_length):
            # Only take complete sequences (no padding)
            seq = all_tokens[i:i + max_length + 1]  # +1 for target
            if len(seq) == max_length + 1:  # Only use complete sequences
                self.sequences.append(torch.tensor(seq, dtype=torch.long))
        
        print(f"Created {len(self.sequences)} sequences of length {max_length}")
        
        # Save cache for future use
        print(f"Saving cache to {cache_path.name}...", flush=True)
        torch.save(self.sequences, cache_path)
        print(f"Cache saved successfully", flush=True)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple:
        seq = self.sequences[idx]
        # Input is all but last token, target is all but first token
        return seq[:-1], seq[1:]


def collate_fn(batch: List[tuple]) -> tuple:
    """Collate function for DataLoader."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets


def create_dataloader(
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    max_examples: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for the Baike dataset."""
    dataset = ConversationDataset(
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        max_examples=max_examples,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_vocab_size(tokenizer_name: str = "gpt2") -> int:
    """Get vocabulary size for a tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return len(tokenizer)


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    # Get vocab size
    vocab_size = get_vocab_size("gpt2")
    print(f"Vocabulary size: {vocab_size}")
    
    # Example: Create dataloader (will use cache if available)
    print("\n--- First run (will process data) ---")
    dataloader = create_dataloader(
        split="train",
        tokenizer_name="gpt2",
        max_length=128,
        batch_size=2,
    )
    
    # Example: Check cache size
    print("\n--- Cache info ---")
    cache_info = ConversationDataset.get_cache_size()
    if cache_info:
        print("Cached datasets:")
        for name, size in cache_info.items():
            print(f"  {name}: {size}")
    else:
        print("No cached datasets found")
    
    # Test iteration
    print("\n--- Data sample ---")
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Sample input tokens: {inputs[0][:20].tolist()}")
        print(f"  Sample target tokens: {targets[0][:20].tolist()}")
        if i >= 2:
            break
    
    # Example: How to clear cache if needed
    # ConversationDataset.clear_cache(split="train")  # Clear only train cache
    # ConversationDataset.clear_cache()  # Clear all cache
