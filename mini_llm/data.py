"""
Data loading and preprocessing for Daily Dialog dataset.
"""

from typing import Iterator, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class DailyDialogDataset(Dataset):
    """Daily Dialog dataset for language modeling."""
    
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
        print(f"Loading Daily Dialog dataset ({split} split)...")
        self.dataset = load_dataset("DeepPavlov/daily_dialog", split=split)
        
        # Process dialogues into text
        self.examples = self._process_dataset()
        print(f"Loaded {len(self.examples)} examples")
    
    def _process_dataset(self) -> List[str]:
        """Process dataset into text examples."""
        examples = []
        for item in self.dataset:
            # Each dialogue is a list of utterances
            dialogue = item["dialog"]
            # Join utterances with newlines
            text = "\n".join(dialogue)
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
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        print(f"Loading Daily Dialog dataset ({split} split)...")
        dataset = load_dataset("DeepPavlov/daily_dialog", split=split)
        
        # Tokenize all dialogues
        all_tokens = []
        for item in dataset:
            dialogue = item["dialog"]
            text = self.tokenizer.eos_token + "\n".join(dialogue) + self.tokenizer.eos_token
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        # Pack into sequences of max_length
        self.sequences = []
        for i in range(0, len(all_tokens), max_length):
            seq = all_tokens[i:i + max_length + 1]  # +1 for target
            if len(seq) > 1:  # Need at least 2 tokens for input and target
                # Pad if necessary
                if len(seq) < max_length + 1:
                    seq = seq + [self.tokenizer.pad_token_id] * (max_length + 1 - len(seq))
                self.sequences.append(torch.tensor(seq, dtype=torch.long))
        
        print(f"Created {len(self.sequences)} sequences of length {max_length}")
    
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
) -> DataLoader:
    """Create a DataLoader for the Daily Dialog dataset."""
    dataset = ConversationDataset(
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
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
    
    # Create dataloader
    dataloader = create_dataloader(
        split="train",
        tokenizer_name="gpt2",
        max_length=128,
        batch_size=2,
    )
    
    # Test iteration
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Sample input tokens: {inputs[0][:20].tolist()}")
        print(f"  Sample target tokens: {targets[0][:20].tolist()}")
        if i >= 2:
            break
