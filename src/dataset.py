import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    """
    Character-level dataset for training a language model.
    Converts text into integer sequences (using char-level vocab),
    and provides (input, target) pairs for next-character prediction.
    """
    def __init__(self, text: str, seq_len: int = 128):
        # Build vocabulary (sorted unique characters)
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Encode entire text as integer IDs
        self.data = [self.stoi[c] for c in text]
        self.seq_len = seq_len

    def __len__(self):
        # number of possible sequences (without overflow)
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns:
          x: input sequence of length seq_len
          y: target sequence (shifted by one character)
        """
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def get_loader(text: str, seq_len: int = 128, batch_size: int = 32, num_workers: int = 0):
    """
    Creates DataLoader for the CharDataset.
    """
    dataset = CharDataset(text, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset
