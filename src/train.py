import torch
import torch.nn as nn
from tqdm import tqdm
from .utils import DEVICE

def causal_mask(seq_len: int):
    """
    Creates a lower-triangular causal attention mask (for autoregressive decoding).
    Prevents a token from attending to future tokens.
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=DEVICE))
    return mask  # shape: (seq_len, seq_len)


def train_model(model, dataloader, epochs: int = 10, lr: float = 3e-4, save_path: str = "transformer.pth"):
    """
    Full training loop for the Transformer model.

    Args:
        model: Transformer model (nn.Module)
        dataloader: DataLoader from dataset.py
        epochs: number of epochs to train
        lr: learning rate
        save_path: file path to save model checkpoint
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Training on device: {DEVICE}")
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for xb, yb in progress_bar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mask = causal_mask(xb.size(1)).unsqueeze(0)  # (1, seq_len, seq_len)

            # Forward pass
            logits = model(xb, mask)  # (batch, seq_len, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} ✅ Avg Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model checkpoint saved to: {save_path}")

    print("🎉 Training complete!")
