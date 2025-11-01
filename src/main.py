import os
from .utils import set_seed, DEVICE
from .transformer_model import DecoderOnlyTransformer
from .dataset import get_loader
from .train import train_model

if __name__ == "__main__":
    set_seed(42)
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "../Dataset/tiny_shakespeare.txt")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded dataset: {len(text)} characters")

    # DataLoader
    seq_len = 128
    batch_size = 32
    dataloader, dataset = get_loader(text, seq_len=seq_len, batch_size=batch_size)
    
    # Model setup
    model = DecoderOnlyTransformer(
        vocab_size=len(dataset.chars),
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_seq_len=seq_len
    )

    # Train
    train_model(model, dataloader, epochs=10, lr=3e-4)
    print("✅ Training complete.")
