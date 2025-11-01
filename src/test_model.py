import torch
import torch.nn.functional as F
from .transformer_model import DecoderOnlyTransformer
from .dataset import get_loader
from .utils import DEVICE

# === Load dataset file ===
DATA_PATH = "/content/Transformer/Dataset/tiny_shakespeare.txt"

# Load the text data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Create dataset and model (same settings as during training)
_, ds = get_loader(text, seq_len=64, batch_size=1)

vocab_size = len(ds.chars)
model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=128,
    num_layers=2,
    num_heads=4,
    d_ff=512,
    max_seq_len=64
).to(DEVICE)

# === Load trained weights ===
# Make sure 'model.pth' exists in your repo root (after training)
checkpoint_path = "/content/Transformer/model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

# === Text generation ===
def generate(model, start_text="To be", length=200, temperature=1.0):
    model.eval()
    input_ids = torch.tensor([ds.stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(length):
        logits = model(input_ids)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    output_text = ''.join([ds.itos[i.item()] for i in input_ids[0]])
    return output_text

# === Run a generation test ===
if __name__ == "__main__":
    print("Generating text...")
    result = generate(model, start_text="To be or not to be", length=300, temperature=0.8)
    print("\nGenerated Text:\n")
    print(result)
