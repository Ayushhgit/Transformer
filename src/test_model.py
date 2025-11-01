import torch
from transformer_model import DecoderOnlyTransformer
from dataset import load_dataset

# Load dataset (to get vocab and encoding logic)
text, stoi, itos = load_dataset()

# Load model
vocab_size = len(stoi)
model = DecoderOnlyTransformer(vocab_size)
model.load_state_dict(torch.load("transformer_trained.pth", map_location=torch.device("cpu")))
model.eval()

# Function to generate text
def generate_text(model, start_text="ROMEO:", length=500):
    model.eval()
    context = torch.tensor([stoi.get(c, 0) for c in start_text], dtype=torch.long).unsqueeze(0)
    for _ in range(length):
        with torch.no_grad():
            logits = model(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
    output = ''.join([itos[int(i)] for i in context[0].tolist()])
    return output

# Test the model
generated = generate_text(model, start_text="ROMEO:", length=500)
print("\n🧾 Generated Text:\n")
print(generated)
