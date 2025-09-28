# train.py
import os, torch, torch.nn as nn
import matplotlib.pyplot as plt
from tokenizer import Tokenizer

def main():
    os.makedirs("results", exist_ok=True)
    data = open("sample_data.txt", "r", encoding="utf-8").read()

    tok = Tokenizer()
    tok.build_vocab(data)
    enc = torch.tensor(tok.encode(data), dtype=torch.long)
    X, y = enc[:-1], enc[1:]

    class TinyLM(nn.Module):
        def __init__(self, vocab_size, hidden=64):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden)
            self.l1 = nn.Linear(hidden, vocab_size)
        def forward(self, x):
            return self.l1(self.emb(x))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyLM(len(tok.vocab)).to(device)
    X, y = X.to(device), y.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(10):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")

    plt.plot(losses)
    plt.savefig("results/loss_curve.png")
    print("Final loss:", losses[-1])

if __name__ == "__main__":
    main()
