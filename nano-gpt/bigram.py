import torch.nn as nn
import torch.nn.functional as F
import torch

with open('./nano-gpt/input.txt') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


device = torch.device(torch.accelerator.current_accelerator()
                      if torch.accelerator.is_available() else "cpu")


data = torch.tensor(encode(text), dtype=torch.long, device=device)

n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

max_iters = 10000
eval_interval = 1000
eval_iters = 200

batch_size = 32
block_size = 8
n_embed = 32


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-block_size-1, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)

        _, T, C = x.shape
        wei = q @ k.transpose(-2, -1) * C ** -.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa(x)
        return self.ffwd(x)


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, y=None):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, 1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = Bigram().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for i in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(idx, 200)
print(decode(out[0].tolist()))
