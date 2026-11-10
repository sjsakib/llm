import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from pathlib import Path

with Path(__file__).with_name('input.txt').open(encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


device = torch.device("cpu")
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()


max_iters = 5000
eval_interval = max_iters // 50
eval_iters = 100

batch_size = 64
block_size = 256
n_embed = 384
n_layer = 6
n_head = 6
dropout = .2
lr = 3e-4
cpt_path = './nano-gpt/model.pth'

data = torch.tensor(encode(text), dtype=torch.long, device=device)

n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-block_size-1, (batch_size,), device=device)
    positions = ix[:, None] + torch.arange(block_size, device=device)
    x = data[positions]
    y = data[positions+1]

    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)

        _, T, C = x.shape
        wei = q @ k.transpose(-2, -1) * q.shape[-1] ** -.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.droput = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(self.droput(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)], nn.LayerNorm(n_embed),
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
            print(decode(idx_next[0].tolist()), end='', flush=True)
        return idx


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == '__main__':

    model = GPT()

    print(
        f'number of paramters: {sum([p.nelement() for p in model.parameters()])}')

    def save_model(iter, losses):
        checkpoint = {
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'iter': iter,
            'losses': losses
        }
        torch.save(checkpoint, cpt_path)

    start_iter = 0
    try:
        cpt = torch.load(cpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(cpt['model_state'])
        start_iter = cpt['iter'] + 1
        losses = cpt['losses']
        print(
            f"resuming from iter: {start_iter}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    except Exception as e:
        print('No saved model found', e)
        cpt = None

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if cpt:
        optimizer.load_state_dict(cpt['optim_state'])

    model.forward = torch.compile(model.forward)
    print('optimization done')

    model.train()
    t = time.perf_counter()
    for iter in range(start_iter, max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            save_model(iter, losses)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {(time.perf_counter() - t)/eval_interval:.2f}s/iter")
            t = time.perf_counter()

    print('starting eval\n\n')
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    out = model.generate(idx, 10000)
