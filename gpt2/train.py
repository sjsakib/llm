
from gpt import GPTConfig, GPT
from nanogpt.gpt import decode, get_batch, vocab_size, device
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    cfg = GPTConfig(vocab_size=vocab_size)
    model = GPT(cfg)

    model.to(device)

    optim = torch.optim.AdamW(model.parameters()) 

    for _ in range(100):
        xb, yb = get_batch('train') # yb: B, block_size


        logits = model(xb)
        logits = logits.view(-1, logits.shape[-1])

        loss = F.cross_entropy(logits, yb.view(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss.item())


    idx = [[0]*cfg.block_size]
    for _ in range(100):
        logits = model(torch.tensor(idx), device=device)[0, -1, :]
        logits = F.softmax(logits, dim=-1)

        c = torch.multinomial(logits, num_samples=1)

        idx[0] = idx[0][1:] + [c.item()]

        print(decode([c.item()]), end='', flush=True)