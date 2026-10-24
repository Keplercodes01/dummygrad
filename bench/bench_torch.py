"""PyTorch reference for bench/bench.cpp: the same ModernTransformer architecture and training
step (RMSNorm, interleaved RoPE, GQA, SwiGLU with biases, tied embeddings, AdamW), timed the
same way, so the two tables can be compared row by row.

    python bench/bench_torch.py            all configurations
    python bench/bench_torch.py small      one configuration
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = [
    # name, vocab, d_model, n_layers, n_heads, n_kv_heads, batch, seq_len
    ("tiny", 512, 128, 2, 4, 2, 8, 64),
    ("small", 4096, 256, 4, 8, 4, 16, 128),
    ("medium", 8192, 512, 6, 8, 8, 8, 256),
]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma


def rope_tables(head_dim, max_len, theta=10000.0):
    i = torch.arange(head_dim // 2, dtype=torch.float32)
    freq = theta ** (-2 * i / head_dim)
    angles = torch.arange(max_len, dtype=torch.float32)[:, None] * freq[None, :]
    return angles.cos(), angles.sin()  # [max_len, head_dim / 2]


def apply_rope(x, cos, sin):
    # x: [B, S, H, D]; rotates interleaved pairs (2j, 2j + 1), as dummygrad does
    s = x.shape[1]
    c, si = cos[:s][None, :, None, :], sin[:s][None, :, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * si
    out[..., 1::2] = x0 * si + x1 * c
    return out


class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.h, self.kv, self.dk = n_heads, n_kv_heads, d // n_heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, n_kv_heads * self.dk)
        self.wv = nn.Linear(d, n_kv_heads * self.dk)
        self.wo = nn.Linear(d, d)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        q = apply_rope(self.wq(x).view(b, s, self.h, self.dk), cos, sin).transpose(1, 2)
        k = apply_rope(self.wk(x).view(b, s, self.kv, self.dk), cos, sin).transpose(1, 2)
        v = self.wv(x).view(b, s, self.kv, self.dk).transpose(1, 2)
        if self.kv < self.h:
            k = k.repeat_interleave(self.h // self.kv, dim=1)
            v = v.repeat_interleave(self.h // self.kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, s, d))


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = int(8.0 * d / 3.0)
        self.w_gate, self.w_up, self.w_down = nn.Linear(d, hidden), nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.rms1, self.attn = RMSNorm(d), Attention(d, n_heads, n_kv_heads)
        self.rms2, self.ffn = RMSNorm(d), SwiGLU(d)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.rms1(x), cos, sin)
        return x + self.ffn(self.rms2(x))


class ModernTransformer(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, n_kv_heads, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, n_kv_heads) for _ in range(n_layers)])
        self.rms_f = RMSNorm(d)
        cos, sin = rope_tables(d // n_heads, max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.rms_f(x) @ self.emb.weight.t()  # tied output head


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True  # dummygrad enables TF32 tensor-core math too
    warmup, steps = 3, 10

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"PyTorch {torch.__version__} benchmark on {device.upper()} ({steps} timed steps after {warmup} warmup)\n")
    print("| config | params | ms/step | tokens/s | forward | backward | optimizer |")
    print("|---|---|---|---|---|---|---|")
    for name, vocab, d, layers, heads, kv, batch, seq in CONFIGS:
        if only and only != name:
            continue
        torch.manual_seed(0)
        model = ModernTransformer(vocab, d, layers, heads, kv, seq).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        idx = torch.arange(batch * seq)
        inputs = ((idx * 7) % vocab).view(batch, seq).to(device)
        targets = ((idx * 7 + 1) % vocab).view(batch, seq).to(device)

        t_fwd = t_bwd = t_opt = 0.0
        for s in range(warmup + steps):
            opt.zero_grad(set_to_none=True)
            sync()
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(inputs).view(-1, vocab), targets.view(-1))
            sync()
            t1 = time.perf_counter()
            loss.backward()
            sync()
            t2 = time.perf_counter()
            opt.step()
            sync()
            t3 = time.perf_counter()
            if s >= warmup:
                t_fwd += t1 - t0
                t_bwd += t2 - t1
                t_opt += t3 - t2
        per_step = (t_fwd + t_bwd + t_opt) / steps * 1000
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"| {name} | {params:.1f}M | {per_step:.1f} | {batch * seq * 1000 / per_step:.0f} | "
              f"{t_fwd / steps * 1000:.1f} | {t_bwd / steps * 1000:.1f} | {t_opt / steps * 1000:.1f} |", flush=True)


if __name__ == "__main__":
    main()
