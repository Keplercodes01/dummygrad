"""PyTorch reference for bench/bench.cpp: the same ModernTransformer architecture and training
step (RMSNorm, interleaved RoPE, GQA, SwiGLU with biases, tied embeddings, AdamW), timed the
same way, so the two tables can be compared row by row.

    python bench/bench_torch.py            all configurations
    python bench/bench_torch.py small      one configuration
    python bench/bench_torch.py --fp16     torch.autocast(float16) with GradScaler (dg_bench --fp16)
    python bench/bench_torch.py --bf16     torch.autocast(bfloat16) (dg_bench --bf16)
    python bench/bench_torch.py --seconds 5   time each configuration for at least 5 seconds (default 2)
    python bench/bench_torch.py --compile     torch.compile the forward + loss (Inductor, default mode)
    python bench/bench_torch.py --compile=reduce-overhead   ... with CUDA graphs
    python bench/bench_torch.py --json out.json   also write the results as JSON (bench/compare.py)

Like dg_bench, each configuration runs 10 warmup steps, then at least 50 timed steps covering at
least 2 seconds, and reports the median step with its 10th-90th percentile spread. Compilation
happens in the warmup steps; its time is reported separately. PyTorch runs with its fastest
settings: fused AdamW, F.rms_norm, scaled_dot_product_attention, TF32 matmuls.

PyTorch's autocast also runs elementwise and normalization ops in half precision, while
dummygrad's covers matmuls and attention, so the --fp16 tables compare the two recipes.
"""

import json
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
    ("large", 32000, 768, 12, 12, 4, 8, 512),
]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.gamma, self.eps)


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


def quantile(values, q):
    """q-quantile (0 <= q <= 1), interpolating between neighbouring order statistics, as dg_bench does."""
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def main():
    args = sys.argv[1:]
    min_seconds = 2.0
    if "--seconds" in args:
        i = args.index("--seconds")
        min_seconds = float(args[i + 1])
        del args[i:i + 2]
    json_path = None
    if "--json" in args:
        i = args.index("--json")
        json_path = args[i + 1]
        del args[i:i + 2]
    compile_mode = None
    for a in list(args):
        if a == "--compile" or a.startswith("--compile="):
            compile_mode = a.split("=", 1)[1] if "=" in a else "default"
            args.remove(a)
    amp = {"--fp16": torch.float16, "--bf16": torch.bfloat16}
    amp_dtype = next((amp[a] for a in args if a in amp), None)
    only = next((a for a in args if a not in amp), None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {torch.float16: "fp16", torch.bfloat16: "bf16", None: "fp32"}[amp_dtype]
    torch.backends.cuda.matmul.allow_tf32 = True  # dummygrad enables TF32 tensor-core math too
    warmup, min_steps, max_steps = 10, 50, 5000  # same as dg_bench

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    variant = "eager" if compile_mode is None else f"compile ({compile_mode})"
    device_name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"PyTorch {torch.__version__} {variant} benchmark on {device_name}, {precision} autocast")
    print(f"median of >= {min_steps} steps and >= {min_seconds:.0f} s per config after {warmup} warmup; "
          f"p10-p90 is the spread of step times\n")
    print("| config | params | ms/step | p10-p90 | tokens/s | forward | backward | optimizer | steps | peak MB | compile s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    results = []
    for name, vocab, d, layers, heads, kv, batch, seq in CONFIGS:
        if only and only != name:
            continue
        if device == "cuda":
            torch.cuda.empty_cache()
        torch._dynamo.reset()  # drop the previous configuration's compiled graphs and their memory
        torch.manual_seed(0)
        model = ModernTransformer(vocab, d, layers, heads, kv, seq).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                                fused=device == "cuda")

        def forward_loss(inputs, targets):
            return F.cross_entropy(model(inputs).view(-1, vocab), targets.view(-1))

        if compile_mode is not None:
            forward_loss = torch.compile(forward_loss, mode=None if compile_mode == "default" else compile_mode)
        scaler = torch.amp.GradScaler(device, enabled=amp_dtype == torch.float16)
        idx = torch.arange(batch * seq)
        inputs = ((idx * 7) % vocab).view(batch, seq).to(device)
        targets = ((idx * 7 + 1) % vocab).view(batch, seq).to(device)

        t_fwd, t_bwd, t_opt, t_step = [], [], [], []
        warmup_s = 0.0
        for s in range(warmup + max_steps):
            if s >= warmup and len(t_step) >= min_steps and sum(t_step) >= min_seconds:
                break
            if s == warmup and device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            sync()
            t0 = time.perf_counter()
            with torch.autocast(device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                loss = forward_loss(inputs, targets)
            sync()
            t1 = time.perf_counter()
            scaler.scale(loss).backward()
            sync()
            t2 = time.perf_counter()
            scaler.step(opt)
            scaler.update()
            sync()
            t3 = time.perf_counter()
            if s >= warmup:
                t_fwd.append(t1 - t0)
                t_bwd.append(t2 - t1)
                t_opt.append(t3 - t2)
                t_step.append(t3 - t0)
            else:
                warmup_s += t3 - t0
        ms = lambda v, q=0.5: quantile(v, q) * 1000
        per_step = ms(t_step)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
        # Warmup time beyond 10 ordinary steps: what compiling (and CUDA graph capture) cost
        compile_s = max(0.0, warmup_s - warmup * per_step / 1000) if compile_mode is not None else 0.0
        print(f"| {name} | {params:.1f}M | {per_step:.2f} | {ms(t_step, 0.1):.2f}-{ms(t_step, 0.9):.2f} | "
              f"{batch * seq * 1000 / per_step:.0f} | {ms(t_fwd):.2f} | {ms(t_bwd):.2f} | {ms(t_opt):.2f} | "
              f"{len(t_step)} | {peak_mb:.0f} | {compile_s:.0f} |", flush=True)
        results.append({"name": name, "params_m": params, "ms": per_step, "p10": ms(t_step, 0.1),
                        "p90": ms(t_step, 0.9), "forward": ms(t_fwd), "backward": ms(t_bwd), "optimizer": ms(t_opt),
                        "steps": len(t_step), "tokens_per_s": batch * seq * 1000 / per_step, "peak_mb": peak_mb,
                        "compile_s": compile_s})
        del model, opt, scaler, forward_loss, loss
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"framework": "PyTorch", "version": torch.__version__, "variant": variant,
                       "device": device_name, "precision": precision, "configs": results}, f)


if __name__ == "__main__":
    main()
