"""One comparison table for a training step: dummygrad, optionally a baseline build of dummygrad,
and PyTorch both eager and torch.compile'd, run back to back on the same machine.

    python bench/compare.py                 fp32, every configuration
    python bench/compare.py --fp16          fp16 autocast (AutocastGuard + GradScaler / torch.autocast)
    python bench/compare.py --fp16 medium   one configuration
    python bench/compare.py --baseline ../dummygrad_baseline/build/dg_bench   also time another build
    python bench/compare.py --profile       also print dummygrad's per-op profile
    python bench/compare.py --no-compile    skip torch.compile (it adds about a minute per configuration)
    python bench/compare.py --seconds 5     time each configuration for at least 5 seconds (default 2)

Every run prints its own full table as it goes; the combined tables come last. Times are medians
of at least 50 steps (see bench.cpp). A ratio above 1 means dummygrad is faster.
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, label):
    print(f"\n### {label}\n$ {' '.join(cmd)}\n", flush=True)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        subprocess.run(cmd + ["--json", path], check=True)
        with open(path) as f:
            return json.load(f)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as e:
        print(f"{label} failed: {e}", flush=True)
        return None
    finally:
        os.remove(path)


def by_name(result):
    return {c["name"]: c for c in result["configs"]} if result else {}


def main():
    args = sys.argv[1:]

    def take_value(flag, default=None):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1]
            del args[i:i + 2]
            return value
        return default

    def take_flag(flag):
        if flag in args:
            args.remove(flag)
            return True
        return False

    dg = take_value("--dg", os.path.join(HERE, "..", "build", "dg_bench"))
    baseline = take_value("--baseline")
    seconds = take_value("--seconds")
    profile = take_flag("--profile")
    no_compile = take_flag("--no-compile")
    common = args + (["--seconds", seconds] if seconds else [])  # precision flag and config name

    torch_bench = [sys.executable, os.path.join(HERE, "bench_torch.py")]
    runs = [("dummygrad", run([dg] + common + (["--profile"] if profile else []), "dummygrad"))]
    if baseline:
        runs.append(("baseline", run([baseline] + common, f"dummygrad baseline ({baseline})")))
    runs.append(("PyTorch eager", run(torch_bench + common, "PyTorch eager")))
    if not no_compile:
        runs.append(("PyTorch compile", run(torch_bench + common + ["--compile"], "PyTorch compile")))

    results = {label: by_name(r) for label, r in runs}
    meta = {label: r for label, r in runs if r}
    dg_res = results["dummygrad"]
    if not dg_res:
        print("dummygrad did not produce results")
        return 1

    first = next(iter(meta.values()))
    precision, device = first["precision"], first["device"]
    versions = ", ".join(f"{label} {r['version']}" for label, r in meta.items())
    others = [label for label, _ in runs if label != "dummygrad"]
    print(f"\n## Training step, {precision}, {device}\n")
    print(f"{versions}. Median ms/step (10th-90th percentile); ratio = other / dummygrad, above 1 means dummygrad is faster.\n")
    print("| config | params | dummygrad | " + " | ".join(others) + " | " + " | ".join(f"vs {o}" for o in others) + " |")
    print("|---|---|---|" + "---|" * (2 * len(others)))

    def cell(c):
        return f"{c['ms']:.2f} ({c['p10']:.2f}-{c['p90']:.2f})" if c else "-"

    for name, c in dg_res.items():
        row = [name, f"{c['params_m']:.1f}M", cell(c)]
        row += [cell(results[o].get(name)) for o in others]
        row += [f"{results[o][name]['ms'] / c['ms']:.2f}x" if results[o].get(name) else "-" for o in others]
        print("| " + " | ".join(row) + " |")

    print("\nPeak GPU memory held by tensors over the timed steps (MB; dummygrad and PyTorch both count live "
          "allocations, rounded to 512 bytes):\n")
    labels = ["dummygrad"] + others
    print("| config | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for name in dg_res:
        row = [name]
        for label in labels:
            c = results[label].get(name)
            row.append(f"{c['peak_mb']:.0f}" if c and c.get("peak_mb") else "-")
        print("| " + " | ".join(row) + " |")

    if not no_compile and results.get("PyTorch compile"):
        compile_s = ", ".join(f"{n} {c['compile_s']:.0f} s" for n, c in results["PyTorch compile"].items())
        print(f"\ntorch.compile time, paid once per shape and not included above: {compile_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
