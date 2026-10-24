#!/usr/bin/env python3
"""skew.py DIR: how skewed is Qwen3.6 35B-A3B's expert routing? Reads routing dumps made with a debug hook in the CUDA
backend that writes the ids tensor of every MUL_MAT_ID, in graph order, to DIR/ids_NNN.bin (6 int32 header
[n_used, T, n_expert, ...], then T x n_used int32 expert ids). A layer's gate, up and down products share one routing,
so file 3*l is layer l. Reports per-layer top-K coverage, the coverage of the best global allocation of B expert slots
(whole layers vs hottest experts across layers), and how well a hot set picked on the first half of the tokens covers
the second half."""
import sys, struct, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "."
layers = []
for l in range(40):
    b = open(f"{D}/ids_{3 * l:03d}.bin", "rb").read()
    nu, T, ne = struct.unpack("<3i", b[:12])
    layers.append(np.frombuffer(b[24:], dtype=np.int32).reshape(T, nu))
L, (T, NU), NE = len(layers), layers[0].shape, 256
print(f"{L} layers, {T} tokens, {NU} of {NE} experts per token")

def counts(ids):
    return np.bincount(ids.ravel(), minlength=NE)

# per-layer coverage of the top-K experts
print("top-K coverage of a layer's selections (median / min / max over layers):")
for K in (8, 16, 32, 54, 64, 96, 128):
    cov = [np.sort(counts(x))[::-1][:K].sum() / x.size for x in layers]
    print(f"  K={K:3d} ({100 * K / NE:4.1f}% of experts): {100 * np.median(cov):5.1f}%  {100 * min(cov):5.1f}%  {100 * max(cov):5.1f}%")

# global budget: B expert slots in VRAM, whole layers vs hottest (layer, expert) pairs
c = np.stack([counts(x) for x in layers])  # L x NE
total = c.sum()
for B in (8 * NE, int(8.5 * NE), 12 * NE):
    whole = np.sort(c.sum(1))[::-1][:B // NE].sum() / total
    hot = np.sort(c.ravel())[::-1][:B].sum() / total
    print(f"budget {B} slots ({B / NE:.1f} layers' worth): whole layers cover {100 * whole:.1f}% of expert reads, "
          f"hottest experts {100 * hot:.1f}%")

# stability: pick on the first half, measure on the second
h = T // 2
c1 = np.stack([counts(x[:h]) for x in layers]); c2 = np.stack([counts(x[h:]) for x in layers])
for B in (8 * NE, int(8.5 * NE)):
    pick = np.argsort(c1.ravel())[::-1][:B]
    print(f"budget {B}: hot set from tokens 0-{h - 1} covers {100 * c2.ravel()[pick].sum() / c2.sum():.1f}% of tokens {h}-{T - 1} "
          f"(oracle on the second half {100 * np.sort(c2.ravel())[::-1][:B].sum() / c2.sum():.1f}%)")
# how many experts a layer never uses in this text
print("experts never selected per layer (median):", int(np.median([(counts(x) == 0).sum() for x in layers])))
