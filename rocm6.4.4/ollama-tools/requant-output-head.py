#!/usr/bin/env python3
"""Re-encode only a GGUF model's output tensor (output.weight), copying everything else unchanged.

    requant-output-head.py IN.gguf OUT.gguf [q8_0]

Token generation reads every weight once per token, and some GGUF uploads keep
the output tensor in bf16 or f16. With a large vocabulary that one tensor can be
a third of the model: a Qwen3.5 9B IQ4_XS upload with a 248k-token vocabulary
carries a 2.03 GB bf16 output tensor in a 6.4 GB file. Converting just that
tensor to q8_0 halves it. Measured on an RX 590: 0.95 GB less VRAM (about 40k
more tokens of context for that model), generation +11% (28.1 -> 31.2 tok/s),
and against the bf16 original the same top token 98.0% of the time, mean KL
divergence 0.0014 (llama-perplexity --kl-divergence over 4k tokens of text).

llama-quantize cannot do this: given a target type it requantizes the other
tensors too, and requantizing already-quantized tensors loses quality. This
script copies every other tensor's bytes and all metadata as they are.

Needs numpy and llama.cpp's gguf package (pip install gguf, or GGUF_PY pointing
at llama.cpp's gguf-py directory). Reads the whole output tensor into memory
(about 4 GB of float32 for the example above).
"""
import os
import sys

import numpy as np

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf
from gguf import quants


def to_f32(t):
    shape = tuple(int(d) for d in reversed(t.shape))  # GGUF lists dims innermost first
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        # the reader hands bf16 over as raw bytes
        raw = np.ascontiguousarray(t.data).view(np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        return np.ascontiguousarray(t.data).astype(np.float32).reshape(shape)
    raise SystemExit(f"output.weight is {t.tensor_type.name}; only bf16, f16 or f32 is converted")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    qtype = gguf.GGMLQuantizationType[(sys.argv[3] if len(sys.argv) > 3 else "q8_0").upper()]
    reader = gguf.GGUFReader(src, "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    converted = {}
    for t in reader.tensors:
        if t.name == "output.weight":
            q = quants.quantize(to_f32(t), qtype)
            # a misread source (bytes taken as values) comes out larger, not smaller
            if q.nbytes >= t.n_bytes:
                raise SystemExit(f"converted output.weight is not smaller ({q.nbytes} >= {t.n_bytes} bytes)")
            converted[t.name] = q
            writer.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, qtype)
            print(f"output.weight: {t.tensor_type.name} {t.n_bytes / 1e9:.2f} GB -> {qtype.name} {q.nbytes / 1e9:.2f} GB")
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    if not converted:
        raise SystemExit("no output.weight tensor (the model may tie it to token_embd.weight)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(converted.get(t.name, t.data), tensor_endianess=reader.endianess)
    writer.close()
    print("written", dst)


if __name__ == "__main__":
    main()
