# IntelMacLlamaCpp

**llama.cpp with a working Metal backend for AMD GPUs on Intel Macs.**

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that makes GPU
inference actually correct on the AMD cards in Intel Mac Pros. Upstream compiles and runs
on these GPUs without a single error message — and produces semantically garbage tokens.
This fork fixes that.

Verified on a **Mac Pro (2019, MacPro7,1)** running **macOS 26.3.1**:

| model | prefill (pp512) | generation (tg64) |
|---|---:|---:|
| **Qwen3-30B-A3B** (MoE, Q4_K_M, 17.3 GB) | **88.5 t/s** | **46.6 t/s** |
| **Qwen3-8B** (dense, Q4_K_M, 4.7 GB) | 48.4 t/s | 32.9 t/s |
| either model on upstream llama.cpp | *garbage output* | *garbage output* |

Yes — a **30B model runs faster than an 8B** here. Qwen3-30B-A3B activates only
~3B parameters (8 of 128 experts) per token, so it is both quicker *and* far more
capable. It is the model to run on this hardware. Verified correct against the CPU
backend, routing included (see [Is it actually right?](#is-it-actually-right)).

Upstream's README is preserved as [README.upstream.md](README.upstream.md).

Forked from upstream `e85caa81ea2b65797396018c179b87ad61fa38ab` (2026-08-22).

---

## Quick start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j
```

```bash
GGML_METAL_DEVICE=Vega ./build/bin/llama-cli \
  -m Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -lm none -st \
  -p "The capital of France is"
```

Get the model (18.6 GB) from the official Qwen repo:

```bash
curl -L -C - -o Qwen3-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
```

Two flags matter enormously on this hardware:

- **`GGML_METAL_DEVICE=Vega`** — selects the compute GPU. Without it, upstream picks the
  GPU driving your display, which on a Mac Pro is usually the *weaker* card.
- **`-lm none`** — loads weights into VRAM. With the default mmap the weights stay in host
  memory and every matmul reads across PCIe: **~2 t/s instead of ~32 t/s**, a 16× penalty.

---

## What to run on 32 GB of VRAM

The card is **kernel-bound, not bandwidth-bound** — the dense 8B sustains only
~152 GiB/s of an HBM2 part capable of ~1 TB/s. So speed tracks *bytes read per
token*, which is why MoE wins so decisively, and why a higher quant costs less
speed than you would expect.

| model | quant | weights | generation |
|---|---|---:|---:|
| **Qwen3-30B-A3B** | **Q4_K_M** | **17.3 GB** | **46.6 t/s** (measured) |
| Qwen3-30B-A3B | Q5_K_M / Q6_K | 21.7 / 25.1 GB | fits, ~5–15% slower |
| Qwen3-8B | Q4_K_M | 4.7 GB | 32.9 t/s (measured) |
| Qwen3-8B | Q8_0 | 8.1 GB | ~19 t/s |
| Qwen3-14B | Q5_K_M / Q6_K | 10–11 GB | ~14 t/s |
| Qwen3-32B (dense) | Q4_K_M | 18.5 GB | ~8 t/s |
| anything 70B | — | ≥31 GB | will not fit with a KV cache |

### Quantization: use K-quants, avoid the IQ family

Verified correct on this card at real model shapes (k = 4096 and 12288, not just
the suite's k = 256):

> `q4_0` `q4_1` `q8_0` `q2_K` `q3_K` `q4_K` `q5_K` `q6_K` `mxfp4`

**Broken here — do not use:** `iq2_xxs` is measurably wrong at large k (fails
24/24), and `iq2_xs` **hangs the GPU**, which on this hardware takes the desktop
session down with it (see the warning below). The rest of the IQ family is
untriaged. `mxfp4` passing means gpt-oss-style models should work.

### Prompt-heavy work: the CPU is still better at prefill

Because `simdgroup_matrix` is unavailable, prompt processing falls back to
mat-vec kernels and the Xeon beats the GPU at it. For the dense 8B, a *partial*
offload is faster than either extreme:

| `-ngl` | prefill (pp2048) | generation |
|---:|---:|---:|
| 0 (CPU) | 77 | 15.3 |
| **12** | **91.5** | 17.1 |
| 36 (all GPU) | 46.5 | **31.3** |

Rule of thumb: use `-ngl 12` when your prompt is more than ~2.5× your expected
output (summarising a long document); use `-ngl 99` otherwise (chat, agents).
This matters much less for MoE, which narrows the GPU's prefill deficit to 1.35×.

## Is it actually right?

Correct-looking text is not evidence. These backends were checked against the CPU
backend on identical input, which is the only thing that catches subtly-wrong
output:

```
Qwen3-8B      Vega II  PPL 23.2535   CPU  PPL 23.2960    delta -0.18%
Qwen3-30B-A3B Vega II  PPL 23.4508   CPU  PPL 23.2909    delta +0.69%
```

Both deltas are floating-point reduction-order noise. The MoE figure was
investigated specifically — top-k expert routing runs through `argsort`/`top_k`,
and wrong routing yields fluent-but-worse text rather than an obvious failure.
Per-chunk residuals straddle zero (the GPU is *better* on 3 of 12 chunks),
t = 2.06 / p = 0.064, and `TOP_K`/`ARGSORT` both pass at `ne = 128`, this model's
exact expert count. The residual is consistent with expert flips on near-tied
router scores, where both backends make an equally valid choice, and it sits well
below Q4_K_M's own ~1–3% quantization cost.

If you port this to other AMD hardware, re-run that comparison rather than
trusting the output — `llama-perplexity` against `-ngl 0` on the same text file.

## The hardware, and why it is awkward

```
Mac Pro 7,1 · macOS 26.3.1 · x86_64
  CPU   Xeon W-3235, 12C/24T, AVX-512 + VNNI, 192 GB RAM
  GPU 0 Radeon Pro Vega II, 32 GB HBM2, Metal 3, wave64   <- compute, headless
  GPU 1 Radeon Pro 580X,     8 GB,      Metal 2, wave64   <- drives the display
```

Three properties break assumptions baked into the Metal backend:

1. **Two GPUs, and the fast one is not the default.** The Vega II is headless; the 580X
   owns the screen.
2. **Wave64.** Apple GPUs execute 32 threads per SIMD group. These AMD GCN parts execute
   **64**. `threadExecutionWidth` reports 64 and the kernels were written for 32.
3. **Discrete, not unified memory.** `hasUnifiedMemory = false`, so weights must be copied
   into VRAM, and `maxBufferLength` is only **3.76 GB** on both cards — a 4.7 GB model
   cannot live in one Metal buffer.

Note also: `has_simdgroup_mm` is gated on `MTLGPUFamilyApple7`, so the `simdgroup_matrix`
matmul path and flash-attention are **off** on any AMD card. Upstream already falls back to
mat-vec kernels — that costs prefill throughput but is not a correctness bug, and it is why
prefill here is slower than the 24-thread CPU while generation is far faster.

---

## What was broken, and what this fork changes

### 1. It picks the wrong GPU

`ggml_metal_device_init()` calls `MTLCreateSystemDefaultDevice()`, which returns the GPU
attached to the main display — here the **8 GB 580X**. That card reports Metal 2, so
`has_simdgroup_reduction` is `false`, which gates *every* mat-vec kernel, softmax and norm.
The Metal backend is effectively dead before it starts:

```
$ ./build/bin/llama-cli --list-devices     # upstream
  MTL0: AMD Radeon Pro 580X (8192 MiB, 8191 MiB free)     <- wrong card, Metal 2
```

**Fix:** added `GGML_METAL_DEVICE`, accepting an index into `MTLCopyAllDevices()` or a
case-insensitive name substring.

```
$ GGML_METAL_DEVICE=Vega ./build/bin/llama-cli --list-devices
  MTL0: AMD Radeon Pro Vega II (32752 MiB, 32751 MiB free)
```

### 2. Wave64: the kernels assume 32-lane SIMD groups

`ggml-metal.metal` hardcodes `#define N_SIMDWIDTH 32`. On a 64-wide wavefront, every
`simd_sum` folds over 64 lanes while the surrounding index arithmetic still tiles for 32.
Nothing crashes; the dot products are just quietly wrong.

**Fix:** the SIMD width is now *probed* at runtime — a trivial kernel is compiled and its
pipeline's `threadExecutionWidth` read — then injected into the shader as an
`N_SIMDWIDTH` preprocessor macro, and threaded through every host-side dispatch size,
threadgroup-memory allocation and coverage formula. Apple Silicon still resolves to 32 and
is unaffected.

```
ggml_metal_device_init: simdgroup reduction   = true
ggml_metal_device_init: simdgroup matrix mul. = false
ggml_metal_device_init: simd group width      = 64     <- new
```

Kernels corrected: `q8_0`, the shared `q4_0/q4_1/q5_0/q5_1` template, `t_t`, `t_t_4`,
`t_t_short`, `tq2_0`, `q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`, `mxfp4`, `iq4_nl`, `iq4_xs`,
and `mul_mv_ext`. Flash-attention kernels are pinned to a 32-lane layout (they are disabled
at runtime on these cards regardless).

Two genuinely distinct bug shapes turned up:

- **Lane→block mapping.** e.g. `nypsg = 32/nxpsg` in the ext kernels, or `tid = tiisg/4`
  in `q5_K` — these must derive from the real wave width.
- **Pointer advances left at a literal stride.** The subtler one. In `q3_K` and `q5_K` the
  loop counter became width-dependent (`ib += N_SIMDWIDTH/8`, i.e. 8 on wave64) while the
  y-pointer at the bottom of the loop still advanced by the old constant:

  ```c
  y1 += 4 * QK_K;          // wrong: strides 8 blocks, advances y by 4
  y1 += (N_SIMDWIDTH/8) * QK_K;   // fixed
  ```

  `q6_K` was immune because it recomputes `yy + i*QK_K` fresh each iteration.

### 3. A race, not arithmetic

With every kernel numerically correct, full-offload output was *still* garbage — and a
**different** garbage on each run. That nondeterminism is the signature of a race.

The concurrent-dispatch path reorders independent ops into one encoder and separates
dependent ones with `memoryBarrierWithScope:MTLBarrierScopeBuffers`. Apple's TBDR honours
that barrier; **the AMD macOS driver does not enforce it between concurrently dispatched
kernels.**

**Fix:** concurrent dispatch is now restricted to Apple-family GPUs. Measured cost on the
Vega II is **~4.6% generation throughput** (32.4 vs 34.0 t/s) — a very cheap correctness fix.
Override with `GGML_METAL_CONCURRENCY_ENABLE=1` if you want to experiment.

An attempt to keep concurrency by ending and restarting the encoder at each dependency point
made things *worse* — restarting discards encoder state that multi-dispatch ops rely on. That
code remains behind `GGML_METAL_BARRIER_RESTART_ENABLE` but is off by default.

### 4. `set_tensor` crashes on discrete GPUs

On a non-unified-memory device, `ggml_metal_buffer_set_tensor` wraps the *host* pointer with
`newBufferWithBytesNoCopy:` and asserts the result — but that call requires a page-aligned
address and length, so it returns `nil` for ordinary tensor pointers and aborts:

```
ggml-metal-device.m:1910: GGML_ASSERT(buf_src) failed
  ggml_metal_buffer_set_tensor
  test_rope_set_rows::initialize_tensors
```

**Fix:** wrap the *containing pages* and blit from the correct offset when the memory can be
wrapped; otherwise fall back to a chunked staged copy through a shared buffer. Also fixed an
off-by-semantics bug in `memset_tensor`, where `NSMakeRange(offs, offs + size)` passed an
end-offset where a length was expected.

---

## Why every one of these passed the upstream test suite

`test-backend-ops` exercises k-quant `MUL_MAT` at **`k = 256` only** — and `q5_K`, `q6_K`,
`q2_K`, `q3_K` only appear via `other_types`, at that single size. `k=256` is `nb=1`: exactly
one block per row, one loop iteration, so *every* tiling and stride variant is trivially
correct.

Qwen3-8B uses `k=4096` and `k=12288` — `nb=16` and `nb=48`.

The consequence is worth stating plainly: **`q5_K` and `q3_K` passed all 1161 MUL_MAT tests
while being catastrophically wrong (relative error ≈ 1.0) at the only shapes the model
actually uses.** A green op suite proved nothing until the tested shapes matched the model's.

`tests/test-backend-ops.cpp` here adds `k = {4096, 12288}` and promotes the k-quants into
`base_types`. That change alone turns a clean run into 24 hard failures on upstream kernels.

The progression, for the record:

| stage | MUL_MAT result |
|---|---|
| upstream on Vega II | 304 pass / 521 fail |
| after wave64 kernel fixes | 1161 / 1161 pass *(at k=256)* |
| after adding real k sizes | q5_K, q3_K fail at k≥4096, ERR ≈ 0.8–1.8 |
| after stride fixes | all k-quants clean at k = 256, 4096, 12288 |

---

## ⚠️ Do not run the full `test-backend-ops` suite on this hardware

It will hang the GPU and take your desktop session with it.

Both cards share the `IOAcceleratorFamily2` kernel driver. A compute hang on the headless
Vega II stalls the **580X**, which is driving the display; the display stack stops
presenting, and after 40 seconds the userspace watchdog kills WindowServer and logs you out.

This is not a kernel panic — the spin dump shows WindowServer's main thread wedged in the
*display* GPU's driver while we were computing on the other card:

```
CoreDisplay → Metal → AMDMTLBronzeDriver → IOAccelerator → [kernel]
  → IOAcceleratorFamily2 → AMDRadeonX4000 (Ellesmere = 580X) → IOPCIFamily
  ... 12/12 samples, never moved
termination: WATCHDOG — "40 seconds since last successful checkin"
```

Test **one op at a time, under a timeout**. `safe-sweep.sh` in this repo does that; every
core inference op passes:

```
GET_ROWS   OK=65   FAIL=0   *** HUNG ***   (hangs on iq2_xs — unused by Q4_K_M)
RMS_NORM   OK=50   FAIL=0
NORM       OK=50   FAIL=0
SOFT_MAX   OK=209  FAIL=0
ROPE       OK=457  FAIL=0
ADD/MUL/SCALE/CPY/CONT/GLU/DIV        all FAIL=0
```

---

## Environment variables added by this fork

| variable | effect |
|---|---|
| `GGML_METAL_DEVICE` | select GPU by index into `MTLCopyAllDevices()` or name substring |
| `GGML_METAL_CONCURRENCY_ENABLE` | force concurrent dispatch on non-Apple GPUs (**produces wrong results here**) |
| `GGML_METAL_NO_ZEROCOPY` | force staged copies in `set_tensor`/`get_tensor` (debugging) |
| `GGML_METAL_BARRIER_RESTART_ENABLE` | barrier via encoder restart instead of `memoryBarrierWithScope:` (experimental) |

---

## Known remaining issues

- **`GET_ROWS(iq2_xs)` hangs the GPU.** Not used by `Q4_K_M`. Untriaged.
- **At least one non-inference op hangs** (the `ssm_scan` / `conv` / `fwht` family was never
  audited for wave64). This is what triggered the WindowServer kill.
- **`DSV4_HC_PRE/POST/COMB` fail** (DeepSeek-V4 hybrid-context ops, ERR ≈ 0.4–1.1). Not on
  any path used here.
- **Prefill is slower than the CPU** because `simdgroup_matrix` is unavailable, so prompt
  processing falls back to mat-vec kernels — the GPU manages only ~6% of its ~14 TFLOPS fp32
  peak. **A wave64 `mul_mm` built on `simd_shuffle` rather than `simdgroup_matrix` is the
  single largest piece of unclaimed performance here**, and the most useful contribution
  anyone could make to this fork.
- **MoE prefill is stuck on the mat-vec path too.** `ggml-metal-ops.cpp` gates the expert
  mat-*mul* on `has_simdgroup_mm`, so `mul_mv_id` is always used. Generation is unaffected
  (mat-vec is the right kernel for one token); long-prompt MoE work pays for it.
- Nothing here is tested on Apple Silicon. The changes are written to be no-ops there — the
  probe returns 32 and concurrency stays enabled — but that is unverified.

---

## License

MIT, same as upstream llama.cpp. See [LICENSE](LICENSE).
