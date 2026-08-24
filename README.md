# IntelMacLlamaCpp

**llama.cpp with a working Metal backend for AMD GPUs on Intel Macs.**

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that makes GPU
inference actually correct on the AMD cards in Intel Mac Pros. Upstream compiles and runs
on these GPUs without a single error message — and produces semantically garbage tokens.
This fork fixes that.

Verified on a **Mac Pro (2019, MacPro7,1)** running **macOS 26.3.1**:

| model | prefill (pp512) | generation (tg64) |
|---|---:|---:|
| **Qwen3-30B-A3B** (MoE, Q4_K_M, 17.3 GB) | **179.2 t/s** | **51.9 t/s** |
| **Qwen3-8B** (dense, Q4_K_M, 4.7 GB) | 162.4 t/s | 46.9 t/s |
| either model on upstream llama.cpp | *garbage output* | *garbage output* |

Yes — a **30B model generates faster than an 8B** here. Qwen3-30B-A3B activates only
~3B parameters (8 of 128 experts) per token, so it is both quicker *and* far more
capable. It is the model to run on this hardware, and now leads on prefill too. Verified correct against the CPU
backend, routing included (see [Is it actually right?](#is-it-actually-right)).

Upstream's README is preserved as [README.upstream.md](README.upstream.md).

Forked from upstream `e85caa81ea2b65797396018c179b87ad61fa38ab` (2026-08-22).

---

## If you just want the best experience: use ToshLLM

**[ToshLLM](https://github.com/engeldlgado/toshllm)**, by Engelbert Delgado, has been doing
this longer and does it more thoroughly. If you own an Intel Mac with an AMD GPU and you
want working local LLMs rather than a study of why they were broken, **go there first.**

It is a polished native macOS app, not a patch set: model management, a real chat UI, and a
far wider range of supported hardware than this one machine. Technically it is ahead in
places this fork does not reach at all — flash attention on AMD, multi-GPU dispatch
(relevant if you have a Vega II *Duo*), a quantised KV cache, and proper mmap residency.

And it is faster. On identical hardware, the same model file and the same flags:

| Qwen3-30B-A3B Q4_K_M, Vega II | prefill | generation |
|---|---:|---:|
| this fork | 177 t/s | 51 t/s |
| **ToshLLM v0.85.7** | **615 t/s** | **53 t/s** |

Generation is level; his prompt processing is **3.5x** ours. Credit where it is due.

### On independence

The two projects arrived at many of the same fixes — wave64 reductions and mat-vec, a
register-tiled GEMM without `simdgroup_matrix`, its MoE variant, 16-bit quant loads,
rows-per-simdgroup tuning, device selection, a buffer-allocation guard. That convergence is
not borrowing in either direction; it is what happens when two people meet the same hardware.

For the avoidance of doubt: this work was done without knowledge of ToshLLM, which we found
on 2026-08-24. Every kernel change here was committed and pushed publicly before that — this
repository has been public since 2026-08-22 13:20 UTC, the wave64 mat-mul landed the same day
at 18:13, and the mat-vec and load-width work on 2026-08-23. The commit and push timestamps
are on GitHub and are not ours to edit. **No code has been copied from ToshLLM into this
repository**, and none can be: ToshLLM is GPL-3.0 and this fork is MIT, which also means
nothing here can travel the other way into his tree.

That licence difference is the one real reason this fork might still be useful to somebody.
llama.cpp is MIT and cannot accept GPL-3.0 patches, so of the two ports only this one could
ever go upstream, or be used in a permissively-licensed or commercial product. If you need
an app, use his. If you need patches you can relicense, or an explanation of *why* the
Metal backend fails on these cards, this is that.

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

Measured, not projected — see [the verified table](#models-verified-on-this-hardware) for
the full set. The short version:

| if you want | run |
|---|---|
| the best quality that fits | **Qwen3-30B-A3B Q6_K** (23 GB) — 208 / 47 t/s |
| the best model per GB | **Gemma-4-26B-A4B QAT** (13 GB) — 241 / 49 t/s |
| the fastest good model | **gpt-oss-20b MXFP4** (12 GB) — 231 / **62** t/s |
| something small and quick | Llama-3.2-3B Q5_K_M (2.3 GB) — 317 / 54 t/s |

### The VRAM budget

Metal reports `recommendedMaxWorkingSetSize = 34343 MB`, which is exactly 32 GiB, and
`hasUnifiedMemory = false`. Because the 580X drives the display, the Vega II pays no
framebuffer tax — all 32 GiB is available to compute.

Three things share it: **weights + KV cache + ~1.5 GiB of compute buffer.** Using the
bits-per-weight our own files imply (Qwen3-30B-A3B is 17.3 GiB for 30.5B = 4.54 bpw), that
leaves roughly 29.5 GiB for weights:

| quant | GiB per 1B params | largest that loads |
|---|---:|---:|
| Q2_K | 0.35 | ~85B |
| Q4_K_M | 0.57 | **~52B** |
| Q5_K_M | 0.68 | ~44B |
| Q6_K | 0.80 | ~37B |
| Q8_0 | 1.00 | ~30B |

Q8_0 of Qwen3-30B is 30.2 GiB and does **not** fit once KV and compute are added — Q6_K at
23.4 GiB is the largest usable quant of that model here.

**KV cache varies far more than people expect** — a 9x spread across models we run:

| | KiB/token | 32K context |
|---|---:|---:|
| gpt-oss-20b | 48 | 1.5 GiB |
| Qwen3-30B-A3B | 96 | 3.0 GiB |
| Gemma-3-12B | 384 | 12 GiB * |
| Gemma-4-26B-A4B | 420 | 13 GiB * |

\* both Gemmas use sliding-window attention (`swa=1024`), so only the global layers hold full
context and the real allocation is a fraction of that — but a Gemma-shaped model at long
context eats headroom a Qwen-shaped one does not.

**Budget ~50B total at Q4_K_M, and make it MoE.** MoE is the only structure where largest and
fastest coincide, since generation cost tracks active parameters. Two constraints, though:
`Q2_K` wrecks an MoE router (see below), and the IQ family — the usual way to squeeze 70B into
32 GB — is broken on this card, so that escape hatch is not available.

**Prefer MoE.** Generation speed tracks *active* parameters, so a 26B MoE outruns a dense
12B by more than 2x while being a far stronger model. All three MoE entries above beat every
dense model here on generation.

The card is **kernel-bound, not bandwidth-bound** — the dense 8B reads weights at roughly a
quarter of the ~830 GB/s the hardware can sustain. So a higher quantisation level costs less
speed than you would expect, and is usually worth taking.

### Quantization: use K-quants, avoid the IQ family

Verified correct on this card at real model shapes (k = 4096 and 12288, not just
the suite's k = 256):

> `q4_0` `q4_1` `q8_0` `q2_K` `q3_K` `q4_K` `q5_K` `q6_K` `mxfp4`

**Broken here — do not use:** `iq2_xxs` is measurably wrong at large k (fails
24/24), and `iq2_xs` **hangs the GPU**, which on this hardware takes the desktop
session down with it (see the warning below). The rest of the IQ family is
untriaged. `mxfp4` passing means gpt-oss-style models should work.

### Models verified on this hardware

Each was checked against the **CPU backend on identical text**, not merely eyeballed for
coherence. Throughput is `pp512` / `tg64` at `-ngl 99 -lm none` on an otherwise idle card.

| model | quant | size | prefill | generation | PPL (GPU) | n | vs CPU |
|---|---|---:|---:|---:|---:|---:|---:|
| **Gemma-4-26B-A4B** (MoE) | QAT Q4_K_XL | 13 GB | **241** | **49** | 777 † | 24 | −1.56% |
| **Qwen3-30B-A3B** (MoE) | **Q6_K** | 23 GB | **208** | 47 | **9.20** | 80 | +0.42% |
| **Qwen3-30B-A3B** (MoE) | Q4_K_M | 17 GB | 177 | **51** | 9.66 | 80 | +0.28% |
| **gpt-oss-20b** (MoE) | MXFP4 | 12 GB | 231 | **62** | 213 † | 80 | −0.02% |
| Qwen3-8B | Q4_K_M | 4.7 GB | 162 | 47 | 10.52 | 80 | bit-identical |
| Gemma-3-12B | Q6_K | 9.7 GB | 129 | 20 | 8.94 | 24 | +0.01% |
| Llama-3.1-8B | Q8_0 | 8.5 GB | 176 | 32 | 7.43 | 24 | −0.00% |
| Llama-3.2-3B | Q5_K_M | 2.3 GB | 317 | 54 | 10.95 | 24 | −0.00% |
| Llama-3.2-3B | f16 | 6.4 GB | 426 | **59** | 10.87 | 24 | −0.00% |

Four architectures (Qwen3, Llama, Gemma 3, Gemma 4), dense and MoE, seven quantisation
formats. Perplexity is wikitext-2 at `-c 512` over **n** chunks — absolute values are only
comparable between rows with the same **n**, so compare within a model, not down the column.
The **vs CPU** column is the one that matters here, since it isolates the backend from the
model.

† Reasoning-tuned models score raw prose poorly regardless of backend — gpt-oss and Gemma 4
are high on *both* CPU and GPU. Compare them against their own CPU figure, not against the
other rows.

**A bigger quant is not simply slower.** Q6_K of Qwen3-30B is 35% larger than Q4_K_M, yet
its **prefill is 17% faster** (208 vs 177 t/s) and its perplexity **4.8% better**, for only 8%
off generation (47 vs 51 t/s). The quality gain is not marginal: scoring both on identical
text and comparing per chunk gives a paired **t = 9.8** over 80 chunks, with Q6_K ahead on
85% of them. q4_K packs its scales as 6-bit fields that must be unpacked
per sub-block; q6_K uses plain 8-bit scales. On a card whose K-quant kernels are *not*
ALU-bound, reading more bytes and doing less work per byte is the better trade. **If the
VRAM is free, take the higher quant** — the usual size-versus-speed intuition is inverted
here for prefill.

**Generation tracks active parameters, not model size.** The three MoE models occupy the top
of the generation column despite being the largest here: a 26B model at 49 t/s and a 30B at
52 t/s, against 21 t/s for a dense 12B. On a 32 GB card, MoE is the format to reach for.

### Two 4-bit builds of the same model can differ by 21x

Worth its own heading because it cost real time to diagnose. Two Gemma-4-26B-A4B builds
from the same publisher, same corpus, same settings:

| build | size | prefill | generation | perplexity |
|---|---:|---:|---:|---:|
| `UD-Q4_K_M` | 16 GB | 214 | 39 t/s | **16097** |
| `qat-UD-Q4_K_XL` | 13 GB | 241 | **49 t/s** | **777** |

A 21x difference between two 4-bit quantisations of the same model — and the QAT build is
*smaller and faster* as well.

**We do not have a confirmed explanation, and the obvious one is wrong.** Inspecting the
tensors of both files: the router (`ffn_gate_inp`) is `F32` in *both*, as it is in every
GGUF we have — these quantisers never quantise it. Worse for the tidy story, the build that
scores 21x worse carries **higher** precision nearly everywhere (`Q8_0` attention, output
and embeddings, against `Q4_0` in the good build). Precision is not the variable separating
them.

Note also that 777 is itself a poor score — Gemma-3-12B manages 8.94 on the same corpus.
*Both* Gemma 4 builds are anomalous; one is merely catastrophic. Since the CPU backend
shows the same thing, this is not our Metal code, and it may not be a quantisation story at
all so much as a question about Gemma 4 support upstream.

**Practical rule, which is empirical and survives regardless of the mechanism: prefer a QAT
build, and measure perplexity before trusting any build.** If a model scores far worse than
you expect, suspect the weights before suspecting the backend — the bad build also showed an alarming +11% GPU-vs-CPU perplexity
gap that vanished (to −1.56%) with the QAT weights.

### Just use `-ngl 99`

Earlier versions of this README recommended a *partial* offload for prompt-heavy work,
because the CPU beat the GPU at prefill. The wave64 mat-mul removed that problem, and the
advice with it:

| `-ngl` | prefill (pp2048) | generation |
|---:|---:|---:|
| 0 (CPU only) | 85.9 | 16.1 |
| 12 (hybrid) | 96.9 | 16.9 |
| **99 (all GPU)** | **156.1** | **33.1** |

Full offload now wins on both axes, so there is no crossover left to reason about.

## Is it actually right?

Correct-looking text is not evidence, so both models were measured against the CPU backend
on **wikitext-2** (the standard corpus), 80 chunks each, comparing per-chunk log-likelihood:

| model | GPU | CPU | delta | paired t |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B (MoE) | 9.6700 | 9.6356 | **+0.357%** | +2.69 |
| Qwen3-8B (dense) | 10.5178 | 10.5342 | **−0.156%** | −2.59 |

Read these carefully, because the naive reading is wrong in both directions.

**These deltas are systematic, not run noise.** Both exceed |t| = 2, and re-running with a
different `ubatch` reproduces the GPU number bit-for-bit. Anyone claiming a GPU backend is
"identical" to CPU on the strength of a single perplexity number is over-claiming.

**But they are not a routing bug either.** The dense control deviates just as significantly
as the MoE model and in the *opposite* direction — the GPU is slightly *better* there. A
genuine MoE routing fault would make MoE worse while leaving the dense control at t ≈ 0.
What this actually shows is that the Metal and CPU backends differ by a small, systematic
amount whose sign depends on the model, which is what accumulated floating-point divergence
looks like. The MoE magnitude is about twice the dense one, consistent with expert selection
being a discrete `argmax` that can flip when two of 128 experts score within rounding error —
both choices being equally legitimate.

At ≤0.36% this sits well under Q4_K_M's own ~1–3% quantization cost, so it does not matter
in practice. Supporting checks: `TOP_K` and `ARGSORT` both pass at `ne = 128`, this model's
exact expert count, and every k-quant passes at the real `k = 4096 / 12288` shapes.

**A methodological warning if you repeat this.** `llama-perplexity` prints *cumulative*
perplexity. Read as per-chunk it manufactures a tidy monotone "error that starts large and
decays", which is purely an averaging artifact — it nearly produced a false positive here.
Differentiate the series first (`n·log(cumₙ) − (n−1)·log(cumₙ₋₁)`), then look at the signs
of the residuals: noise straddles zero, a real fault does not.

If you port this to other AMD hardware, re-run the comparison rather than trusting output:
`llama-perplexity -f wiki.test.raw -c 512 --chunks 80` with `-ngl 99` and then `-ngl 0`.

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
Vega II is **~6% generation throughput** (42.0 vs 44.8 t/s) — a cheap correctness fix.
Override with `GGML_METAL_CONCURRENCY_ENABLE=1` if you want to experiment.

**Retested after every kernel fix**, in case the nondeterminism had really been the broken
wave64 kernels rather than the barrier: it was not. With concurrency forced on, three
identical greedy runs still produce three *different* garbage outputs while the serial path is
correct. The barrier behaviour is the driver's, and the serialisation stays.

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

### 5. No mat-mul at all

`simdgroup_matrix` is gated on Apple7, so prompt processing fell back to mat-*vec* kernels —
one output column at a time — at roughly a third of the achievable rate. This is not merely
a missing optimisation: it also made speculative decoding unprofitable, because verifying K
drafted tokens costs ~Kx rather than ~1x without a batched mat-mul.

**Fix:** `kernel_mul_mm_w64`, an ordinary register-tiled GEMM that uses no matrix intrinsics,
so the 64-wide wavefront is simply 64 independent lanes:

* 64x32 output tile per threadgroup, K stepped in 32s
* 4 simdgroups x 64 lanes = 256 threads; **4x2 accumulators per thread**, kept in registers
* A and B staged in threadgroup memory **k-major**, so the inner loop reads its 4 rows as one
  `half4` and its 2 columns as one `half2`
* shared memory is 4096+2048 bytes — identical to the kernel it replaces, so the host
  allocation is untouched

| Qwen3-8B-Q4_K_M | before | after |
|---|---:|---:|
| pp512 | 48.4 | **162.9** |
| pp2048 | 45.3 | **156.7** |
| tg32 | 33.3 | 32.1 (unchanged — generation is still mat-vec) |

**~2.67 TFLOP/s, about 19% of the card's fp32 peak** — a 3.4x improvement, and the first time
the GPU beats the 24-thread Xeon at prompt processing (93.9 t/s).

The same kernel is applied to MoE expert layers as `kernel_mul_mm_id_w64`, where A is the
expert's weight matrix and the B rows are gathered through the id map:

| Qwen3-30B-A3B-Q4_K_M | before | after |
|---|---:|---:|
| pp512 | 88.4 | **179.2** |
| pp2048 | 76.4 | **172.2** |
| tg32 | 46.3 | 46.4 (unchanged) |

**Prefill in one line: 48 -> 163 t/s dense (3.4x), 88 -> 179 t/s MoE (2.0x).** Generation is
untouched by design — decoding one token at a time is genuinely a mat-vec problem.

### A compiler bug worth knowing about

The MoE kernel was wrong at first, and the cause is worth recording because anyone writing
Metal kernels for these GPUs can hit it. **Multiplying a `uint64_t` stride by a `short` index
is miscompiled by AMD's Metal compiler** — the short lands in the high word:

```
args.nb12 * i12          // i12 is short, value 128
  -> 4398046511104       // == nb12 << 32, not nb12*128 == 131072
args.nb12 * (int)i12     // correct
```

The B pointer therefore landed far outside the buffer and the tile multiplied uninitialised
memory. The symptom was bizarre and initially looked like a tiling bug: output was wrong from
**exactly token 128 onwards**, independent of matrix size, expert count and tile geometry —
128 being where a short index first pushes the miscompiled product past anything the buffer
covers. Use `int` for any index that multiplies a stride.

Verified rather than assumed: `MUL_MAT` passes **2129/2163**, all 34 failures being
pre-existing `iq2_xxs` breakage on the mat-vec path and none on the mat-mul path; perplexity
on wikitext-2 is **10.5168** against **10.5178** for the previous GPU path, a 0.01% shift.

`GGML_METAL_MM_W64_DISABLE=1` restores the old path. Note there is **no fallback** for a
missing type: the stock `simdgroup_matrix` kernel does not merely underperform here, it fails
to compile ("call to an undefined label"). All 51 type combinations are therefore
instantiated, and a missing one asserts rather than failing obscurely.

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

## Making it faster — what actually works here

The usual tuning advice assumes a CUDA card with flash-attention. Most of it is wrong on
this hardware, sometimes badly. Measured on Qwen3-8B-Q4_K_M:

### Already on, and worth more than everything else combined

**Prompt prefix caching.** llama-server keeps the KV cache per slot and only evaluates the
tokens you added, so a follow-up question does not reprocess the conversation:

| | time |
|---|---:|
| first turn, ~1460-token prompt | 21.19 s |
| follow-up, same prefix | **1.56 s** |
| fully warm | **1.30 s** |

**13.6×.** Nothing else on this list comes close. The practical consequence for anything
built on top: *do not modify the front of the conversation*. Dropping the oldest messages
to save context invalidates the prefix and forces a full reprocess. If you must trim, trim
in large steps with hysteresis so you pay that cost rarely rather than every turn.

### Actively harmful here

**KV cache quantization (`-ctk`/`-ctv q8_0`) — do not use.**

| KV type | prefill | generation |
|---|---:|---:|
| f16 (default) | 162.9 | **41.8** |
| q8_0 | 114.0 | **16.2** |

Generation drops **61%**, and prefill 30% with it. Dequantizing K/V inside attention costs far more than the
bandwidth it saves, because the flash-attention kernels that normally absorb that cost are
disabled on any non-Apple7 GPU.

**Speculative decoding — also a loss**, whether by draft model or n-gram:

| | generation |
|---|---:|
| baseline | 25.1 t/s |
| draft model (Qwen3-0.6B Q8_0) | **15.3 t/s** |
| n-gram (`ngram-simple`/`map-k`/`mod`) | 24–26 t/s (noise) |

These numbers predate the wave64 mat-mul and **should be re-measured**; the reason
speculation lost is exactly the thing that was fixed. The draft model was not the problem —
acceptance was **65%, mean accepted length 2.95**,
which would normally be a solid win. The problem is verification. Speculation assumes
checking K drafted tokens costs about as much as generating one, which requires a batched
**mat-mul**. This card only has the mat-**vec** fallback, so verifying ~4 tokens costs
roughly 4×, and you pay for the draft model on top.

### Mat-vec: rows per simdgroup (+27% generation)

Prefill was the obvious target; generation turned out to have a cheaper win. Measuring each
kernel against the card's ~830 GB/s copy ceiling located it immediately — and it was **not** a
bandwidth problem:

| kernel | GB/s | % of ceiling |
|---|---:|---:|
| f32 / f16 | 675 / 654 | **~80%** |
| q4_0 | 428 | 52% |
| q4_K | 219 | 26% |
| q5_K | 90 | **11%** |

The float kernels nearly saturate the card, so the memory path was fine; only the *quantised*
kernels were starved. The cause is `N_R0_*` — how many src0 rows one simdgroup accumulates,
which sets how far the activation-vector load is amortised. The stock values are tuned for
32-wide Apple waves, and the correlation is already visible in the stock numbers above:
q4_0 has `nr0=4`, q4_K has `2`, q5_K has `1`.

Raising it to 8 (measured optimum) gives:

| | q4_K | q5_K | q6_K | q2_K | q3_K | q5_0 |
|---|---:|---:|---:|---:|---:|---:|
| stock | 219 | 90 | 191 | 123 | 92 | 344 |
| tuned | **369** | **139** | **208** | **216** | **129** | **395** |

**Generation: 33.2 -> 42.0 t/s on the dense 8B, 46.4 -> 49.9 on the MoE.** Prefill unchanged.
Perplexity is bit-identical (10.5168) — this redistributes work without touching arithmetic.

Two findings worth keeping. **`nsg` is irrelevant** — 1, 2 and 4 simdgroups give 369/368/368
GB/s, so `nr0` is the entire effect. And **`nr0` is non-monotonic**: 8 is optimal, 16 and 32
are both worse as register pressure outweighs the amortisation. `q8_0` regressed at 8 and
keeps its stock value.

The constants are keyed on the probed wave width rather than changed outright, so Apple GPUs
are unaffected: `N_SIMDWIDTH` is injected when the shader library is compiled, and the host
selects the matching variant from `simd_width`.

### Load width: 16-bit reads in q5_K and q6_K (+37% on Q6_K models)

After the `nr0` tuning above, q4_K reached 44% of the card's bandwidth but q5_K was still
at 11% and q6_K at 25%. The cause was not arithmetic and not launch parameters.

**The diagnostic that settled it:** stripping almost all the arithmetic out of the q5_K
inner loop, while leaving every memory access in place, made it **2.5% faster**. A kernel
that does not care whether you remove its maths is not ALU-bound. Comparing against q4_K,
which reaches 2.7x its bandwidth, showed the real difference:

| | reads quants as | loads per 8 quants |
|---|---|---:|
| q4_K | `uint16_t` | 4 |
| q5_K, q6_K | `uint8_t` | **8** |

Twice the memory transactions at half the width. Both kernels now read 16 bits at a time
and split the bytes in registers, which the block layouts permit — q5_K has `qh` at offset
16 and `qs` at 48 in a 176-byte block, q6_K has `ql` at 0 and `qh` at 128 in a 210-byte
block, so every cast is 2-byte aligned.

| | before | after |
|---|---:|---:|
| q5_K | 139 | **206 GB/s** (1.48x) |
| q6_K | 208 | **328 GB/s** (1.58x) |

Generation, end to end:

| model | before | after |
|---|---:|---:|
| **Gemma-3-12B Q6_K** | 15.0 | **20.6 t/s** (1.37x) |
| Llama-3.2-3B Q5_K_M | 44.7 | **54.4 t/s** |
| Qwen3-8B Q4_K_M | 42.0 | **46.9 t/s** |

Perplexity is bit-identical before and after (Gemma-3-12B 8.9436, Llama-3.2-3B 10.9549).
`q2_K` and `q3_K` already used 16-bit loads and were not affected.

### The one that mattered, and what is left

Every item above traced back to `has_simdgroup_mm` being gated on `MTLGPUFamilyApple7`. That
is now addressed for `MUL_MAT` by [`kernel_mul_mm_w64`](#5-no-mat-mul-at-all): prefill went
from 48.4 to 162.9 t/s on the dense 8B, and the GPU now beats the CPU at prompt processing
instead of losing to it.

Both `MUL_MAT` and `MUL_MAT_ID` now use it, so dense and MoE prefill are both covered, and
the mat-vec tuning above lifted generation by 27%. Where the card stands now:

| | dense 8B | MoE 30B-A3B |
|---|---:|---:|
| prefill | 48 -> **163 t/s** | 88 -> **175 t/s** |
| generation | 33 -> **42 t/s** | 46 -> **50 t/s** |

**The remaining ~2x in generation is a memory access pattern, not a tuning constant.** q4_K now
reaches 369 GB/s — 44% of the ceiling, up from 26% — but the float kernels still get ~80%. A
K-quant super-block scatters its quants, high bits and packed scales across 144 bytes, and
eight threads read different fields of it; f16 reads contiguous `half4x4`. Closing that means
loading blocks cooperatively before dequantising, which is a kernel rewrite rather than a
constant change.

**Speculative decoding is still a loss, and now we know why for certain.** It was retested
after the mat-mul landed and remains ~40% slower. The cause is not draft quality (acceptance
0.60-0.65, mean accepted length ~2.8) and not the absence of mat-mul. It is that throughput
barely improves at the batch sizes speculation produces:

| batch | 1 | 4 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|---:|
| relative throughput | 1.00x | **0.92x** | 0.98x | 2.42x | 3.93x |

Verifying a 3-token draft costs *more* than three single-token passes, so the draft model is
pure overhead twice over. Batching only becomes cheap above ~32. Note this got **worse** after
the mat-vec tuning below: single-token decode sped up 27% while batch-4 did not, widening the
gap that makes speculation unprofitable.

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
| `GGML_METAL_MM_W64_DISABLE` | disable the wave64 mat-mul (dense and MoE) and fall back to mat-vec prefill, for A/B measurement |
| `GGML_METAL_MM_MIN` / `GGML_METAL_MM_ID_MIN` | batch-size thresholds above which the mat-mul kernels are used (defaults 8 / 32) |

---

## Known remaining issues

- **`GET_ROWS(iq2_xs)` hangs the GPU.** Not used by `Q4_K_M`. Untriaged.
- **At least one non-inference op hangs** (the `ssm_scan` / `conv` / `fwht` family was never
  audited for wave64). This is what triggered the WindowServer kill.
- **`DSV4_HC_PRE/POST/COMB` fail** (DeepSeek-V4 hybrid-context ops, ERR ≈ 0.4–1.1). Not on
  any path used here.
- **Quantised mat-vec still leaves ~2x on the table.** After tuning, q4_K reaches 44% of the
  card's bandwidth against ~80% for the f16 kernel. The gap is the K-quant super-block access
  pattern, not the launch parameters, so closing it means restructuring the kernel to load
  blocks cooperatively before dequantising.
- **Speculative decoding cannot be rescued by tuning.** Throughput at batch 4 is now *below*
  batch 1 (0.92x), because the mat-vec tuning sped up single-token decode without helping
  small batches. Verifying a draft costs more than decoding it.
- Nothing here is tested on Apple Silicon. The changes are written to be no-ops there — the
  probe returns 32 and concurrency stays enabled — but that is unverified.

---

## License

MIT, same as upstream llama.cpp. See [LICENSE](LICENSE).
