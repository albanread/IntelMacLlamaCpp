# IntelMacLlamaCpp

**llama.cpp with a working Metal backend for AMD GPUs on Intel Macs.**

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that makes GPU
inference actually correct on the AMD cards in Intel Mac Pros. Upstream compiles and runs
on these GPUs without a single error message — and produces semantically garbage tokens.
This fork fixes that.

Verified on a **Mac Pro (2019, MacPro7,1)** running **macOS 26.3.1**:

| | prefill (pp128) | generation (tg64) |
|---|---:|---:|
| **Radeon Pro Vega II — Metal, this fork** | **50.0 t/s** | **32.4 t/s** |
| Xeon W-3235, 24 threads — CPU | 64.1 t/s | 14.1 t/s |
| Radeon Pro Vega II — Metal, upstream | *garbage output* | *garbage output* |

Model: `Qwen3-8B-Q4_K_M` (4.68 GiB, 8.19B params). Generation is **2.3× the CPU**.
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
  -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -lm none -st \
  -p "The capital of France is"
```

Two flags matter enormously on this hardware:

- **`GGML_METAL_DEVICE=Vega`** — selects the compute GPU. Without it, upstream picks the
  GPU driving your display, which on a Mac Pro is usually the *weaker* card.
- **`-lm none`** — loads weights into VRAM. With the default mmap the weights stay in host
  memory and every matmul reads across PCIe: **~2 t/s instead of ~32 t/s**, a 16× penalty.

---

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
- **Prefill is slower than the CPU** (50 vs 64 t/s) because `simdgroup_matrix` is
  unavailable, so prompt processing falls back to mat-vec kernels. A wave64 `mul_mm` written
  with SIMD shuffles is the obvious next win.
- Nothing here is tested on Apple Silicon. The changes are written to be no-ops there — the
  probe returns 32 and concurrency stays enabled — but that is unverified.

---

## License

MIT, same as upstream llama.cpp. See [LICENSE](LICENSE).
