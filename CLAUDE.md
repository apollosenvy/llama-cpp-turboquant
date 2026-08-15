# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

This is a **fork** of `ggml-org/llama.cpp` that adds **TurboQuant** (turbo2/turbo3/turbo4) — a family of WHT-rotated, Lloyd-Max-quantized KV cache compression types with backend kernels for CUDA, HIP/ROCm, Vulkan, and Metal. The headline win is turbo3 at ~3 bits per K/V element with attention quality within 5% perplexity of `q8_0` (validated by `scripts/turbo-quality-gate.sh`).

Upstream remote is `origin` (TheTom/llama-cpp-turboquant); Gary's fork remote is `fork` (apollosenvy/llama-cpp-turboquant). The base llama.cpp tree follows `ggml-org/llama.cpp` master via periodic upstream-sync commits.

**`AGENTS.md` is the upstream policy and forbids AI-generated PRs to mainline llama.cpp.** That policy does not apply to work landing on this fork (private/personal). When the goal is an upstream PR (any `pr/*` branch destined for `ggml-org/llama.cpp`), AGENTS.md and CONTRIBUTING.md apply in full — AI may assist a human contributor only, not author. When the goal is fork-local work or experiments, work freely.

## Branch Convention

| Prefix | Meaning |
|--------|---------|
| `master` | Tracks upstream `ggml-org/llama.cpp` master + periodic syncs |
| `feature/*` | Long-lived fork feature branches (e.g. `feature/turboquant-kv-cache`) |
| `pr/*` | Branches prepared for upstream PR submission — AGENTS.md applies |
| `experiment/*` | Research branches, may be abandoned (`fork/experiment/*` on remote) |
| `fix/*` | Targeted bugfixes |
| `backup/*-YYYY-MM-DD` | Pre-rebase snapshots; do not delete without checking |

Current canonical work branch is `feature/turboquant-kv-cache` with `pr/vulkan-turbo3-rebase` as the sibling stabilization branch.

## Build (Aegis hardware — 7900 XTX / gfx1100)

The canonical configuration is HIP + Vulkan, Release, in `build-hipvk/`:

```bash
cmake -B build-hipvk -DGGML_HIP=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
      -DAMDGPU_TARGETS=gfx1100 -DGPU_TARGETS=gfx1100 -G Ninja
cmake --build build-hipvk -j$(nproc)
```

Other present build directories (`build/`, `build-vk/`, `build-vulkan/`, `build-hip-sanity/`) reflect prior configurations; check `CMakeCache.txt` before reusing one. `CMakePresets.json` covers many platforms but the HIP+Vulkan combo isn't preset, so use the explicit `cmake -B` form above.

Binaries land in `build-hipvk/bin/llama-*`.

## Test & Quality Gates

**The single most important command in this fork** is the TurboQuant quality gate. Run it BEFORE any push that touches turbo* kernels, the FA dispatch, or the KV cache:

```bash
LLAMA=build-hipvk/bin MODEL=~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf \
  bash scripts/turbo-quality-gate.sh
```

It checks two invariants:
1. **Perplexity** of `-ctk turbo3 -ctv turbo3 -fa on` is within 5% of the `q8_0` baseline (6.111 reference)
2. **Context-scaling speed** ratio turbo3/q8_0 at 4K context is > 0.95

A regression on either is a release-blocker.

C-level round-trip test (no GPU required, fast):

```bash
cmake --build build-hipvk --target test-turbo-quant
./build-hipvk/bin/test-turbo-quant
```

Backend equivalence test (run when modifying ggml ops or backend kernels):

```bash
./build-hipvk/bin/test-backend-ops
```

Single-test invocation (CTest):

```bash
ctest --test-dir build-hipvk -R <test-name-regex> --output-on-failure
```

Bench:

```bash
./build-hipvk/bin/llama-bench -m <model> -ctk turbo3 -ctv turbo3 -fa 1 -ngl 99
```

## TurboQuant Architecture

The fork adds three new `ggml_type` values — `GGML_TYPE_TURBO2_0`, `_TURBO3_0`, `_TURBO4_0` — and three corresponding `llama` cache types (`LLAMA_TYPE_TURBO*`). They share an encode/decode pipeline:

```
       store path                              load path
  +----------------------+               +----------------------+
  | K (or V) tile  [F32] |               | block_turbo3_0       |
  +----------+-----------+               +----------+-----------+
             | WHT rotation (128x128)               | centroid lookup
             v                                      v (Lloyd-Max LUT)
  +----------------------+               +----------------------+
  | rotated tile [F32]   |               | rotated values [F16] |
  +----------+-----------+               +----------+-----------+
             | Lloyd-Max quantize                   | inverse WHT
             v (8 centroids, FP16 norm)             | (applied to Q at
  +----------------------+                          |  graph build time
  | block_turbo3_0       |                          |  so attention is
  |  qs[8] signs[4] norm |                          |  computed in the
  +----------------------+                          |  rotated domain)
                                                    v
                                          +----------------------+
                                          | <Q_rot, K_rot> score |
                                          +----------------------+
```

Critical detail: dequant leaves K in the **rotated domain**; Q is also rotated by the graph build, so `<Q_rot, K_rot>` yields the correct attention score with no explicit inverse on the hot path. Inverse WHT is applied to V output only.

**Files:**

- `ggml/src/ggml-turbo-quant.c` — block types, CPU encode/decode, type traits
- `src/turbo-rotation-data.h` and `-32.h` — precomputed 128x128 and 32-group rotation matrices
- `ggml/src/ggml-cuda/turbo-wht.{cu,cuh}` + `turbo-innerq.{cu,cuh}` — CUDA/HIP WHT and inner-quant kernels
- `ggml/src/ggml-cuda/fattn-vec-instance-turbo*_0-*.cu` — flash-attention instantiations for every (K-type, V-type) combination involving turbo2/3/4 and q8_0/f16
- `ggml/src/ggml-vulkan/...` — Vulkan equivalents (shader-based, currently being rebased on `pr/vulkan-turbo3-rebase`)
- `ggml/src/ggml-metal/turbo-matrices.h` + `turbo-wht.h` — Metal precomputed data

The KV cache integration sits in `src/llama-kv-cache.cpp:215-359` — when type is `turbo*_0`, the cache allocates the per-context WHT rotation tensors (`turbo_rotation`, `turbo_rotation_inv`, `turbo_innerq_scale_inv`) once and passes them to the graph via `get_turbo_rotation*()` accessors. Per-layer K and V tensors are allocated with `n_embd_*_gqa_eff` padded up to the next 128-element boundary so the WHT group fits without straddle.

## Layer-Adaptive Precision

Set `TURBO_LAYER_ADAPTIVE=N` (env var) to mix `q8_0` and `turbo*` per layer for quality recovery. Mode `7` is auto-enabled when V uses `turbo2_0` and gives a free PPL recovery by promoting the first+last 2 layers of K and V to `q8_0`. Per-layer logic lives in `src/llama-kv-cache.cpp:252-301`.

This is the same machinery a hot/cold KV tiering implementation would extend (see `docs/hot-cold-kv-design.md`).

## Where Backend Equivalence Is Maintained

Every turbo* code path must have:

1. CPU reference encode/decode in `ggml-turbo-quant.c` (the truth)
2. CUDA path (which compiles for HIP via translated `.cu` files)
3. HIP-specific FA instances when CUDA dispatch doesn't cover (search `fattn-vec-instance-*.cu`)
4. Vulkan shader path under `ggml/src/ggml-vulkan/`
5. Metal path under `ggml/src/ggml-metal/`

When adding a new type or kernel, all five must remain in sync. `test-backend-ops` is the cross-check; missing instances are the most common cause of "works on CPU, crashes on GPU" regressions.

## Coding Guidelines

Inherited from upstream `CONTRIBUTING.md`:

- No third-party dependencies, no extra headers
- Cross-platform: never break Apple, Windows, RISC-V
- Plain C++: basic `for` loops, no fancy STL, avoid templates where possible
- 4-space indent, brackets on same line, `void * ptr`, `int & a`
- Vertical alignment is preferred — makes batch edits easier
- Sized integer types (`int32_t`) in public API; `size_t` for sizes/offsets
- `struct foo {}` not `typedef struct foo {} foo`

When work is destined for upstream (any `pr/*` branch heading to `ggml-org/llama.cpp`), follow `AGENTS.md`: a human contributor must understand, defend, and maintain the code; AI assistance only.

When work is fork-local: write code that another developer in this fork can understand and extend, but the upstream "no AI PRs" policy does not apply.

## Project-Specific Things That Look Weird But Aren't

- **WHT padding to 128:** if `head_dim % 128 != 0` (e.g. some Gemma variants), the cache pads up. Padding zeros don't affect dot products because WHT preserves inner products.
- **`v_trans=false` is required for turbo*:** turbo cache layout differs from the transposed-V layout used by some attention paths. The cache constructor enforces this.
- **CUDA kernels actually run on HIP via translation:** `.cu` files are compiled with `hipcc` in HIP builds. Most "CUDA" changes are HIP changes too — verify the HIP target builds (`-DGGML_HIP=ON`) before assuming a fix is portable.
- **Vulkan turbo3 dequant has a tail-block ceil bug history:** see commits `06c74c8d1` and `8f1978f8f`. Subgroup-size handling is fragile across drivers; test on the actual target driver before claiming a Vulkan fix works.
- **`TURBO_LAYER_ADAPTIVE` is silently auto-enabled** for `turbo2_0` V — explicit `=0` is needed to disable. Other turbo types do not auto-enable adaptive.

## Active Work Surfaces

- `feature/turboquant-kv-cache` — the canonical TurboQuant cache feature
- `pr/vulkan-turbo3-rebase` — current Vulkan-side stabilization
- `docs/hot-cold-kv-design.md` — design for hot/cold KV tiering (Phase 1 pending implementation)

When extending the fork, place new long-form designs under `docs/` (the repo's documentation convention), name them by date prefix when they're proposal-shaped (e.g. `2026-05-15-feature-name.md`), and update this file's "Active Work Surfaces" list if the work spans more than a single PR.
