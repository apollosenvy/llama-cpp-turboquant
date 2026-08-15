# Hot/Cold KV Tiering on TurboQuant

**Status:** Design draft, 2026-05-15
**Author:** Heph (with Gary)
**Base branch:** `feature/turboquant-kv-cache` (current `pr/vulkan-turbo3-rebase` HEAD)

## Goal

Add a per-conversation tiered KV cache so a fixed-size **hot window** of recent
tokens lives in VRAM (read at GPU bandwidth) and **cold older tokens** spill to
host DRAM (paid via PCIe only when attention reaches back). Matches the actual
attention access pattern for chat workloads.

Composes with turbo3: a turbo3-encoded hot tier + a turbo3-encoded cold tier
fits ~5x more context at the same VRAM budget vs uncompressed all-VRAM.

User-facing flag: `--kv-hot-window-tokens N` (or `--kv-hot N`). Default `0`
disables tiering and the cache behaves exactly as today.

## Non-goals (Phase 1)

- LRU hotness; Phase 1 is strict position-based (oldest tokens are cold).
- Mixed types between tiers (hot type == cold type in Phase 1).
- Single-pass composite attention kernel; Phase 1 uses two-pass attention with
  fused softmax via online log-sum-exp.
- SWA-specific optimizations (no-op for cold tier when window < hot capacity).

## Architecture

### Per-layer cache layout (when tiering active)

```
        ┌─ k_hot[il] ─┐   GPU buffer, capacity = N_hot tokens, type = T_kv
        │             │
layer i ┤             ├─ holds positions [pos_min_hot ... pos_max] sliding window
        │ k_cold[il]  │   CPU pinned buffer, capacity = N_total - N_hot
        │             │   holds positions [0 ... pos_min_hot)
        └─────────────┘
```

`v_hot`, `v_cold` mirror. Buffer-type lookup via existing `ctx_map` machinery
in `llama_kv_cache::init` — already supports multiple backend types in one
cache.

### Cell metadata

Add `tier[i]` field (1 bit: 0 = hot, 1 = cold) to `llama_kv_cells`. Position
data stays as today; tier is derived from `pos[i]` vs `pos_max - N_hot`
threshold, but we cache the bit to avoid recomputing on every read.

When a new cell is appended:
1. Try hot tier first (always preferred for incoming tokens).
2. If hot full, evict the oldest hot cell to cold tier (FIFO by position).
3. Eviction = D2H async copy + free the hot slot.

### Attention path: two-pass with online softmax fusion

For each layer's attention:

```
  pass_hot  = flash_attn(Q, k_hot,  v_hot,  causal_mask_hot)   → (lse_h, out_h)
  pass_cold = flash_attn(Q, k_cold, v_cold, causal_mask_cold)  → (lse_c, out_c)
  lse_total = log_sum_exp(lse_h, lse_c)
  out_final = exp(lse_h - lse_total) * out_h + exp(lse_c - lse_total) * out_c
```

This is the standard split-K trick and is numerically identical to a single-
pass attention over the concatenated K/V. The cold pass dispatches a host→
device prefetch (cudaMemcpyAsync) for k_cold, v_cold before the kernel; the
prefetch overlaps with the hot pass kernel execution.

### Buffer transfers

- **Eviction (hot→cold):** `cudaMemcpyAsync` (or `hipMemcpyAsync`) issued on
  the cache's eviction stream. Non-blocking; sync at next attention layer
  boundary.
- **Cold read for attention:** allocate a per-step "cold scratch" GPU buffer
  sized to `N_cold * sizeof(kv_element)`. Prefetch into scratch before the
  cold-pass kernel. Free scratch at end of step (or reuse a pool).

PCIe traffic per decode step:
- Hot eviction: 0 unless a new token pushed an old hot token out (1 token / step max).
- Cold read: full `N_cold * KV_per_token` bytes per layer per step.

For Gemma 256k with N_hot=48k, N_cold=208k, ~50 KB/tok:
- Cold read per step per layer: 10.4 GB
- Across 28 layers: 291 GB / step
- At 40 GB/s PCIe Gen4 x16: 7.3 seconds / step ← unviable for naive impl

Mitigation: **batch cold pages across layers** (single large transfer once per
step instead of per layer). Cold KV for all layers is contiguous when laid out
with layer as outer dim. Drops to one PCIe transfer per step at ~7.3s for the
full cold cache, which is still bad but no longer per-layer.

**Real Phase 1 viability target:** SWA models where the kernel only reads the
recent window per step. Cold pages are accessed only when explicit
cross-window attention fires (rare, query-dependent). Full-attention models
get the feature but with the known slowdown; benchmarks will document.

## Phase 1 task breakdown

Each task is intended as one subagent invocation. File:line cites are best-
effort against current HEAD (`06c74c8d1`).

### Task 1: CLI flag + cparams plumbing
**Files:** `common/arg.cpp`, `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp`
- Add `--kv-hot-window-tokens N` (alias `--kv-hot N`) to arg parser
- Add `n_kv_hot` field to `llama_context_params`
- Thread it through to `llama_cparams` and onward to `llama_kv_cache` constructor
- Default 0 = disabled (no behavior change)
- Add a `--kv-cold-type` flag (defaults to `--kv-type`) — Phase 2 will diverge

### Task 2: Cells tier field
**Files:** `src/llama-kv-cells.h`
- Add `std::vector<uint8_t> tier` parallel to `pos`, `ext`, `seq`, `shift`
- Helper: `is_hot(i)`, `set_tier(i, t)`
- Update `add_cell`, `evict`, `shift` to maintain the field
- Add `n_hot()`, `n_cold()` count accessors
- Tier defaults to 0 (hot) when added; transitions to 1 (cold) only via explicit `set_tier(i, 1)` from cache layer

### Task 3: Dual-buffer cache allocation
**Files:** `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`
- Constructor: when `n_kv_hot > 0`, allocate `k_hot`/`v_hot` (GPU) AND `k_cold`/`v_cold` (CPU pinned)
- Reuse existing `ctx_map` for the CPU buffer type (`ggml_backend_cpu_buffer_type()`)
- `get_k(il)`, `get_v(il)` return the hot tensor by default; add `get_k_cold(il)` / `get_v_cold(il)`
- `memory_breakdown()` reports both tiers separately

### Task 4: Eviction logic
**Files:** `src/llama-kv-cache.cpp`
- New method `evict_hot_to_cold(seq_id, n_to_evict)`
- Picks oldest-position hot cells for `seq_id`, sets `tier=1`, issues
  `cudaMemcpyAsync`/`hipMemcpyAsync` from hot tensor row → cold tensor row
- Called from `cpy_k` / `cpy_v` when hot is full and a new cell needs to land

### Task 5: Two-pass attention dispatch
**Files:** `src/llama-graph.cpp` (around line 1959-1980)
- When `n_kv_hot > 0` for this cache, build TWO `ggml_flash_attn_ext` ops:
  one over hot K/V, one over cold K/V
- Build the online-softmax fusion using existing `ggml_log`, `ggml_exp`,
  `ggml_softmax`, etc. — see `examples/parallel/parallel.cpp` for prior art
  on splitting attention
- Single-pass fallback (`n_kv_hot == 0`) stays unchanged

### Task 6: Cold-page prefetch coordination
**Files:** `src/llama-kv-cache.cpp`, `ggml/src/ggml-cuda/cpy.cu`, `ggml/src/ggml-hip/...`
- Before each attention layer's cold pass, issue prefetch of all-layer cold
  KV in one DMA (layout: cold is [layer, cell, head_dim] with layer outer)
- Use a small pool of scratch GPU buffers (double-buffered) to overlap prefetch
  with hot-pass compute
- Sync point: hot pass completion

### Task 7: Layer-adaptive interop
**Files:** `src/llama-kv-cache.cpp` (lines 252-301 area)
- Verify `TURBO_LAYER_ADAPTIVE=7` still works when tiering enabled (per-layer
  Q8_0 vs turbo2 on a per-layer basis; tiering is orthogonal)
- Add test that combines `--kv-hot 48000` + adaptive mode

### Task 8: Tests + benchmarks
**Files:** `tests/test-kv-hot-cold.cpp` (new), `tools/llama-bench/...`
- Round-trip correctness: same input, all-VRAM vs tiered, output token-equivalence
- PPL parity: confirm tiered output matches all-VRAM within numerical noise
- Throughput bench: decode tok/s with and without tiering, at various N_hot values
- Memory bench: VRAM occupied across configs

### Task 9: Docs + flag help
**Files:** `docs/hot-cold-kv-design.md` (this file — update with Phase 1 results), `common/arg.cpp` help text
- Update this doc with measured numbers
- Add a section to README pointing at the flag

## Phase 2 (later, separate plan)

1. Mixed types per tier (`--kv-type f16 --kv-cold-type turbo3`)
2. LRU hotness instead of strict position
3. Composite single-pass attention kernel (avoids two-pass overhead)
4. SWA-aware: skip cold pass entirely when window < hot capacity
5. Per-sequence tier policy (chat vs RAG sequences want different windows)

## Open design questions

These need a call before Task 5 lands; the rest of the tasks don't depend on
them:

1. **Prefetch granularity.** All-layers-once vs per-layer-as-needed? All-once
   is simpler but pessimizes pipelining; per-layer overlaps better but
   complicates the dependency graph.
2. **Cold scratch buffer sizing.** Static "max possible cold" allocation, or
   dynamic per-step sized to actual cold pages touched? Dynamic is friendlier
   to VRAM-budget users but adds allocator churn.
3. **What happens on sequence shift / defrag?** Cold-tier shifts require D2H
   copies too. May warrant disabling defrag for tiered caches in Phase 1.

## Backout

Removing the feature = setting `n_kv_hot = 0` in cparams. The dual-buffer
allocation, eviction, and two-pass paths are all guarded by `n_kv_hot > 0`
checks; with the flag off, behavior is byte-identical to current main.
