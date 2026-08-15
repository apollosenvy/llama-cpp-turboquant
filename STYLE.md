# llama-cpp-turboquant Engineering Style

TurboQuant is the KV cache compression layer that lets a 7900 XTX hold a 256k-token Qwen3.5-35B context where stock llama.cpp tops out around 48k. It is the runtime backbone for Gary's Hearth platform, the local-inference path for Moirai planner/coder/reviewer slots, and the substrate for the hot/cold KV tiering work tracked in `docs/hot-cold-kv-design.md`. When TurboQuant misbehaves silently — a kernel that "works" but skews attention scores by 2% — every downstream user (Forge SDK sessions, Moirai chains, OpenCode loops, Hearth chat) gets quietly worse output and may never realize it. Numerical correctness is not optional.

This is not a court-grade artifact like Sentinel. There is no DOPSR clearance, no PSAP customer, no audit chain. But TurboQuant sits closer to Sentinel than to a daily-ops tool because the failure mode that matters — silent numerical drift across CUDA / HIP / Vulkan / Metal backends — is invisible to operators and corrupts every inference. The bar follows from that: rigor on the numerical and backend-equivalence layers; pragmatism everywhere else, since we inherit the upstream llama.cpp codebase and cannot rewrite it.

This document is the standard a TurboQuant change must meet to land on `feature/turboquant-kv-cache` or any `pr/vulkan-*` sibling. Heph reads it on SessionStart under `/home/aegis/Projects/llama-cpp-turboquant` and refuses to help violate hard rules without explicit override. When work targets upstream (`pr/*` heading to `ggml-org/llama.cpp`), the upstream `AGENTS.md` and `CONTRIBUTING.md` apply *in addition* to this file.

---

## Core Principles

1. **The CPU reference is the truth. Every backend must match it.** `ggml/src/ggml-turbo-quant.c` defines the canonical encode/decode. CUDA, HIP, Vulkan, and Metal implementations are *approximations of that reference within numerical tolerance*. When a kernel disagrees with the reference, the kernel is wrong. The quality gate is the contract.

2. **Silent numerical drift is the worst sin.** A turbo3 kernel that produces "plausible-looking" attention scores but is off by 1% will pass smoke tests, pass perplexity within noise, and silently degrade every user's inference quality for months. The quality gate exists to catch this; bypassing it (skipping `turbo-quality-gate.sh`, weakening the 5% PPL threshold, ignoring a `test-backend-ops` divergence) is the failure mode this project is built to prevent.

3. **Backend equivalence is sacred.** A change to the CPU path that does not update the CUDA path is a bug. A new turbo type that ships without Vulkan, HIP, and Metal instances is a regression for every user of the missing backend. The quintet (CPU + CUDA + HIP + Vulkan + Metal) is the invariant.

4. **GPU API returns are checked or asserted, always.** `cudaMalloc`, `cudaMemcpyAsync`, `hipMalloc`, `hipMemcpyAsync`, `cudaMemcpyToSymbol`, every kernel launch — wrap in `CUDA_CHECK` / `GGML_CUDA_CHECK`. The `turbo-quant.cuh:179-241` calibration path does this correctly today; new code matches that bar. Unchecked GPU API calls are how we get silent allocation failures that produce garbage outputs.

5. **Kernel assertions are armor against template instantiation drift.** Templated CUDA/HIP kernels are easy to instantiate with the wrong type pair, the wrong group size, the wrong head dimension. Static asserts on template params + runtime asserts on tensor shapes catch the bugs that show up as "works on D=128, garbage on D=256." NASA Power of Ten #5 applies to kernel code too.

6. **Upstream is a moving target. We don't fight it.** llama.cpp main moves fast and rewrites things we depend on. Periodic `master` syncs (see `67559e580 Upstream sync to b8871`) bring our fork forward; we do not get clever about diverging permanently. Our value-add is the turbo* types and their kernels, not a fork of the whole world.

7. **Quality gate before push.** No commit landing on `feature/turboquant-kv-cache` or any `pr/vulkan-*` branch ships without `scripts/turbo-quality-gate.sh` having passed locally on the change. Heph runs it; if it fails, the change does not land.

---

## Hard Rules

### Numerical correctness

- **Every new turbo type ships with a CPU-vs-GPU equivalence test in `test-backend-ops`.** Currently `tests/test-turbo-quant.c` is a CPU-only round-trip test. The CPU↔GPU cross-check happens inside `test-backend-ops` but coverage of turbo types is **thin** — searching the tests directory shows no dedicated turbo backend-equivalence file. This is a known gap; any new kernel must close it for that kernel.
- **PPL regression > 5% on `scripts/turbo-quality-gate.sh` is a release blocker.** The baseline is `q8_0` PPL = 6.111 on Qwen3.5-35B-A3B over wikitext-2-raw. If your change pushes turbo3 PPL above 6.417, do not push. Find the bug first.
- **Speed regression below 0.95x of `q8_0` at 4K context is a release blocker.** Same script. A perf regression on the turbo path means cold context is now slower than uncompressed — defeats the point of the project.
- **Centroid tables are calibration artifacts, not magic numbers.** The Lloyd-Max centroids in `ggml/src/ggml-cuda/turbo-quant.cuh:24-42` (the `-0.133462f, -0.039994f, ...` literals) come from offline optimization against Gaussian-distributed K vectors. Any change to a centroid table requires regenerating from the offline calibration tool and updating *every* backend's copy (CUDA + Metal + Vulkan precomputed data) in the same commit.
- **The WHT is a unitary rotation.** `<Q_rot, K_rot> == <Q, K>` is the invariant that makes turbo3 attention valid. Any kernel that touches the rotation step must preserve this. The `inv_sqrt` constants at `turbo-wht.cu:83-85` (`0.08838834764831845f = 1/sqrt(128)` etc.) are the normalization factors that make the WHT unitary — never edit them without re-deriving.

### Backend equivalence

- **Every turbo* code path has five implementations or a documented gap.** CPU (the truth) in `ggml-turbo-quant.c`; CUDA in `ggml/src/ggml-cuda/turbo*.{cu,cuh}`; HIP via the same .cu files compiled with hipcc; Vulkan shaders in `ggml/src/ggml-vulkan/`; Metal in `ggml/src/ggml-metal/turbo-*.h`. A new feature missing one of the five is filed as a tracked backend gap with an issue number in the commit message.
- **The four-bit centroid table TODO at `ggml-turbo-quant.c:577` is the canonical example of a documented gap.** "TODO: add proper 4-bit centroid table to C code (currently only in Metal)" — this is the right *form* of a gap acknowledgement. New gaps follow this pattern: TODO with the exact backend missing and the workaround in place.
- **`test-backend-ops` is the cross-check that catches divergence.** Run it after any change to a turbo kernel, on at least the two backends you build for (`-DGGML_HIP=ON` and CPU). For Vulkan changes, run the Vulkan backend test specifically — Vulkan turbo3 has a known subgroup-size fragility (commits `06c74c8d1` and `8f1978f8f` are prior dequant tail-block + signs-packing fixes).
- **HIP is not "CUDA but for AMD."** Treat HIP as a first-class build target. Code that compiles for CUDA but fails for HIP gets caught only by the HIP build (`cmake -B build-hipvk -DGGML_HIP=ON`). The fork's primary target *is* HIP (7900 XTX). Verify HIP builds before assuming CUDA-side changes are portable.
- **Vulkan shader changes get tested on the actual deployment driver.** Shader behavior diverges across mesa / RADV / Nvidia / lavapipe drivers, especially around subgroup operations. "It works on my driver" is not a merge criterion.

### GPU API error handling

- **`CUDA_CHECK` / `GGML_CUDA_CHECK` wraps every `cudaXxx` / `hipXxx` API call that returns a status.** Unwrapped calls are bugs. `turbo-quant.cuh:179-241` is the reference pattern. Search for `cudaMemcpy[A-Z]` and `hipMemcpy[A-Z]` not preceded by a checker macro before pushing.
- **Kernel launches are followed by a sync + check in debug builds.** A silent kernel failure produces zero-filled or stale output, which downstream kernels happily process. `GGML_CUDA_DEBUG` builds insert the sync; release does not, but a kernel that fails silently in release was caught in debug.
- **No `printf` from kernels except behind a debug flag.** Kernel printf serializes warps and breaks perf. `turbo-wht.cu` and `turbo-innerq.cu` are clean today; keep them that way.
- **CUDA memory allocations are checked for `cudaErrorMemoryAllocation` and reported with the requested size.** A failed allocation message that does not include the requested byte count is debugger-hostile.

### Assertions

- **Templated kernels assert their template params.** `static_assert(group_size == 32 || group_size == 64 || group_size == 128, "...")` at the top of a templated kernel catches misinstantiation at compile time. `turbo-wht.cu` has 6 asserts across the file; the inner-quant kernel at `turbo-innerq.cu` has 0 — that is a hardening target.
- **Tensor shape assertions are required at every kernel entry point.** Wrong head_dim, wrong group_size, wrong batch dim — these are the bugs that show up as "the test case passes, the real model crashes." `GGML_ASSERT(tensor->ne[0] % 128 == 0)` is cheap and catches a real bug class.
- **Assertions stay enabled in release.** A failed `GGML_ASSERT` in production is a logged abort. A silent numerical corruption is a silently degraded model output. Always prefer the former.

### Naming and structure

- **The fork inherits god modules from upstream; we do not make them worse.** `ggml-vulkan.cpp` is 17,051 lines, `ggml.c` is 7,839, `llama-model.cpp` is 9,467. These are upstream-shape and not on us to fix. *Our* additions to those files stay small and tagged with a comment block identifying them as turbo-related so the next upstream sync can resolve conflicts cleanly.
- **Our own new files cap at 600 lines.** This is enforced going forward. `src/llama-kv-cache.cpp` at 2,765 lines is upstream-shape; any new file we add (e.g. a future `llama-kv-cache-tiered.cpp`) starts under 600.
- **Turbo-related additions to existing files are bracketed.** A comment header like `// --- TurboQuant: WHT rotation tensors ---` above an inserted block and `// --- end TurboQuant ---` after it makes upstream rebases trivial. The current per-context turbo tensor allocation at `src/llama-kv-cache.cpp:349-359` is a good example; the layer-adaptive precision block at `src/llama-kv-cache.cpp:252-301` could use this treatment.
- **One concept per file.** `turbo-wht.cu` is the WHT kernel, period — no inner-quant code, no cache management. `turbo-innerq.cu` is the inner-quant kernel, period. Keep them split.

### Testing

- **`turbo-quality-gate.sh` is the ship gate.** No exceptions. The script's two checks (PPL within 5% of q8_0, speed > 0.95x q8_0) are the operational contract.
- **`test-backend-ops` runs on every kernel change.** This is upstream's mechanism for catching backend divergence; we lean on it heavily because we add backend implementations of new types.
- **CPU-only round-trip tests for each turbo type.** `tests/test-turbo-quant.c` covers turbo3 and turbo4 round-trip; a new turbo type ships its round-trip test in the same commit. WHT inverse must be applied before comparison (see existing test's Test 1 comment for the rationale).
- **Backend equivalence test per type, per FA pair.** When adding a new K/V type pairing (search `fattn-vec-instance-turbo*` for current set), add a test that runs the same attention through CPU reference and the GPU backend, asserts cosine similarity > 0.999 on the output. Mock test data is fine; the bar is the kernel's own correctness, not realistic model data.
- **No live-model tests in PR CI.** Real model evaluation belongs in the quality gate (which runs Qwen3.5-35B-A3B locally on Aegis). PR-level CI runs CPU and GPU unit tests against synthetic inputs.

### Dependencies and upstream sync

- **No new third-party dependencies.** Upstream's policy is strict on this; we inherit it. Anything we add ships with the codebase or is an OS package.
- **Periodic upstream syncs land as a single commit titled `Upstream sync to <upstream-tag>`.** Pattern set by `67559e580`. Merge conflicts get resolved in that commit; turbo-specific code stays bracketed (see naming rules above) so the resolution is mechanical, not interpretive.
- **Diverging from upstream is debt.** Every divergence (a function signature we changed, a header we added, a build flag we modified) makes the next upstream sync harder. New divergences require a commit message explaining why upstream's shape didn't fit.

### Branch hygiene

- **`pr/*` branches follow upstream AGENTS.md.** AI may assist; humans author and defend. Heph runs in advisory-only mode on those branches: review, suggest, never author commits.
- **`feature/*` and `experiment/*` branches follow this STYLE.md.** AI authorship is fine; quality gate still applies.
- **`backup/*-YYYY-MM-DD` branches are landmarks, not work surfaces.** Do not edit, do not rebase. Delete only after explicit confirmation that the snapshot is no longer needed.
- **Force-push is forbidden on `feature/turboquant-kv-cache`.** It is the canonical fork-feature branch; downstream consumers (Hearth, future fork users) track it. Rewrites land on a sibling branch and merge cleanly.

---

## Anti-patterns we will not ship

Each has a name so it can be flagged in review.

- **Silent kernel.** A CUDA/HIP kernel that runs to completion on bad inputs and produces wrong output without asserting. The fix is shape/dtype asserts at entry.
- **Unchecked GPU API.** A `cudaMemcpy*` or `hipMalloc` without `CUDA_CHECK` / `GGML_CUDA_CHECK` wrapping it.
- **Backend gap.** A new turbo* type or kernel that ships missing one of CPU / CUDA / HIP / Vulkan / Metal without a TODO marker and a tracked issue.
- **Quality-gate bypass.** Pushing a change that touches turbo kernels without `turbo-quality-gate.sh` having run and passed. "It only changes the comments" is not an excuse if any executable line moved.
- **PPL drift > 5%.** A change that pushes turbo3 PPL above `q8_0 * 1.05`. Stop, debug, do not push.
- **Magic centroid.** Editing a Lloyd-Max centroid value directly without regenerating from the calibration tool and propagating to all backends in one commit.
- **WHT normalization fudge.** Changing the `inv_sqrt_128 = 0.0883...` constants without re-deriving from the WHT unitarity proof. These are not free parameters.
- **Stringly-typed dispatch.** A function that branches on a string type name instead of `GGML_TYPE_TURBO*` enum.
- **Permanent divergence.** A fork-only change that breaks the periodic upstream sync without a written justification.
- **Force-push on `feature/turboquant-kv-cache`.** Downstream tracks it; do not rewrite history.
- **God-module growth.** Adding more than 200 lines of fork-specific code to an already-god upstream file without bracketing or extracting.
- **Template misinstantiation.** Instantiating a templated kernel with an unsupported (group_size, head_dim, type) tuple. Catch at compile with `static_assert`, not at runtime with a crash.

---

## Enforcement

The fork does not have a project-specific CI pipeline beyond upstream's. The enforcement model is:

1. **Heph reads this on SessionStart** when CWD is under `/home/aegis/Projects/llama-cpp-turboquant`. Hard-rule violations are flagged before code is written.
2. **`scripts/turbo-quality-gate.sh`** is the canonical ship gate. Two invariants: PPL within 5% of q8_0, speed > 0.95x q8_0 at 4K context. Failure blocks push.
3. **`test-backend-ops`** is the cross-backend equivalence check. Run on every kernel change, on at least CPU + the active GPU backend (HIP for Aegis hardware).
4. **`test-turbo-quant`** is the cheap CPU-only round-trip gate. Runs in seconds; no excuse to skip.
5. **Pre-commit local check** (future, modeled on Sentinel's hook in `~/.claude/hooks/`): the cheapest rules — `cudaMemcpy[A-Z]` outside `CUDA_CHECK`, missing template `static_assert`, new files over 600 lines, edits to centroid tables without companion-backend changes in the same commit.
6. **Upstream sync gating.** Before any `Upstream sync to <tag>` commit, verify `test-backend-ops` and `turbo-quality-gate.sh` pass on the post-sync state. A regression introduced by the sync is fixed in the sync commit, not deferred.

---

## On scope and humility

This document is the bar we are choosing. The bar may move up over time as we hit new failure modes and write the lessons in. The bar should not move down. If a rule starts hurting more than it helps, the move is to debate it and edit this file, not to silently ignore it.

Engineering philosophy is only useful when it is written down and enforced. Otherwise it is wishful thinking that survives one stressful sprint and evaporates.

We do not own the codebase. We own a fork of it. That means our discipline matters most on *our* code and at *our* seams — the turbo* kernels, the cache integration, the hot/cold tiering. Upstream's many imperfections (god modules, sprawling switches, inconsistent style) are theirs to fix. Our additions hold a higher bar.

---

## Operational SLAs

TurboQuant is the runtime substrate for Gary's local LLM work. These budgets are the operational contract a release must meet on Aegis hardware (7900 XTX, gfx1100).

### Numerical accuracy
- **Perplexity:** turbo3 within 5% of `q8_0` on the quality-gate model (Qwen3.5-35B-A3B over wikitext-2-raw). Currently baseline `q8_0 = 6.111`, so turbo3 ceiling is `6.417`. Below this, do not push.
- **Backend cosine similarity:** any GPU backend's attention output vs CPU reference, cosine > 0.999 on random-input adversarial tests.
- **Round-trip MSE:** turbo3 CPU round-trip MSE < `5e-4` on Gaussian-distributed K vectors of dim 128 (`test-turbo-quant.c` baseline).

### Speed
- **Context-scaling ratio:** turbo3/q8_0 > 0.95 at 4K context, fully GPU-offloaded, on Aegis 7900 XTX.
- **Decode tok/s:** turbo3 within 10% of f16 decode rate on the quality-gate model at 32K context. Above 32K, turbo3 must win (because f16 OOMs).
- **Prefill tok/s:** turbo3 within 20% of f16 prefill rate at 4K. The encode overhead is real; budget it.

### Memory
- **VRAM at 256K context, Qwen3.5-35B-A3B Q8 model, turbo3 K+V:** under 24 GB (fits 7900 XTX). Baseline ~22 GB weights + ~2.4 GB turbo3 KV. Currently validated; regression blocks ship.
- **No GPU memory leak across 1000 decode steps.** Measured via `rocm-smi --showmemuse` before and after; drift under 10 MB.

### Backend coverage
- **Every turbo type (turbo2/3/4) has working CPU + HIP + CUDA + Vulkan + Metal,** or a TODO with the exact gap named and the workaround documented (e.g. `ggml-turbo-quant.c:577` for the 4-bit Metal-only centroid).
- **`test-backend-ops` passes on CPU + HIP + Vulkan** on every push to `feature/turboquant-kv-cache`.

### Build
- **Fresh `cmake -B build-hipvk -DGGML_HIP=ON -DGGML_VULKAN=ON -DAMDGPU_TARGETS=gfx1100`** completes in under 5 minutes on Aegis with ccache warm.
- **No new warnings** from the HIP build. `-Werror` is too aggressive given upstream noise, but new warnings introduced by our changes are a release blocker.

### Quality gate runtime
- **`turbo-quality-gate.sh` completes in under 4 minutes** on Aegis at 7900 XTX MCLK 1350 / UV -50mV / 402W cap. Slower means the perf-regression check is itself drifting and needs investigation.

### Upstream sync
- **Sync cadence:** at minimum monthly, targeting the latest upstream tag. Longer drift makes the merge resolution worse.
- **Sync commit must include:** all backend-equivalence tests passing, `turbo-quality-gate.sh` passing, plus a one-paragraph commit-message note covering any non-trivial conflict resolution and any new upstream behavior that affects turbo kernels.

---

Last revised: 2026-05-15. Owner: Gary + Heph.
