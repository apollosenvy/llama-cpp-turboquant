# turbo3 arch gaps: deepseek4 MLA + gpt-oss head_dim 64

**Request (Gary, 2026-08-16):** turbo3 KV is the house default (ruling: `feedback_turbo3_default`).
Rebase, build locally, patch until deepseek4 (MLA) and gpt-oss serve on turbo3, then PR upstream (TheTom).

**Symptoms (against be5b16793, deployed 2026-08-16 18:25):**
- deepseek4 + `-ctk/-ctv turbo3`: `GGML_ASSERT(tensor->data != NULL && "tensor not allocated")`,
  ggml-backend.cpp:333, at context creation, ANY KV placement (GPU or `--no-kv-offload`).
- gpt-oss + turbo3: `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)`, ggml.c:3728 (`ggml_reshape_3d`),
  during load, reproduced at 8k ctx CPU-only. Not memory. Not ctx size.

**Hypothesis to verify first:** turbo3 block = 128 elems; rows must divide by block.
gpt-oss head_dim = 64 (64 % 128 != 0). MLA latent row = 576 (576 % 128 = 64). One root, two symptoms.
Prior art: `backup/pr-vulkan-turbo3-rebase-pre-parity-2026-04-24` has "dequant_turbo3_0 tail-block ceil";
`feature/instella-moe-turbo` has "hip: enable MLA turbo FA for D=576/640".

## Todo
- [x] Recon: repo/branch/build provenance (branch `sync/upstream-2026-08-15`, builds current)
- [ ] Verify block-size hypothesis in ggml-common.h / ggml-turbo-quant.c
- [ ] Map KV alloc path for MLA shape (why tensor->data NULL instead of clean unsupported error)
- [ ] Check what feature/instella-moe-turbo + vulkan backup branch already solved; reuse, don't rewrite
- [ ] Rebase onto origin/master (TheTom 57f6b9365) — keep PR#5 lineage intact
- [ ] Patch: tail-block (row % 128) support in quant/dequant + KV size math, CPU + HIP FA paths
- [ ] Gates per STYLE.md (load STYLE.md BEFORE writing tests; plant failures first):
      test-backend-ops for turbo3 at ne0=64 and ne0=576; planted-failure run;
      serve gpt-oss-20b turbo3 small ctx; serve deepseek4 turbo3 small ctx GPU KV first;
      output-quality sanity vs f16 (same prompt, coherent + no repetition collapse)
- [ ] Fleet: flip deepseek/20b/120b units to turbo3 at policy ctx, re-run sweep gate
      (deepseek DRAM-KV first-inference test = the config family that kernel-panicked; run with
      crash capture in mind, small ctx ladder, journal watched)
- [ ] PR to TheTom once gated; branch off clean, self-contained commits

## Log
- 2026-08-16 21:2x: campaign opened. Deployed builds = be5b16793 (both build/ and build-sync/).
  Branch tip 54e3cd324 is one ahead (json-schema commit, upstream cherry).
- Branch `fix/turbo3-arch-gaps`. Merged TheTom 57f6b9365 (conflict: his FA pool bypass
  superseded by upstream's graph-carved f16 temps; kept ours, documented).
- FIX 1 (26402c331): deepseek4+turbo crash at context creation = missing no_alloc guard on
  the turbo rotation re-upload in llama_kv_cache::clear(), reached by dsv4 ctor's
  clear_compressed during the fit-params dry run. NOT an MLA-layout gap: the 576->640
  pad machinery already existed (Gary's 7dbcfb88c). deepseek4 turbo3 now loads (CPU-verified).
- FIX 2 (e494a4ed0): gpt-oss reshape assert = V-unpad blocks in all three build_attn
  overloads reshaped by n_head_kv; output rows are per QUERY head. Latent until the first
  misaligned-V model (gpt-oss 64->128, 64q/8kv). Now derived from cur->ne[0]/padded_v_head.
  20b serves coherently on turbo3 GPU (first ever).
- QUALITY GATE FAILED: wikitext-12-chunk PPL — muse control f16 5.299 vs turbo3 5.478 (+3.4%,
  the normal turbo3 envelope) but gpt-oss f16 1624 vs turbo3 2337 (+44%). Real defect in the
  new path. Triangulating: K-only turbo3, V-only turbo3, q8_0/q8_0 control (running).
- Instrument notes: absolute gpt-oss PPL on raw wikitext is degenerate (harmony model);
  only paired deltas are meaningful. Back-to-back GPU ppl runs need a sleep between them
  or the second process starts before VRAM teardown and dies silently.
- INSTRUMENT RETRACTION: the +44% PPL delta was invalid evidence (chunk1 f16=9322 vs
  turbo3=6040-6651 cross-backend — quant "beating" f16 at degenerate PPL = noise regime;
  Pensive atom corrected, 01M0751BGT...). Valid instrument = per-token KL vs own-f16 logits.
- KL VERDICT (wikitext probe, chunks 3): muse turbo3 median KLD 0.018, same-top 91.5%
  (healthy envelope). gpt-oss turbo3 median KLD 0.666, same-top 35.4% — 37x envelope.
  DEFECT CONFIRMED with calibrated instrument. Model stays locally fluent (said Paris fine)
  while picking a different top token 2/3 of the time.
- Semantics established by source read: turbo3 dequant does NOT unrotate (rotation-open);
  cache holds rotated data; Q pre-rotated graph-side; single graph-side inverse WHT on
  attention output; CPU FA consumes K via vec_dot (rotated ✓) and V via to_float (rotated ✓).
  All stages consistent at reading altitude — defect is numeric or in a path divergence.
- Localization running: (a) K-only vs V-only turbo3 KL (q8_0 counterpart, chunks 1);
  (b) muse CPU turbo3 KL control (was only ever validated on GPU).
- Mixed KV-type FA combos run ~25x slower (fallback), so their "silent failures" earlier
  were timeouts, not crashes.

## RESOLUTION of the gpt-oss K-side KL defect (2026-08-17 ~00:15)

NOT an implementation bug. Full evidence chain:
- Reproduced in a CPU test tube ONLY with real layer-0 activations (COMMON_DEBUG_DUMP_DIR
  patch in common/debug.cpp + PROBE_LOAD_DIR in tests/probe-pad-k-fa.c): NMSE 0.17-0.36
  vs f16-K; synthetic data healthy at any outlier scale.
- Final diagnostic: graph rotations exactly orthogonal on real data (ref==graphrot logits
  to 3 decimals); K decode fidelity AT SPEC (cos 0.983-0.989 = turbo3's designed ~1.5%
  angular error); logits nonetheless scrambled by +-2.
- Mechanism: gpt-oss K rows carry sink-bias outliers (|K| to 23, row norms ~30, Q norms
  ~10). In-spec quantization noise x norm product / scale(8) = multi-logit noise, and
  softmax exponentiates it. Even q8_0-K degrades this model (KLD 0.055 = 3x muse's whole
  turbo3 budget): the K side is hypersensitive to ANY quantization. Model-class accuracy
  cliff, not a code defect.
- Gain-corrected (dot-unbiased) norm was implemented, measured (moves NMSE third decimal),
  and REVERTED: it fixes a real 1.2% shrinkage bias that is not the failure mode here;
  candidate for a separate measured experiment.
- PAD PATH FULLY EXONERATED -> deepseek4's 576->640 K padding carries no taint.
- gpt-oss disposition: tool-class model (Gary). Options if it ever matters: K=q8_0
  (KLD 0.055), K=turbo4 (measuring now), or documented f16-K exception to the ruling.
