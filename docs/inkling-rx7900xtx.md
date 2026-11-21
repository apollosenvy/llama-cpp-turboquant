# Inkling-2-Mini on one Radeon RX 7900 XTX

This branch makes the HIP (ROCm) backend run Inkling-2-Mini about three times faster on a single
RX 7900 XTX, with the same GGUF and the same sampling. It is built on llama.cpp at `29b52d398`
(`model : Inkling 2 Mini`) plus the TurboQuant `turbo3` KV cache type.

| | Before | After |
|---|---|---|
| Decode, empty context | 29 tok/s | 103 tok/s |
| Decode at 33K context | | 94 tok/s |
| Decode at about 100K context | 23.5 tok/s (90K) | 80 tok/s (104K) |
| Prefill, 8K prompt, 256K context | about 140 tok/s | 1,040 tok/s |
| Prefill, 104K prompt | about 130 tok/s | 727 tok/s |
| GPU kernel launches per token | 7,677 | 1,075 |

Model: `Inkling-2-Mini-IQ4_NL.gguf` (26B parameters, 24 layers of d_model 2304, 128-expert MoE with 8
active, layers 6 to 17 looped three times, so 48 layer passes per token). Card: RX 7900 XTX, 24 GB,
ROCm 10.1.0. "Before" is the same tree without this series, as it was deployed: 512K context, four expert
layers on the CPU, `-ub 16`. "After" runs 256K context, all experts on the GPU, `-ub 128`.

Output is not token-identical to the unmodified backend, and quality is the same. Against the unmodified
tree on wikitext-2 (context 2048, `llama-perplexity --kl-divergence`): perplexity 6.199 against 6.217
± 0.19, mean KL divergence 0.018, same top token 94.2%. That divergence is the model's own response to
float rounding: scaling the Q, K, V and bias rows by one ulp (`GGML_CUDA_HEPH_PERTURB=1e-7`) inside this
build moves its logits by the same 0.018, and by no more at 1e-4. See Quality.

## Where the time went

A decoded token reads about 3.6 GB of weights. The card reads 890 GB/s at best, so the floor is about
4 ms. The unmodified backend took 31.5 ms, and only 6.5 ms of that was in the weight kernels. The rest
was 7,200 tiny kernels per token: the model's graph is thousands of small element-wise nodes, and every
kernel launch costs about 3 microseconds on this stack however little it does.

Three measured facts shaped the work.

1. **Launch count.** Each launch costs roughly 1.6 us of gap plus at least 0.85 us of kernel time, inside
   HIP graph replay. 7,677 launches is most of a token.
2. **Load latency.** A kernel thread that loops over memory pays about 0.1 us per dependent loop iteration,
   whatever the data size. A 256-thread kernel doing nine iterations over five small vectors took 10.7 us.
   Issuing every load of a thread at once and keeping the values in registers took it to 2.6 us. Kernels
   here have no per-thread loop over memory: 1024-thread blocks, unrolled loads, or one wave per output
   element sharing a short dot product.
3. **Slow grid sizes.** A launch whose total thread count is a multiple of 98,304 (96 compute units x 1024),
   or up to 1,536 below one, takes 6 to 9 us longer, for the first four multiples. 384 blocks of 256 threads
   is exactly that, and so is 3072 rows at 8 rows per block. Launches that would land there are padded by
   one empty block. `tests/inkling-fusetest` case `mvrows` under `rocprofv3` reproduces it.

## What changed

**Fused kernels** (`ggml/src/ggml-cuda/inkling-fused.cu`, matchers in `ggml-cuda.cu`). Chains of small nodes
are recognised in the graph and run as one launch: row clip, the depth-attention mix, the loop control,
SiTU, the short convolution with its state store, the relative-position bias and mask, MoE weight
normalisation.

**A staged executor inside the graph.** The model's graph builder (`src/models/inkling2.cpp`) orders its
nodes so each stage of a layer is one contiguous run, and the backend runs each run as a single launch
with a job table. Node order is the only thing that changes in the graph. Stages of a layer pass:

- controls: one gate launch and one apply launch for the Q, K and V inputs together
- norms: `rms_norm * weight` for the three inputs in one launch, which also writes their q8_1 form
- projections: the Q, K, V, relative-bias, sink and gate matvecs as one launch (`mmv_q8_0_multi`),
  21 MB in 27 us
- post-projection: Q clip and rotation, K and V convolution, clip or norm, rotation, conv state store and
  the turbo3 KV cache store, and the relative-bias clip, in one launch (`row_jobs_f32`)
- bias mask, flash attention
- output: rotation, gate sigmoid and q8_1 for the output projection in one launch
- MoE: router (top-k and weight normalisation in one kernel), up projection, SiTU with q8_1 output,
  down projection, then weighted sum, convolution, residual add and state store in one launch

A looped layer pass went from about 160 launches to about 22. Reshape views keep each stage's inputs
allocated until the stage has run, so the allocator cannot place one job's output on another job's input.

**A graph op for the bias mask** (`GGML_OP_REL_BIAS_MASK`, CPU and CUDA/HIP). The per-head attention bias
was built by a chain that held two float tensors of `n_kv x n_tokens x n_head` at once. The op writes one
half-precision buffer that every layer reuses. Reserved compute memory per micro-batch token at 256K
context went from 50 MB to 13.5 MB, which is what allows `-ub 128` instead of 32.

**An attention kernel for the turbo3 cache** (`ggml/src/ggml-cuda/fattn-turbo3.cuh`). For one decoded token the
generic vector kernel is bound by instruction count, not by memory: its time does not move with the number of
parallel blocks, and its assembly shows about 2,800 instructions per 128-key tile and wave, of which the 3-bit
decode is a third. The rest is 64-bit address arithmetic for byte loads, shuffles through LDS and waits. The
dedicated kernel loads whole code words off a scalar base, decodes 16 values per word (values k, k+4, k+8, k+12
share a bit offset, so one shift and one mask index four of them), sums across the 8 lanes of a key and broadcasts
the softmax weights in registers (DPP), and applies the K and V norms once per key. From 8K keys up the three
query heads that share a K/V head are handed over as three columns of that head, so K and V are read and decoded
once for the three.

| kernel time per call | 1K keys | 8K keys | 65K keys |
|---|---|---|---|
| generic vector kernel | 15.8 us | 41 us | about 300 us |
| dedicated, one column | 8.0 us | 20 us | 166 us |
| dedicated, three columns | 10.2 us | 16.5 us | 113 us |

**Lane swaps in registers** (`ggml_cuda_shfl_xor` in `common.cuh`). `__shfl_xor` is a `ds_bpermute`, which goes
through LDS. For xor masks 1, 2, 4 and 8 the two lanes sit in the same row of 16, where DPP swaps them in
registers. Same lanes, same results (every test output is bit-identical), and every reduction in the backend gets
cheaper: 99.5 to 100.6 tok/s.

**Lone Q8_0 matvecs on the grouped kernel.** The 3072 x 2304 output projection and lm_head went through the generic
matvec kernel at 75% and 79% of the card's bandwidth. The grouped-projection kernel with a single job computes the
same partial sums in the same order, so the result is bit-identical, and it is faster for both: 100.2 to 103.2
tok/s. `GGML_CUDA_HEPH_MMV1_ROWS=<min rows>` moves the limit (default 1024), 0 keeps them on the generic kernel.

## Build and run

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
```

```bash
cmake --build build --target llama-server -j
```

```bash
SMITHY_CONFIG=docs/inkling-rx7900xtx-smithy.json build/bin/llama-server -m Inkling-2-Mini-IQ4_NL.gguf \
  -c 262144 -np 1 -ngl all -ncmoe 0 -b 2048 -ub 128 --fit off -fa on -ctk turbo3 -ctv turbo3 --jinja
```

`-ub 256` allocates at 256K context but overcommits a 24 GB card and runs about 40 times slower. Check
used VRAM, not only that the server started.

Switches, read from the environment at start:

- `GGML_CUDA_HEPH_FUSE=0` turns every fusion in this series off.
- `GGML_CUDA_HEPH_NOFUSE=a,b,...` turns named ones off: `clip`, `mix`, `control`, `controlraw`, `ctrlgroup`,
  `situ`, `conv`, `convst`, `moew`, `relbias`, `relpad`, `relfill`, `rot`, `qcache`, `mmv`, `normq`, `normgroup`,
  `postproj`, `outgate`, `route`, `ffntail`.
  `t3attn` falls back to the generic attention kernel.
- `LLAMA_GQA_COLS=<keys>` sets the cache length from which query heads are grouped as columns (default 8193,
  0 never).
- `GGML_CUDA_DUMP_NODES=1` prints every evaluated node and what was fused, once per captured graph.
- `GGML_CUDA_HEPH_PERTURB=<eps>` scales the post-projection rows by 1 + eps (sensitivity measurements only).
- `GGML_CUDA_FATTN_PB=<n>` overrides the number of parallel blocks of the vector attention kernels (tuning only).

## Tests

`tests/inkling-fusetest` builds each chain as a small graph exactly as the model does, runs it on the GPU
and on the CPU backend, and compares the outputs. 107 cases, single and multi token.

```bash
tests/inkling-fusetest/build.sh build && tests/inkling-fusetest/fusetest
```

- A case fails if no fusion fired, so a matcher that quietly stops matching is caught.
- Every graph is evaluated twice with different inputs, which catches state carried between evaluations.
- `FUSEREPEAT=200` re-evaluates each graph 200 times and requires bit-identical GPU output.
- `FUSEDUMP=<dir>` writes every output as raw floats, to compare two builds bit for bit. A change that claims to
  be exact (the lane swaps above) is checked this way; the CPU comparison is too loose for that.
- `kbench.sh <case>` prints per-kernel times for one case from a `rocprofv3` trace. A case only covers a
  kernel if the kernel shows up there. `kgrid.sh <case> <kernel>` adds grid, block size and register count.
- About twenty errors were planted across the kernels (wrong scale, shifted state, bit order, job index,
  stale cache) and each made the expected case fail.

Known gaps:

- The tests cover the fused stages, not the whole model. End to end, recall of a code planted at the start
  of an 8K prompt is correct at 256K context.
- Constants are for this card: thread counts, the slow grid sizes, the q8_1 cache slot size.
  Other GPUs are untested.

## A runtime fault found on the way: hipGraphExecUpdate

About one full test run in twenty failed the same case with the same wrong numbers, and never when that case ran
alone. Cause: the backend keys captured device graphs by the address of a graph's first node, and re-captures over
an existing executable with `cudaGraphExecUpdate`, rebuilding only if that reports failure. With ROCm 10.1,
`hipGraphExecUpdate` sometimes returns `hipSuccess` for a graph whose node kinds differ from the executable's at one
position (a memcpy node where the new graph has a kernel) and keeps the old node. Every later launch then runs the
stale copy. For the same pair of graphs the call usually returns the failure code, which is why it was rare.
`AMD_LOG_LEVEL=3 GGML_CUDA_GRAPH_DEBUG=1` shows both outcomes.

llama.cpp builds every graph of a context in one buffer, so the same key is re-captured at each prefill/decode
transition and each growth of the KV cache. This series always destroys and instantiates the executable on HIP
(`GGML_CUDA_HIP_GRAPH_EXEC_UPDATE=1` restores the update). A 1,500-token decode runs at the same speed either way.
The fault is in the path the unmodified backend uses too.

The test harness kept what found it: `FUSEDIAG=<case>` prints the allocator layout and every mismatching element,
`FUSEUNTIL=<case>` stops after a case, `FUSEFIXEDCTX=1` builds every graph at one address.

## Quality

Wikitext-2 test set, context 2048, against logits saved from the unmodified tree.

| path | perplexity (unmodified) | mean KL divergence | same top token |
|---|---|---|---|
| batches (512 / 128) | 6.199 (6.217) | 0.0185 | 94.2% |
| token by token, one column | 6.367 (6.386) | 0.0156 | 94.2% |
| token by token, three columns in every layer | 6.353 | 0.0159 | 94.7% |
| this build with rows scaled by 1 + 1e-7, against itself | | 0.0186 | 94.6% |

Two runs of the same build agree exactly. End to end, a code planted at the start of an 8K prompt is recalled at
256K context, and two codes (start and middle) of a 33.6K-token prompt are recalled with one column and with three.

A build is not measured for quality by the kernel tests, and a decode-speed run does not notice a wrong router:
both checks are needed. One build of this series had a router kernel left over from a timing experiment and
scored perplexity 88; the speed benchmarks looked normal.

## What is left

Weight reads are 5.4 ms of the 9.7 ms per token (the two control kernels read 7 MB of weights per layer pass and
count here) and run at about 80% of peak bandwidth while active. Small kernels are 2.2 ms and the gaps between
launches 2.1 ms: every dispatch of the replayed graph waits for the one before it. At long context, decode pays
about 1.4 ms per token per 100K tokens in the attention kernel. For batches, the tile attention kernel converts
the whole K and V of a layer to half floats on every micro-batch.

## Credits

llama.cpp by ggml-org and contributors. TurboQuant by TheTom. The optimization commits in this series were
written with Claude; each carries a co-author line.
