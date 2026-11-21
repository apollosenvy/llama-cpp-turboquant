# Inkling-2-Mini on one Radeon RX 7900 XTX

This branch makes the HIP (ROCm) backend run Inkling-2-Mini about three times faster on a single
RX 7900 XTX, with the same GGUF and the same sampling. It is built on llama.cpp at `29b52d398`
(`model : Inkling 2 Mini`) plus the TurboQuant `turbo3` KV cache type.

| | Before | After |
|---|---|---|
| Decode, empty context | 29 tok/s | 95 tok/s |
| Decode at about 100K context | 23.5 tok/s (90K) | 64 tok/s (104K) |
| Prefill, 8K prompt, 256K context | about 140 tok/s | 1,035 tok/s |
| Prefill, 104K prompt | about 130 tok/s | 727 tok/s |
| GPU kernel launches per token | 7,677 | 1,079 |

Model: `Inkling-2-Mini-IQ4_NL.gguf` (26B parameters, 24 layers of d_model 2304, 128-expert MoE with 8
active, layers 6 to 17 looped three times, so 48 layer passes per token). Card: RX 7900 XTX, 24 GB,
ROCm 10.1.0. "Before" is the same tree without this series, as it was deployed: 512K context, four expert
layers on the CPU, `-ub 16`. "After" runs 256K context, all experts on the GPU, `-ub 128`.

Output is not token-identical to the unmodified backend. The model is sensitive to float rounding:
turning the stock backend's own fusions off already shifts next-token probabilities by 0.03 within five
tokens. Correctness is checked at tensor level instead (see Tests).

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
- `GGML_CUDA_DUMP_NODES=1` prints every evaluated node and what was fused, once per captured graph.

## Tests

`tests/inkling-fusetest` builds each chain as a small graph exactly as the model does, runs it on the GPU
and on the CPU backend, and compares the outputs. 99 cases, single and multi token.

```bash
tests/inkling-fusetest/build.sh build && tests/inkling-fusetest/fusetest
```

- A case fails if no fusion fired, so a matcher that quietly stops matching is caught.
- Every graph is evaluated twice with different inputs, which catches state carried between evaluations.
- `FUSEREPEAT=200` re-evaluates each graph 200 times and requires bit-identical GPU output.
- `kbench.sh <case>` prints per-kernel times for one case from a `rocprofv3` trace. A case only covers a
  kernel if the kernel shows up there.
- About twenty errors were planted across the kernels (wrong scale, shifted state, bit order, job index,
  stale cache) and each made the expected case fail.

Known gaps:

- Two full runs out of about 55 had one failing case that did not reproduce, including under 14
  instrumented runs. The cause is unknown.
- The tests cover the fused stages, not the whole model. End to end, recall of a code planted at the start
  of an 8K prompt is correct at 256K context.
- Constants are for this card: thread counts, the slow grid sizes, the q8_1 cache slot size.
  Other GPUs are untested.

## What is left

Weight reads are 5.1 ms of the 10.5 ms per token and run at about 80% of peak bandwidth while active. The
other 5.4 ms is still small kernels and gaps. At long context, decode pays about 5 ms per token per 100K
tokens in `flash_attn_ext_vec`, which is bound by decoding the 3-bit cache. For batches, the tile attention
kernel converts the whole K and V of a layer to half floats on every micro-batch.

## Credits

llama.cpp by ggml-org and contributors. TurboQuant by TheTom. The optimization commits in this series were
written with Claude; each carries a co-author line.
