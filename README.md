<pre>

                                        ╔══╗
                                        ║  ║
                                   ╔════╬══╬════╗
                                   ║    ║  ║    ║
                                   ║  ╔═╩══╩═╗  ║
                                   ║  ║ ░░░░ ║  ║
                                   ║  ║ ░░░░ ║  ║
                                   ║  ║ ░░░░ ║  ║
                                   ║  ╚══════╝  ║
                                   ║     ()     ║
                                   ╚════════════╝
                                      ╔══════╗
                                      ║      ║
                                      ║      ║
                                      ╚══════╝




                              M I N I _ I N F E R

                      research inference engine for LLMs
                               written in C

                                  mar 2025




<========================================================================================>
abstract:

    mini_infer is a from-scratch LLM inference engine built for research.
    every component — KV cache, attention, sampling, positional encoding,
    quantization — is a pluggable vtable you can swap without touching
    anything else.

    this is not a production inference server. llama.cpp exists for that.
    this is a workbench for answering questions like:

        "how does int8 KV compression affect generation quality?"
        "what if attention used a different kernel?"
        "at what window size does sliding-window cache diverge from dense?"
        "how do different RoPE scaling methods compare on long context?"

    you write 4 functions matching a vtable, call mi_model_set_*(), and run
    your experiment against a real model generating real text.

    5,300 lines of C. zero dependencies beyond libc and libm.
    loads any HuggingFace LLaMA-family model via a Python converter.


<========================================================================================>
performance:

------------------------------------------------------------------------------------------
|                       SmolLM2-135M (30 layers, d=576, GQA)                             |
------------------------------------------------------------------------------------------

    +=========================================================================+
    | Metric                | Value                                           |
    +=========================================================================+
    | Decode throughput     | 53 tok/s  (single-thread, fp32)                 |
    | Prefill throughput    | 57 tok/s                                        |
    | Model load time       | < 1 second                                      |
    | Peak memory           | ~600 MB (fp32 weights + KV cache)               |
    | SIMD                  | NEON (ARM64) / AVX2+FMA (x86_64) / scalar       |
    +=========================================================================+

    matvec kernel: 21.6 GFLOPS (NEON, 4-row batched, 2x unrolled FMA)

    platform detection is automatic at compile time:
        ARM64 (linux, macos)    → NEON intrinsics
        x86_64 (linux, macos)   → AVX2 + FMA intrinsics
        anything else           → scalar fallback


<========================================================================================>
architecture:

------------------------------------------------------------------------------------------
|                                pluggable modules                                       |
------------------------------------------------------------------------------------------


    every research-relevant component is behind a vtable:

    ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
    │  KV Cache     │     │  Attention     │     │  Sampling     │
    │               │     │               │     │               │
    │  • dense      │     │  • standard   │     │  • greedy     │
    │  • paged      │     │  • flash      │     │  • top-k      │
    │  • sliding    │     │  • linear     │     │  • top-p      │
    │  • compressed │     │               │     │  • min-p      │
    └──────┬───────┘     └──────┬────────┘     │  • typical    │
           │                     │              │  • mirostat   │
           │                     │              │  • repetition │
           │                     │              │  • chain      │
           │                     │              └──────┬────────┘
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │                        MiModel                          │
    │                                                         │
    │   mi_model_set_cache(&model, mi_cache_paged(...));      │
    │   mi_model_set_attention(&model, mi_attention_flash()); │
    │   mi_model_set_rope(&model, mi_rope_yarn(...));         │
    │                                                         │
    │   mi_model_forward(&model, token, logits, scratch);     │
    │                                                         │
    └──────────────────────────┬──────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  RoPE         │    │  Quantization │    │  Memory       │
    │               │    │               │    │               │
    │  • standard   │    │  • INT8 absmax│    │  • attn sink  │
    │  • NTK-aware  │    │  • INT8 zp    │    │  • H2O evict  │
    │  • YaRN       │    │  • INT4 group │    │  • token merge│
    │  • dynamic    │    │  • Q4_0 block │    │  • vector RAG │
    │  • ALiBi      │    │  • Q8_0 block │    │               │
    │  • none       │    │  • FP16/BF16  │    │               │
    └──────────────┘    └──────────────┘    └──────────────┘


    to add your own implementation of anything:

        // 1. write your functions
        static void my_append(MiCache *c, int layer,
                              const float *k, const float *v) {
            /* your implementation */
        }
        // ... other vtable methods ...

        // 2. wire up the vtable
        static const MiCacheVT my_vt = {
            .name = "my_cache", .append = my_append, /* ... */
        };

        // 3. constructor
        MiCache my_cache_create(/* params */) {
            MyCtx *ctx = calloc(1, sizeof(MyCtx));
            return (MiCache){ .vt = &my_vt, .ctx = ctx, /* ... */ };
        }

        // 4. plug it in
        mi_model_set_cache(&model, my_cache_create(/* ... */));
        // done. run your experiment.


------------------------------------------------------------------------------------------
|                              forward pass (single token)                               |
------------------------------------------------------------------------------------------


    mi_model_forward(model, token, logits, scratch)
         │
         ▼
    ┌─ embedding lookup ──────────────────────────────────┐
    │  x = tok_emb[token]                                 │
    └──────────────────────────────────────────────────────┘
         │
         ▼
    ┌─ for each layer (0 .. n_layers-1) ──────────────────┐
    │                                                      │
    │   x_norm = RMSNorm(x, rms_att)                      │
    │                                                      │
    │   q = Wq @ x_norm          [n_heads * d_head]        │
    │   k = Wk @ x_norm          [n_kv_heads * d_head]     │
    │   v = Wv @ x_norm          [n_kv_heads * d_head]     │
    │                                                      │
    │   RoPE(q, k, pos)          ◄── pluggable             │
    │                                                      │
    │   cache.append(layer, k, v) ◄── pluggable            │
    │   K, V = cache.get(layer)                            │
    │                                                      │
    │   attn = Attention(q, K, V)  ◄── pluggable           │
    │   x = x + Wo @ attn                                 │
    │                                                      │
    │   x_norm = RMSNorm(x, rms_ffn)                      │
    │   x = x + SwiGLU(x_norm)    or  ReLU FFN            │
    │                                                      │
    └──────────────────────────────────────────────────────┘
         │
         ▼
    ┌─ output ────────────────────────────────────────────┐
    │   logits = out_proj @ RMSNorm(x, rms_final)         │
    └──────────────────────────────────────────────────────┘


<========================================================================================>
modules:

------------------------------------------------------------------------------------------
|                                  KV cache (cache.h)                                    |
------------------------------------------------------------------------------------------

    Dense         contiguous pre-allocated arrays. baseline reference.
    Paged         block-allocated pages with free list. vLLM-style.
    Sliding       ring buffer, keeps last N tokens. Mistral-style.
    Compressed    recent entries fp32, older entries auto-quantised to int8.


------------------------------------------------------------------------------------------
|                                  attention (attention.h)                                |
------------------------------------------------------------------------------------------

    Standard      O(n²) with explicit scores buffer. handles GQA natively.
    Flash         online softmax, O(d_head) peak memory per head. no scores buffer.
    Linear        ELU+1 kernel approximation. O(n·d²) total. experimental.


------------------------------------------------------------------------------------------
|                                  sampling (sampling.h)                                 |
------------------------------------------------------------------------------------------

    Greedy        argmax.
    Top-K         temperature + keep K highest logits.
    Top-P         nucleus sampling. Holtzman et al. 2019.
    Min-P         keep tokens with p ≥ min_p · max(p).
    Typical       keep tokens near expected information content. Meister 2023.
    Mirostat v2   adaptive sampling targeting surprise value τ. Basu 2021.
    Repetition    penalise recently-generated tokens (logit transform).
    Chain         compose multiple transforms: repetition → top-p.


------------------------------------------------------------------------------------------
|                                  positional encoding (rope.h)                          |
------------------------------------------------------------------------------------------

    Standard      original RoPE. Su et al. 2021.
    NTK-Aware     scale θ for context extension beyond training length.
    YaRN          frequency interpolation + attention scaling. context extension.
    Dynamic       auto-scale θ when position exceeds training length.
    ALiBi         linear attention bias. Press et al. 2022. no rotation.
    None          identity. for ablation.


------------------------------------------------------------------------------------------
|                                  quantization (quant.h)                                |
------------------------------------------------------------------------------------------

    INT8 absmax   symmetric per-tensor quantization. 127 levels.
    INT8 zp       asymmetric with zero-point.
    INT4 group    GPTQ-style group quantization (32/64/128 group sizes).
    Q4_0          GGML-style block quantization. 32 values per block.
    Q8_0          GGML-style block quantization. 32 values per block.
    FP16/BF16     conversion utilities for storage experiments.

    includes: quantized matvec, error analysis (MSE, SNR dB, cosine sim).


------------------------------------------------------------------------------------------
|                                  memory management (memory.h)                          |
------------------------------------------------------------------------------------------

    Attention Sink     StreamingLLM: keep first K + last N tokens. Xiao 2023.
    H2O                Heavy-Hitter Oracle: evict least-attended positions. Zhang 2023.
    Token Merge        merge similar adjacent KV entries by cosine similarity.
    Vector Store       brute-force kNN for retrieval-augmented memory.
    RAG Augment        inject retrieved embeddings into KV cache.


------------------------------------------------------------------------------------------
|                                  speculative decoding (speculative.h)                  |
------------------------------------------------------------------------------------------

    Standard acceptance-rejection algorithm. Leviathan et al. 2023.
    Draft model proposes K tokens, target model verifies in one batch pass.
    Guarantees target-model distribution. Tracks acceptance rate statistics.


<========================================================================================>
experiment example:

------------------------------------------------------------------------------------------
|           KV cache compression: int8 vs dense vs sliding window                        |
------------------------------------------------------------------------------------------

    built-in experiment: examples/experiment_cache.c

    runs the same prompt through dense (ground truth), compressed (int8 old
    entries), and sliding window caches. greedy decoding so divergence is
    purely from the cache, not sampling randomness.

    results on SmolLM2-135M, 35-token prompt, 128 generated tokens:


    compressed cache (int8 quantization of old entries):
    ┌───────────────────────────────────────────────────────────────────────┐
    │  fresh_count │ 1st diverge │ match%  │ mean cos_sim │ mem savings   │
    ├──────────────┼─────────────┼─────────┼──────────────┼───────────────┤
    │           4  │   tok   0   │   3.1%  │   0.9186     │  0.27x        │
    │           8  │   tok   0   │   1.6%  │   0.9416     │  0.29x        │
    │          16  │   tok   0   │   0.8%  │   0.9678     │  0.32x        │
    │          32  │   tok   0   │   7.0%  │   0.8290     │  0.40x        │
    │          64  │   tok   0   │   5.5%  │   0.8683     │  0.54x        │
    │         128  │   tok   0   │   5.5%  │   0.8692     │  0.84x        │
    └───────────────────────────────────────────────────────────────────────┘

    sliding window cache:
    ┌───────────────────────────────────────────────────────────────────────┐
    │  window_size │ 1st diverge │ match%  │ mean cos_sim                 │
    ├──────────────┼─────────────┼─────────┼──────────────────────────────┤
    │          32  │   tok   0   │   0.8%  │   0.8568                     │
    │          64  │   tok  30   │  28.9%  │   0.8517                     │
    │         128  │   tok  95   │  79.7%  │   0.8753                     │
    │         256  │   never     │ 100.0%  │   1.0000                     │
    └───────────────────────────────────────────────────────────────────────┘

    finding: for SmolLM2-135M, sliding window degrades gracefully while
    int8 KV quantization diverges immediately at all fresh_count values.
    the int8 rounding error on 192-dimensional KV vectors (3 heads × 64 dim)
    is large enough relative to attention score margins to flip token choices.
    sliding window cleanly drops old entries instead of corrupting them.

    reproduce:
        python tools/convert_hf.py HuggingFaceTB/SmolLM2-135M ./models/smollm --download
        make
        ./examples/experiment_cache ./models/smollm 128


<========================================================================================>
usage:

------------------------------------------------------------------------------------------
|                                   build                                                |
------------------------------------------------------------------------------------------

    $ make                  # library + all examples + tests
    $ make test             # run 23 unit tests
    $ make debug            # build with ASan + UBSan
    $ make CFLAGS+="-DMI_NO_SIMD"   # force scalar (disable NEON/AVX2)


------------------------------------------------------------------------------------------
|                              load a real model                                         |
------------------------------------------------------------------------------------------

    # install dependencies
    $ pip install safetensors huggingface_hub torch

    # download + convert any LLaMA-family model from HuggingFace
    $ python tools/convert_hf.py HuggingFaceTB/SmolLM2-135M ./models/smollm --download

    # generate text
    $ ./examples/generate_real ./models/smollm "The meaning of life is" 64

    works with: SmolLM2, TinyLlama, Llama-2, Llama-3, Mistral, Qwen2, etc.
    anything using LlamaForCausalLM architecture.


------------------------------------------------------------------------------------------
|                              run examples                                              |
------------------------------------------------------------------------------------------

    $ ./examples/basic_generate          # compare 6 samplers on a tiny random model
    $ ./examples/cache_bench             # benchmark 4 cache layouts
    $ ./examples/quant_compare           # compare 8 quantization methods (error analysis)
    $ ./examples/sampling_explore        # token frequency analysis across samplers
    $ ./examples/rag_memory              # RAG retrieval + sink + H2O + merge demos
    $ ./examples/generate_real <dir>     # generate text with a real model
    $ ./examples/experiment_cache <dir>  # compressed vs dense vs sliding cache experiment


<========================================================================================>
source files:

------------------------------------------------------------------------------------------
|                                   file layout                                          |
------------------------------------------------------------------------------------------

    include/mi/
        base.h              macros, logging, timer, xoshiro256** RNG
        tensor.h            2D float matrix with views
        arena.h             bump allocator for scratch buffers
        ops.h               matvec, softmax, RMSNorm, SiLU, SwiGLU FFN
        cache.h             KV cache interface + 4 implementations
        attention.h         attention interface + 3 implementations
        sampling.h          sampler interface + 8 implementations
        rope.h              positional encoding interface + 6 implementations
        quant.h             quantization schemes + error analysis
        speculative.h       speculative decoding
        memory.h            compression, eviction, RAG memory
        model.h             multi-layer transformer (GQA, SwiGLU, save/load)
        generate.h          generation orchestrator with benchmarking
        tokenizer.h         byte-level BPE tokenizer

    src/                    implementations (14 files, ~5300 lines)
    examples/               runnable demos + experiments (7 files)
    tests/                  unit test suite (23 tests)
    tools/convert_hf.py     HuggingFace model converter


<========================================================================================>
how it differs from llama.cpp:

------------------------------------------------------------------------------------------
|                                   positioning                                          |
------------------------------------------------------------------------------------------

    llama.cpp is a production inference engine.
    mini_infer is a research workbench.

    +---------------------------+--------------------+--------------------+
    | Dimension                 | mini_infer         | llama.cpp          |
    +---------------------------+--------------------+--------------------+
    | Lines of code             | 5,300              | 500,000+           |
    | Can you read it all?      | yes, in a day      | no                 |
    | GPU support               | no                 | Metal/CUDA/Vulkan  |
    | Model architectures       | LLaMA-family       | hundreds           |
    | Quantization formats      | 6 (research)       | 30+ (production)   |
    | Add a new cache layout    | 4 functions        | understand GGML    |
    | Add a new attention type  | 4 functions        | fork the project   |
    | Run a comparison expt     | built-in           | write from scratch |
    | Serve 1000 users          | no                 | yes                |
    +---------------------------+--------------------+--------------------+

    if your goal is to serve models fast, use llama.cpp.
    if your goal is to test ideas fast, use this.


<========================================================================================>

    "premature optimization is the root of all evil"
        — donald knuth

    "but you need to be fast enough to actually iterate"
        — the reason this has SIMD


                                         *

</pre>
