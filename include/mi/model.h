/* ═══════════════════════════════════════════════════════════════════
 * model.h — multi-layer transformer with pluggable components
 *
 * Supports MHA / GQA / MQA, ReLU or SwiGLU FFN, any cache layout,
 * any attention back-end, any positional encoding.
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_MODEL_H
#define MI_MODEL_H

#include "tensor.h"
#include "cache.h"
#include "attention.h"
#include "rope.h"

typedef enum {
    MI_FFN_RELU   = 0,
    MI_FFN_SWIGLU = 1,
} MiFFNType;

/* ── Model hyper-parameters ── */
typedef struct {
    int       d_model;
    int       n_heads;
    int       n_kv_heads;     /* == n_heads → MHA, < n_heads → GQA, == 1 → MQA */
    int       d_head;         /* typically d_model / n_heads */
    int       d_ff;
    int       n_layers;
    int       vocab_size;
    int       max_seq_len;
    float     norm_eps;
    float     rope_theta;
    MiFFNType ffn_type;
} MiModelConfig;

/* ── Per-layer weights ── */
typedef struct {
    MiTensor Wq;           /* [n_heads   * d_head, d_model] */
    MiTensor Wk;           /* [n_kv_heads* d_head, d_model] */
    MiTensor Wv;           /* [n_kv_heads* d_head, d_model] */
    MiTensor Wo;           /* [d_model, n_heads * d_head]   */
    MiTensor W1;           /* FFN first  / gate */
    MiTensor W2;           /* FFN second / down */
    MiTensor W3;           /* SwiGLU up (unused for ReLU) */
    float   *rms_att;      /* [d_model] */
    float   *rms_ffn;      /* [d_model] */
} MiLayerWeights;

/* ── Full weight set ── */
typedef struct {
    MiTensor        tok_emb;       /* [vocab_size, d_model] */
    MiLayerWeights *layers;        /* [n_layers] */
    float          *rms_final;     /* [d_model] */
    MiTensor        out_proj;      /* [vocab_size, d_model] */
} MiModelWeights;

/* ── Complete model ── */
typedef struct MiModel {
    MiModelConfig   cfg;
    MiModelWeights  w;
    MiCache         cache;
    MiAttention     attn;
    MiRoPE          rope;
    int             pos;           /* next absolute position */
} MiModel;

/* ── Lifecycle ── */
MiModel  mi_model_create(MiModelConfig cfg);
void     mi_model_init_random(MiModel *m, MiRng *rng);

/* Load weights into an already-created model (config must match). */
MiStatus mi_model_load(MiModel *m, const char *path);
MiStatus mi_model_save(const MiModel *m, const char *path);

/* Create model from file: reads config + weights in one call.
 * Sets up default cache/attention/rope from config.
 * Caller should mi_model_set_*() to override components. */
MiModel  mi_model_load_file(const char *path);

void     mi_model_free(MiModel *m);

/* Plug components (call before first forward) */
void mi_model_set_attention(MiModel *m, MiAttention attn);
void mi_model_set_cache(MiModel *m, MiCache cache);
void mi_model_set_rope(MiModel *m, MiRoPE rope);

/* ── Forward pass ── */

/* Single token → logits[vocab_size].  scratch must be ≥ mi_model_scratch_size. */
void mi_model_forward(MiModel *m, int token, float *logits, float *scratch);

/* Process n tokens, return logits for EVERY position.
 * logits: [n * vocab_size].  Used for prefill and speculative verify. */
void mi_model_forward_batch(MiModel *m, const int *tokens, int n,
                            float *logits, float *scratch);

size_t mi_model_scratch_size(const MiModelConfig *cfg);

void mi_model_reset(MiModel *m);

#endif /* MI_MODEL_H */
