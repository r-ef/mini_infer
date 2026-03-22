
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

typedef struct {
    int       d_model;
    int       n_heads;
    int       n_kv_heads;
    int       d_head;
    int       d_ff;
    int       n_layers;
    int       vocab_size;
    int       max_seq_len;
    float     norm_eps;
    float     rope_theta;
    MiFFNType ffn_type;
} MiModelConfig;

typedef struct {
    MiTensor Wq;
    MiTensor Wk;
    MiTensor Wv;
    MiTensor Wo;
    MiTensor W1;
    MiTensor W2;
    MiTensor W3;
    float   *rms_att;
    float   *rms_ffn;
} MiLayerWeights;

typedef struct {
    MiTensor        tok_emb;
    MiLayerWeights *layers;
    float          *rms_final;
    MiTensor        out_proj;
} MiModelWeights;

typedef struct MiModel {
    MiModelConfig   cfg;
    MiModelWeights  w;
    MiCache         cache;
    MiAttention     attn;
    MiRoPE          rope;
    int             pos;
} MiModel;

MiModel  mi_model_create(MiModelConfig cfg);
void     mi_model_init_random(MiModel *m, MiRng *rng);

MiStatus mi_model_load(MiModel *m, const char *path);
MiStatus mi_model_save(const MiModel *m, const char *path);

MiModel  mi_model_load_file(const char *path);

void     mi_model_free(MiModel *m);

void mi_model_set_attention(MiModel *m, MiAttention attn);
void mi_model_set_cache(MiModel *m, MiCache cache);
void mi_model_set_rope(MiModel *m, MiRoPE rope);

void mi_model_forward(MiModel *m, int token, float *logits, float *scratch);

void mi_model_forward_batch(MiModel *m, const int *tokens, int n,
                            float *logits, float *scratch);

size_t mi_model_scratch_size(const MiModelConfig *cfg);

void mi_model_reset(MiModel *m);

#endif
