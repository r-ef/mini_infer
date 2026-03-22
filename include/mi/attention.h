
#ifndef MI_ATTENTION_H
#define MI_ATTENTION_H

#include "tensor.h"

typedef struct MiAttention MiAttention;

typedef struct {
    const char *name;


    void (*decode)(MiAttention *a,
                   const float *q, const float *K, const float *V,
                   float *out,
                   int n_heads, int n_kv_heads, int d_head,
                   int seq_len, int pos,
                   float *scratch);


    void (*prefill)(MiAttention *a,
                    const float *Q, const float *K, const float *V,
                    float *out,
                    int n_heads, int n_kv_heads, int d_head,
                    int n_pos, int seq_len,
                    float *scratch);

    void (*destroy)(MiAttention *a);
} MiAttentionVT;

struct MiAttention {
    const MiAttentionVT *vt;
    void *ctx;

    float *alibi_slopes;
};

static inline void mi_attention_decode(
        MiAttention *a, const float *q,
        const float *K, const float *V, float *out,
        int n_heads, int n_kv_heads, int d_head,
        int seq_len, int pos, float *scratch) {
    a->vt->decode(a, q, K, V, out,
                  n_heads, n_kv_heads, d_head,
                  seq_len, pos, scratch);
}

MiAttention mi_attention_standard(void);
MiAttention mi_attention_flash(void);
MiAttention mi_attention_linear(void);

int mi_attention_scratch_size(const MiAttention *a,
                              int n_heads, int seq_len);

#endif
