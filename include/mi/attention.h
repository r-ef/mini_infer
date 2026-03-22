/* ═══════════════════════════════════════════════════════════════════
 * attention.h — pluggable attention back-ends
 *
 * All variants handle GQA (n_kv_heads ≤ n_heads) transparently.
 *
 * Standard — O(n²) full attention, explicit scores vector
 * Flash    — online softmax, O(d) peak memory per head (decode)
 * Linear   — ELU+1 kernel approximation, O(n·d²) total
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_ATTENTION_H
#define MI_ATTENTION_H

#include "tensor.h"

typedef struct MiAttention MiAttention;

typedef struct {
    const char *name;

    /* Decode: single query position.
     * q    [n_heads * d_head]
     * K    [seq_len * kv_dim]     (kv_dim = n_kv_heads * d_head)
     * V    [seq_len * kv_dim]
     * out  [n_heads * d_head]
     * scratch: implementation-specific workspace */
    void (*decode)(MiAttention *a,
                   const float *q, const float *K, const float *V,
                   float *out,
                   int n_heads, int n_kv_heads, int d_head,
                   int seq_len, int pos,
                   float *scratch);

    /* Prefill: multiple query positions (batch).
     * Q    [n_pos * n_heads * d_head]
     * K,V  [seq_len * kv_dim]
     * out  [n_pos * n_heads * d_head]  */
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
    /* Optional ALiBi slopes — if non-NULL, attention adds
     * slope[h] * (q_pos − k_pos) to every score. [n_heads] */
    float *alibi_slopes;
};

/* ── Dispatch ── */
static inline void mi_attention_decode(
        MiAttention *a, const float *q,
        const float *K, const float *V, float *out,
        int n_heads, int n_kv_heads, int d_head,
        int seq_len, int pos, float *scratch) {
    a->vt->decode(a, q, K, V, out,
                  n_heads, n_kv_heads, d_head,
                  seq_len, pos, scratch);
}

/* ── Constructors ── */
MiAttention mi_attention_standard(void);
MiAttention mi_attention_flash(void);
MiAttention mi_attention_linear(void);

/* Scratch-space query (caller must allocate at least this many floats) */
int mi_attention_scratch_size(const MiAttention *a,
                              int n_heads, int seq_len);

#endif /* MI_ATTENTION_H */
