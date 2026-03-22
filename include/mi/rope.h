/* ═══════════════════════════════════════════════════════════════════
 * rope.h — positional-encoding methods
 *
 * Standard  — original RoPE (Su et al. 2021)
 * NTK-Aware — scale θ for context extension
 * YaRN      — NTK + attention-scale + frequency interpolation
 * Dynamic   — auto-scale θ when position exceeds training length
 * ALiBi     — linear attention bias (Press et al. 2022, no rotation)
 * None      — identity (for ablation / debugging)
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_ROPE_H
#define MI_ROPE_H

#include "base.h"

typedef struct MiRoPE MiRoPE;

typedef struct {
    const char *name;

    /* Apply positional rotation in-place.
     * vec: [n_heads * d_head], pos: absolute sequence position. */
    void (*apply)(MiRoPE *r, float *vec,
                  int pos, int n_heads, int d_head);

    /* (ALiBi only) Return score bias for head h, query at qp, key at kp. */
    float (*bias)(MiRoPE *r, int head, int q_pos, int k_pos);

    void (*destroy)(MiRoPE *r);
} MiRoPEVT;

struct MiRoPE {
    const MiRoPEVT *vt;
    void *ctx;
};

/* ── Dispatch ── */
static inline void mi_rope_apply(MiRoPE *r, float *vec,
                                  int pos, int n_heads, int d_head) {
    if (r->vt->apply) r->vt->apply(r, vec, pos, n_heads, d_head);
}
static inline float mi_rope_bias(MiRoPE *r, int h, int qp, int kp) {
    return r->vt->bias ? r->vt->bias(r, h, qp, kp) : 0.0f;
}
static inline void mi_rope_destroy(MiRoPE *r) {
    if (r->vt->destroy) r->vt->destroy(r);
}

/* ── Constructors ── */
MiRoPE mi_rope_standard(float theta);
MiRoPE mi_rope_ntk(float theta, float scale_factor);
MiRoPE mi_rope_yarn(float theta, float scale_factor, int original_max_pos);
MiRoPE mi_rope_dynamic(float theta, int original_max_pos);
MiRoPE mi_rope_alibi(int n_heads);
MiRoPE mi_rope_none(void);

#endif /* MI_ROPE_H */
