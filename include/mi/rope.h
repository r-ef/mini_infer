
#ifndef MI_ROPE_H
#define MI_ROPE_H

#include "base.h"

typedef struct MiRoPE MiRoPE;

typedef struct {
    const char *name;


    void (*apply)(MiRoPE *r, float *vec,
                  int pos, int n_heads, int d_head);


    float (*bias)(MiRoPE *r, int head, int q_pos, int k_pos);

    void (*destroy)(MiRoPE *r);
} MiRoPEVT;

struct MiRoPE {
    const MiRoPEVT *vt;
    void *ctx;
};

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

MiRoPE mi_rope_standard(float theta);
MiRoPE mi_rope_ntk(float theta, float scale_factor);
MiRoPE mi_rope_yarn(float theta, float scale_factor, int original_max_pos);
MiRoPE mi_rope_dynamic(float theta, int original_max_pos);
MiRoPE mi_rope_alibi(int n_heads);
MiRoPE mi_rope_none(void);

#endif
