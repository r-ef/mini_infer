
#ifndef MI_SAMPLING_H
#define MI_SAMPLING_H

#include "base.h"

typedef struct MiSampler MiSampler;

typedef struct {
    const char *name;


    int  (*sample)(MiSampler *s, float *logits, int vocab_size, MiRng *rng);


    void (*accept)(MiSampler *s, int token_id);


    void (*reset)(MiSampler *s);

    void (*destroy)(MiSampler *s);
} MiSamplerVT;

struct MiSampler {
    const MiSamplerVT *vt;
    void *ctx;
};

static inline int mi_sampler_sample(MiSampler *s, float *logits,
                                     int vocab_size, MiRng *rng) {
    return s->vt->sample(s, logits, vocab_size, rng);
}
static inline void mi_sampler_accept(MiSampler *s, int tok) {
    if (s->vt->accept) s->vt->accept(s, tok);
}
static inline void mi_sampler_reset(MiSampler *s) {
    if (s->vt->reset) s->vt->reset(s);
}
static inline void mi_sampler_destroy(MiSampler *s) {
    if (s->vt->destroy) s->vt->destroy(s);
}

MiSampler mi_sampler_greedy(void);
MiSampler mi_sampler_top_k(int k, float temperature);
MiSampler mi_sampler_top_p(float p, float temperature);
MiSampler mi_sampler_min_p(float min_p, float temperature);
MiSampler mi_sampler_typical(float tau, float temperature);
MiSampler mi_sampler_mirostat_v2(float tau, float eta);
MiSampler mi_sampler_repetition(float penalty, int window_size);

MiSampler mi_sampler_chain(MiSampler *samplers, int n);

#endif
