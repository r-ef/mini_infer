/* ═══════════════════════════════════════════════════════════════════
 * sampling.h — pluggable sampling / logit-processing strategies
 *
 * Greedy       — argmax
 * Top-K        — keep K highest, temperature, categorical sample
 * Top-P        — nucleus sampling (Holtzman et al.)
 * Min-P        — keep tokens with p ≥ min_p · p_max
 * Typical      — keep tokens near expected information content
 * Mirostat v2  — adaptive targeting a surprise value τ
 * Repetition   — penalise recently-generated tokens (logit xform)
 *
 * Chain        — compose transforms: e.g. repetition → top-p
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_SAMPLING_H
#define MI_SAMPLING_H

#include "base.h"

typedef struct MiSampler MiSampler;

typedef struct {
    const char *name;

    /* Transform logits and/or sample a token.
     * Returns the chosen token id (≥ 0) or -1 if this sampler only
     * transforms logits (useful in a chain). */
    int  (*sample)(MiSampler *s, float *logits, int vocab_size, MiRng *rng);

    /* Notify the sampler that token_id was accepted (for stateful
     * samplers like repetition penalty, mirostat). */
    void (*accept)(MiSampler *s, int token_id);

    /* Reset internal state. */
    void (*reset)(MiSampler *s);

    void (*destroy)(MiSampler *s);
} MiSamplerVT;

struct MiSampler {
    const MiSamplerVT *vt;
    void *ctx;
};

/* ── Dispatch ── */
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

/* ── Constructors ── */
MiSampler mi_sampler_greedy(void);
MiSampler mi_sampler_top_k(int k, float temperature);
MiSampler mi_sampler_top_p(float p, float temperature);
MiSampler mi_sampler_min_p(float min_p, float temperature);
MiSampler mi_sampler_typical(float tau, float temperature);
MiSampler mi_sampler_mirostat_v2(float tau, float eta);
MiSampler mi_sampler_repetition(float penalty, int window_size);

/* Chain: the last sampler in the array must return a token (≥0);
 * earlier ones may be logit-only transforms (return -1).
 * The chain takes ownership of the array (caller must not free items). */
MiSampler mi_sampler_chain(MiSampler *samplers, int n);

#endif /* MI_SAMPLING_H */
