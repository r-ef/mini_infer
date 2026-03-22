/* sampling.c — greedy, top-k, top-p, min-p, typical, mirostat v2,
 *               repetition penalty, and chain composition */
#include "mi/sampling.h"
#include "mi/ops.h"

/* ── Helper: categorical sample from a probability distribution ── */
static int categorical(const float *probs, int n, MiRng *rng) {
    float r = mi_rng_float(rng);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (r <= cdf) return i;
    }
    return n - 1;
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  1. Greedy                                                        ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

static int greedy_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    MI_UNUSED(s); MI_UNUSED(rng);
    return mi_argmax(logits, n);
}
static void noop_destroy(MiSampler *s) { MI_UNUSED(s); }

static const MiSamplerVT greedy_vt = {
    .name = "greedy", .sample = greedy_sample,
    .accept = NULL, .reset = NULL, .destroy = noop_destroy,
};

MiSampler mi_sampler_greedy(void) {
    return (MiSampler){ .vt = &greedy_vt, .ctx = NULL };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  2. Top-K                                                         ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct { int k; float temperature; } TopKCtx;

static int topk_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    TopKCtx *c = (TopKCtx *)s->ctx;
    int k = MI_MIN(c->k, n);

    if (c->temperature <= 0.0f) return mi_argmax(logits, n);

    /* Partial sort: find the k-th largest value */
    float *tmp = (float *)malloc(n * sizeof(float));
    MI_CHECK_OOM(tmp);
    memcpy(tmp, logits, n * sizeof(float));

    /* Simple partial selection sort for top-k threshold */
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (tmp[j] > tmp[best]) best = j;
        float t = tmp[i]; tmp[i] = tmp[best]; tmp[best] = t;
    }
    float threshold = tmp[k - 1];
    free(tmp);

    /* Apply temperature and mask below threshold */
    float inv_t = 1.0f / c->temperature;
    for (int i = 0; i < n; i++)
        logits[i] = (logits[i] >= threshold) ? logits[i] * inv_t : -1e9f;

    mi_softmax(logits, n);
    return categorical(logits, n, rng);
}

static void topk_destroy(MiSampler *s) { free(s->ctx); }

static const MiSamplerVT topk_vt = {
    .name = "top_k", .sample = topk_sample,
    .accept = NULL, .reset = NULL, .destroy = topk_destroy,
};

MiSampler mi_sampler_top_k(int k, float temperature) {
    TopKCtx *c = (TopKCtx *)malloc(sizeof(TopKCtx));
    MI_CHECK_OOM(c);
    c->k = k; c->temperature = temperature;
    return (MiSampler){ .vt = &topk_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  3. Top-P (nucleus)                                               ║
 * ║                                                                   ║
 * ║  Sort by probability, keep smallest set with cumulative ≥ p.    ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct { float p; float temperature; } TopPCtx;

/* ── qsort comparator for (prob, index) pairs ── */
typedef struct { float val; int idx; } ProbIdx;

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->val;
    float pb = ((const ProbIdx *)b)->val;
    return (pa < pb) - (pa > pb);  /* descending */
}

static int topp_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    TopPCtx *c = (TopPCtx *)s->ctx;

    if (c->temperature <= 0.0f) return mi_argmax(logits, n);

    /* Apply temperature + softmax */
    float inv_t = 1.0f / c->temperature;
    for (int i = 0; i < n; i++) logits[i] *= inv_t;
    mi_softmax(logits, n);

    /* Sort by probability descending — O(n log n) */
    ProbIdx *pairs = (ProbIdx *)malloc(n * sizeof(ProbIdx));
    MI_CHECK_OOM(pairs);
    for (int i = 0; i < n; i++) { pairs[i].val = logits[i]; pairs[i].idx = i; }
    qsort(pairs, n, sizeof(ProbIdx), cmp_prob_desc);

    /* Find nucleus cutoff */
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += pairs[i].val;
        if (cumsum >= c->p) { cutoff = i + 1; break; }
    }

    /* Zero out tokens outside nucleus and renormalise */
    for (int i = cutoff; i < n; i++) logits[pairs[i].idx] = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += logits[i];
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < n; i++) logits[i] *= inv; }

    int tok = categorical(logits, n, rng);
    free(pairs);
    return tok;
}

static void topp_destroy(MiSampler *s) { free(s->ctx); }

static const MiSamplerVT topp_vt = {
    .name = "top_p", .sample = topp_sample,
    .accept = NULL, .reset = NULL, .destroy = topp_destroy,
};

MiSampler mi_sampler_top_p(float p, float temperature) {
    TopPCtx *c = (TopPCtx *)malloc(sizeof(TopPCtx));
    MI_CHECK_OOM(c);
    c->p = p; c->temperature = temperature;
    return (MiSampler){ .vt = &topp_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  4. Min-P                                                         ║
 * ║                                                                   ║
 * ║  Keep tokens where  prob ≥ min_p · max(prob).                    ║
 * ║  Simpler and often better than top-p for creative generation.    ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct { float min_p; float temperature; } MinPCtx;

static int minp_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    MinPCtx *c = (MinPCtx *)s->ctx;

    if (c->temperature <= 0.0f) return mi_argmax(logits, n);

    float inv_t = 1.0f / c->temperature;
    for (int i = 0; i < n; i++) logits[i] *= inv_t;
    mi_softmax(logits, n);

    float max_p = mi_vec_max(logits, n);
    float threshold = c->min_p * max_p;

    for (int i = 0; i < n; i++)
        if (logits[i] < threshold) logits[i] = 0.0f;

    float sum = mi_vec_sum(logits, n);
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < n; i++) logits[i] *= inv; }

    return categorical(logits, n, rng);
}

static void minp_destroy(MiSampler *s) { free(s->ctx); }

static const MiSamplerVT minp_vt = {
    .name = "min_p", .sample = minp_sample,
    .accept = NULL, .reset = NULL, .destroy = minp_destroy,
};

MiSampler mi_sampler_min_p(float min_p, float temperature) {
    MinPCtx *c = (MinPCtx *)malloc(sizeof(MinPCtx));
    MI_CHECK_OOM(c);
    c->min_p = min_p; c->temperature = temperature;
    return (MiSampler){ .vt = &minp_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  5. Typical sampling (Meister et al. 2023)                       ║
 * ║                                                                   ║
 * ║  Keep tokens whose information content (-log p) is near the     ║
 * ║  expected information content (entropy H).  Sort by |−log p − H|║
 * ║  and keep cumulative probability ≥ τ.                            ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct { float tau; float temperature; } TypicalCtx;

static int typical_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    TypicalCtx *c = (TypicalCtx *)s->ctx;

    if (c->temperature <= 0.0f) return mi_argmax(logits, n);

    float inv_t = 1.0f / c->temperature;
    for (int i = 0; i < n; i++) logits[i] *= inv_t;
    mi_softmax(logits, n);

    /* Compute entropy H */
    float H = 0.0f;
    for (int i = 0; i < n; i++)
        if (logits[i] > 1e-12f)
            H -= logits[i] * logf(logits[i]);

    /* Compute deviation |−log p − H| for each token, sort ascending */
    ProbIdx *pairs = (ProbIdx *)malloc(n * sizeof(ProbIdx));
    MI_CHECK_OOM(pairs);
    for (int i = 0; i < n; i++) {
        float info = (logits[i] > 1e-12f) ? -logf(logits[i]) : 30.0f;
        pairs[i].val = fabsf(info - H);
        pairs[i].idx = i;
    }

    /* Sort by deviation ascending — O(n log n) */
    qsort(pairs, n, sizeof(ProbIdx), cmp_prob_desc); /* reuse desc... */
    /* Actually we want ascending — reverse the comparator: */
    /* Reverse in-place */
    for (int i = 0, j = n-1; i < j; i++, j--) {
        ProbIdx tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
    }

    /* Keep cumulative ≥ τ */
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += logits[pairs[i].idx];
        if (cumsum >= c->tau) { cutoff = i + 1; break; }
    }
    for (int i = cutoff; i < n; i++) logits[pairs[i].idx] = 0.0f;

    float sum = mi_vec_sum(logits, n);
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < n; i++) logits[i] *= inv; }

    int tok = categorical(logits, n, rng);
    free(pairs);
    return tok;
}

static void typical_destroy(MiSampler *s) { free(s->ctx); }

static const MiSamplerVT typical_vt = {
    .name = "typical", .sample = typical_sample,
    .accept = NULL, .reset = NULL, .destroy = typical_destroy,
};

MiSampler mi_sampler_typical(float tau, float temperature) {
    TypicalCtx *c = (TypicalCtx *)malloc(sizeof(TypicalCtx));
    MI_CHECK_OOM(c);
    c->tau = tau; c->temperature = temperature;
    return (MiSampler){ .vt = &typical_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  6. Mirostat v2 (Basu et al. 2021)                               ║
 * ║                                                                   ║
 * ║  Adaptively targets a surprise value τ by maintaining a maximum  ║
 * ║  surprise threshold μ.  After each token, μ is adjusted:         ║
 * ║     μ ← μ − η · (surprise − τ)                                  ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float tau;    /* target surprise */
    float eta;    /* learning rate */
    float mu;     /* current max surprise threshold */
    /* We also cache the logprobs from the last sample */
    float *last_logprobs;
    int    last_n;
} MirostatCtx;

static int mirostat_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    MirostatCtx *c = (MirostatCtx *)s->ctx;

    /* Compute log-probs */
    float *lp = (float *)realloc(c->last_logprobs, n * sizeof(float));
    MI_CHECK_OOM(lp);
    c->last_logprobs = lp;
    c->last_n = n;

    mi_log_softmax(logits, lp, n);

    /* Keep tokens whose surprise (−log p) is ≤ μ */
    for (int i = 0; i < n; i++)
        logits[i] = (-lp[i] <= c->mu) ? expf(lp[i]) : 0.0f;

    float sum = mi_vec_sum(logits, n);
    if (sum <= 0.0f) {
        /* Fallback: just pick the max-prob token */
        mi_softmax(lp, n);  /* oops, lp is logprobs, use for softmax */
        return mi_argmax(lp, n);
    }

    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) logits[i] *= inv;

    return categorical(logits, n, rng);
}

static void mirostat_accept(MiSampler *s, int token_id) {
    MirostatCtx *c = (MirostatCtx *)s->ctx;
    if (!c->last_logprobs || token_id < 0 || token_id >= c->last_n) return;

    float surprise = -c->last_logprobs[token_id];
    c->mu -= c->eta * (surprise - c->tau);
    /* Clamp μ to be positive */
    if (c->mu < 0.01f) c->mu = 0.01f;
}

static void mirostat_reset(MiSampler *s) {
    MirostatCtx *c = (MirostatCtx *)s->ctx;
    c->mu = 2.0f * c->tau;  /* re-init */
}

static void mirostat_destroy(MiSampler *s) {
    MirostatCtx *c = (MirostatCtx *)s->ctx;
    free(c->last_logprobs); free(c);
}

static const MiSamplerVT mirostat_vt = {
    .name = "mirostat_v2", .sample = mirostat_sample,
    .accept = mirostat_accept, .reset = mirostat_reset,
    .destroy = mirostat_destroy,
};

MiSampler mi_sampler_mirostat_v2(float tau, float eta) {
    MirostatCtx *c = (MirostatCtx *)calloc(1, sizeof(MirostatCtx));
    MI_CHECK_OOM(c);
    c->tau = tau; c->eta = eta;
    c->mu  = 2.0f * tau;
    return (MiSampler){ .vt = &mirostat_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  7. Repetition penalty — logit transform (returns -1)            ║
 * ║                                                                   ║
 * ║  Divides logits for recently-seen tokens by penalty (>1).        ║
 * ║  Meant to be used in a chain before a real sampler.              ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float penalty;
    int   window;
    int  *history;      /* ring buffer */
    int   hist_len;
    int   hist_pos;
} RepCtx;

static int rep_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    MI_UNUSED(rng);
    RepCtx *c = (RepCtx *)s->ctx;
    int count = MI_MIN(c->hist_len, c->window);

    for (int i = 0; i < count; i++) {
        int idx = c->history[(c->hist_pos - 1 - i + c->window) % c->window];
        if (idx >= 0 && idx < n) {
            if (logits[idx] > 0.0f)
                logits[idx] /= c->penalty;
            else
                logits[idx] *= c->penalty;
        }
    }
    return -1;  /* logit-only transform */
}

static void rep_accept(MiSampler *s, int token_id) {
    RepCtx *c = (RepCtx *)s->ctx;
    c->history[c->hist_pos % c->window] = token_id;
    c->hist_pos++;
    if (c->hist_len < c->window) c->hist_len++;
}

static void rep_reset(MiSampler *s) {
    RepCtx *c = (RepCtx *)s->ctx;
    c->hist_len = 0; c->hist_pos = 0;
}

static void rep_destroy(MiSampler *s) {
    RepCtx *c = (RepCtx *)s->ctx;
    free(c->history); free(c);
}

static const MiSamplerVT rep_vt = {
    .name = "repetition", .sample = rep_sample,
    .accept = rep_accept, .reset = rep_reset, .destroy = rep_destroy,
};

MiSampler mi_sampler_repetition(float penalty, int window) {
    RepCtx *c = (RepCtx *)calloc(1, sizeof(RepCtx));
    MI_CHECK_OOM(c);
    c->penalty = penalty;
    c->window  = window;
    c->history = (int *)calloc(window, sizeof(int));
    MI_CHECK_OOM(c->history);
    return (MiSampler){ .vt = &rep_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  8. Chain — compose multiple samplers                             ║
 * ║                                                                   ║
 * ║  Runs each sampler in sequence.  Earlier ones transform logits   ║
 * ║  (return -1).  The last one produces the actual token.           ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    MiSampler *items;
    int        n;
} ChainCtx;

static int chain_sample(MiSampler *s, float *logits, int n, MiRng *rng) {
    ChainCtx *c = (ChainCtx *)s->ctx;
    int tok = -1;
    for (int i = 0; i < c->n; i++) {
        int result = mi_sampler_sample(&c->items[i], logits, n, rng);
        if (result >= 0) tok = result;
    }
    return tok;
}

static void chain_accept(MiSampler *s, int token_id) {
    ChainCtx *c = (ChainCtx *)s->ctx;
    for (int i = 0; i < c->n; i++)
        mi_sampler_accept(&c->items[i], token_id);
}

static void chain_reset(MiSampler *s) {
    ChainCtx *c = (ChainCtx *)s->ctx;
    for (int i = 0; i < c->n; i++)
        mi_sampler_reset(&c->items[i]);
}

static void chain_destroy(MiSampler *s) {
    ChainCtx *c = (ChainCtx *)s->ctx;
    for (int i = 0; i < c->n; i++)
        mi_sampler_destroy(&c->items[i]);
    free(c->items);
    free(c);
}

static const MiSamplerVT chain_vt = {
    .name = "chain", .sample = chain_sample,
    .accept = chain_accept, .reset = chain_reset, .destroy = chain_destroy,
};

MiSampler mi_sampler_chain(MiSampler *samplers, int n) {
    ChainCtx *c = (ChainCtx *)malloc(sizeof(ChainCtx));
    MI_CHECK_OOM(c);
    c->items = (MiSampler *)malloc(n * sizeof(MiSampler));
    MI_CHECK_OOM(c->items);
    memcpy(c->items, samplers, n * sizeof(MiSampler));
    c->n = n;
    return (MiSampler){ .vt = &chain_vt, .ctx = c };
}
