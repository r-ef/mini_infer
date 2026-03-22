/* speculative.c — speculative decoding (Leviathan et al. 2023)
 *
 * Algorithm:
 *   1. Draft model generates K tokens greedily
 *   2. Target model scores all K+1 positions in one batch
 *   3. Accept/reject each draft token:
 *        if p_target(x) ≥ p_draft(x)  → accept
 *        else                          → accept with prob p_target(x)/p_draft(x)
 *      on rejection, sample from max(0, p_target − p_draft)
 *   4. If all K accepted, sample one more from target's K+1 logits
 */
#include "mi/speculative.h"
#include "mi/ops.h"
#include "mi/sampling.h"

MiSpecDecoder mi_spec_create(
    void *draft_ctx,
    void (*draft_forward)(void *ctx, int token, float *logits),
    void (*draft_rollback)(void *ctx, int n),
    void (*draft_accept)(void *ctx, int token),
    int   vocab_size,
    int   n_draft)
{
    MiSpecDecoder sd;
    memset(&sd, 0, sizeof(sd));
    sd.cfg.n_draft    = n_draft;
    sd.draft_ctx      = draft_ctx;
    sd.draft_forward  = draft_forward;
    sd.draft_rollback = draft_rollback;
    sd.draft_accept   = draft_accept;
    sd.vocab_size     = vocab_size;
    return sd;
}

int mi_spec_step(
    MiSpecDecoder *spec,
    int            input_token,
    void (*target_forward_batch)(void *ctx, const int *tokens,
                                 int n, float *logits_out),
    void          *target_ctx,
    MiRng         *rng,
    int           *out_tokens)
{
    int K  = spec->cfg.n_draft;
    int V  = spec->vocab_size;

    /* Allocate workspace */
    float *draft_logits  = (float *)malloc((size_t)K * V * sizeof(float));
    int   *draft_tokens  = (int *)malloc(K * sizeof(int));
    float *target_logits = (float *)malloc((size_t)(K + 1) * V * sizeof(float));
    float *p_draft       = (float *)malloc(V * sizeof(float));
    float *p_target      = (float *)malloc(V * sizeof(float));
    float *residual      = (float *)malloc(V * sizeof(float));
    int   *verify_tokens = (int *)malloc((K + 1) * sizeof(int));
    MI_CHECK_OOM(draft_logits); MI_CHECK_OOM(draft_tokens);
    MI_CHECK_OOM(target_logits); MI_CHECK_OOM(p_draft);
    MI_CHECK_OOM(p_target); MI_CHECK_OOM(residual);
    MI_CHECK_OOM(verify_tokens);

    /* ── 1. Draft K tokens ── */
    int cur = input_token;
    for (int i = 0; i < K; i++) {
        spec->draft_forward(spec->draft_ctx, cur, draft_logits + i * V);
        /* Greedy draft */
        draft_tokens[i] = mi_argmax(draft_logits + i * V, V);
        cur = draft_tokens[i];
    }

    /* ── 2. Verify with target model ── */
    verify_tokens[0] = input_token;
    for (int i = 0; i < K; i++) verify_tokens[i + 1] = draft_tokens[i];
    target_forward_batch(target_ctx, verify_tokens, K + 1, target_logits);

    /* ── 3. Accept/reject ── */
    int accepted = 0;
    for (int i = 0; i < K; i++) {
        /* Convert to probabilities */
        memcpy(p_draft,  draft_logits  + i * V, V * sizeof(float));
        memcpy(p_target, target_logits + i * V, V * sizeof(float));
        mi_softmax(p_draft, V);
        mi_softmax(p_target, V);

        int tok = draft_tokens[i];
        float pt = p_target[tok];
        float pd = p_draft[tok];

        spec->cfg.total_drafted++;

        if (pd <= 0.0f || pt >= pd) {
            /* Accept deterministically */
            out_tokens[accepted++] = tok;
            spec->cfg.total_accepted++;
        } else {
            /* Accept with probability pt/pd */
            float r = mi_rng_float(rng);
            if (r < pt / pd) {
                out_tokens[accepted++] = tok;
                spec->cfg.total_accepted++;
            } else {
                /* Reject: sample from max(0, p_target − p_draft) */
                float sum = 0.0f;
                for (int j = 0; j < V; j++) {
                    residual[j] = MI_MAX(0.0f, p_target[j] - p_draft[j]);
                    sum += residual[j];
                }
                if (sum > 0.0f) {
                    float inv = 1.0f / sum;
                    for (int j = 0; j < V; j++) residual[j] *= inv;
                }
                /* Categorical sample from residual */
                float cdf = 0.0f;
                float rr = mi_rng_float(rng);
                int corrected = V - 1;
                for (int j = 0; j < V; j++) {
                    cdf += residual[j];
                    if (rr <= cdf) { corrected = j; break; }
                }
                out_tokens[accepted++] = corrected;

                /* Rollback draft model to the point of rejection */
                if (spec->draft_rollback)
                    spec->draft_rollback(spec->draft_ctx, accepted);
                goto done;
            }
        }
    }

    /* All K tokens accepted — sample one more from target's final logits */
    {
        memcpy(p_target, target_logits + K * V, V * sizeof(float));
        mi_softmax(p_target, V);
        float r = mi_rng_float(rng);
        float cdf = 0.0f;
        int bonus = V - 1;
        for (int j = 0; j < V; j++) {
            cdf += p_target[j];
            if (r <= cdf) { bonus = j; break; }
        }
        out_tokens[accepted++] = bonus;
    }

done:
    spec->cfg.total_steps++;

    /* Notify draft model about accepted tokens */
    if (spec->draft_accept) {
        for (int i = 0; i < accepted; i++)
            spec->draft_accept(spec->draft_ctx, out_tokens[i]);
    }

    free(draft_logits); free(draft_tokens);
    free(target_logits); free(p_draft);
    free(p_target); free(residual); free(verify_tokens);

    return accepted;
}

float mi_spec_acceptance_rate(const MiSpecDecoder *s) {
    if (s->cfg.total_drafted == 0) return 0.0f;
    return (float)s->cfg.total_accepted / (float)s->cfg.total_drafted;
}

void mi_spec_reset_stats(MiSpecDecoder *s) {
    s->cfg.total_accepted = 0;
    s->cfg.total_drafted  = 0;
    s->cfg.total_steps    = 0;
}
