/* ═══════════════════════════════════════════════════════════════════
 * speculative.h — speculative decoding (Leviathan et al. 2023)
 *
 * Draft model proposes K tokens; target model verifies in one pass.
 * Acceptance-rejection guarantees target-model distribution.
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_SPECULATIVE_H
#define MI_SPECULATIVE_H

#include "base.h"

typedef struct MiModel MiModel;  /* forward decl, defined in model.h */

typedef struct {
    int   n_draft;           /* how many tokens to draft per step */
    /* Stats */
    int   total_accepted;
    int   total_drafted;
    int   total_steps;
} MiSpecConfig;

typedef struct {
    MiSpecConfig cfg;

    /* Draft model — a small/fast model (or the same model
     * running with fewer layers, skipped heads, etc.). */
    void *draft_ctx;

    /* draft_forward: given current token, fill logits[vocab_size]. */
    void (*draft_forward)(void *ctx, int token, float *logits);

    /* draft_rollback: truncate draft-model state to position n. */
    void (*draft_rollback)(void *ctx, int n);

    /* draft_accept: confirm that a token was accepted. */
    void (*draft_accept)(void *ctx, int token);

    int   vocab_size;
} MiSpecDecoder;

/* Create the speculative decoder. */
MiSpecDecoder mi_spec_create(
    void *draft_ctx,
    void (*draft_forward)(void *ctx, int token, float *logits),
    void (*draft_rollback)(void *ctx, int n),
    void (*draft_accept)(void *ctx, int token),
    int   vocab_size,
    int   n_draft);

/* Run one speculative step.
 *
 * target_forward_batch: run the *target* model on a batch of tokens
 *     tokens[n_tokens], writing logits for each position into
 *     logits_out[i * vocab_size] for i in 0..n_tokens-1.
 *
 * Returns: how many tokens were produced (1 … n_draft+1).
 * out_tokens[] is filled with the accepted token ids. */
int mi_spec_step(
    MiSpecDecoder *spec,
    int            input_token,
    void (*target_forward_batch)(void *ctx, const int *tokens,
                                 int n, float *logits_out),
    void          *target_ctx,
    MiRng         *rng,
    int           *out_tokens);

float mi_spec_acceptance_rate(const MiSpecDecoder *s);
void  mi_spec_reset_stats(MiSpecDecoder *s);

#endif /* MI_SPECULATIVE_H */
