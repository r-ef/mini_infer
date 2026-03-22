
#ifndef MI_SPECULATIVE_H
#define MI_SPECULATIVE_H

#include "base.h"

typedef struct MiModel MiModel;

typedef struct {
    int   n_draft;

    int   total_accepted;
    int   total_drafted;
    int   total_steps;
} MiSpecConfig;

typedef struct {
    MiSpecConfig cfg;


    void *draft_ctx;


    void (*draft_forward)(void *ctx, int token, float *logits);


    void (*draft_rollback)(void *ctx, int n);


    void (*draft_accept)(void *ctx, int token);

    int   vocab_size;
} MiSpecDecoder;

MiSpecDecoder mi_spec_create(
    void *draft_ctx,
    void (*draft_forward)(void *ctx, int token, float *logits),
    void (*draft_rollback)(void *ctx, int n),
    void (*draft_accept)(void *ctx, int token),
    int   vocab_size,
    int   n_draft);

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

#endif
