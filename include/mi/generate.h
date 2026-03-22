/* ═══════════════════════════════════════════════════════════════════
 * generate.h — generation orchestrator
 *
 * Wires model + sampler + optional speculative / memory management
 * into a single mi_generate() call with streaming support.
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_GENERATE_H
#define MI_GENERATE_H

#include "model.h"
#include "sampling.h"
#include "speculative.h"
#include "memory.h"

/* Streaming callback: called for each token as it is generated.
 * Return false to stop generation early. */
typedef bool (*MiTokenCallback)(int token_id, int pos, void *user_data);

typedef struct {
    MiModel        *model;
    MiSampler      *sampler;
    MiRng          *rng;

    /* Optional components — set NULL to disable */
    MiSpecDecoder  *speculative;
    MiSinkConfig   *sink;
    MiH2O          *h2o;
    MiRAGConfig    *rag;

    /* Limits */
    int             max_tokens;
    int             eos_token;       /* -1 to disable */

    /* Streaming */
    MiTokenCallback on_token;
    void           *callback_data;
} MiGenerateConfig;

/* Generate tokens auto-regressively.
 * Returns number of tokens written to out_tokens. */
int mi_generate(MiGenerateConfig *cfg,
                const int *prompt, int prompt_len,
                int *out_tokens);

/* ── Benchmarking ── */
typedef struct {
    double prefill_s;
    double decode_s;
    int    prefill_tokens;
    int    decode_tokens;
    double prefill_tok_s;
    double decode_tok_s;
} MiGenStats;

MiGenStats mi_generate_bench(MiGenerateConfig *cfg,
                             const int *prompt, int prompt_len,
                             int *out_tokens);
void mi_gen_stats_print(const MiGenStats *s);

#endif /* MI_GENERATE_H */
