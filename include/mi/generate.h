
#ifndef MI_GENERATE_H
#define MI_GENERATE_H

#include "model.h"
#include "sampling.h"
#include "speculative.h"
#include "memory.h"

typedef bool (*MiTokenCallback)(int token_id, int pos, void *user_data);

typedef struct {
    MiModel        *model;
    MiSampler      *sampler;
    MiRng          *rng;


    MiSpecDecoder  *speculative;
    MiSinkConfig   *sink;
    MiH2O          *h2o;
    MiRAGConfig    *rag;


    int             max_tokens;
    int             eos_token;


    MiTokenCallback on_token;
    void           *callback_data;
} MiGenerateConfig;

int mi_generate(MiGenerateConfig *cfg,
                const int *prompt, int prompt_len,
                int *out_tokens);

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

#endif
