/* generate.c — generation orchestrator with benchmarking */
#include "mi/generate.h"
#include "mi/ops.h"

int mi_generate(MiGenerateConfig *cfg,
                const int *prompt, int prompt_len,
                int *out_tokens) {
    MiModel  *m  = cfg->model;
    MiSampler *s = cfg->sampler;
    MiRng    *rng = cfg->rng;
    int V = m->cfg.vocab_size;

    size_t scratch_bytes = mi_model_scratch_size(&m->cfg);
    float *scratch = (float *)malloc(scratch_bytes);
    float *logits  = (float *)malloc(V * sizeof(float));
    MI_CHECK_OOM(scratch); MI_CHECK_OOM(logits);

    int generated = 0;

    /* ── Prefill prompt ── */
    for (int i = 0; i < prompt_len; i++) {
        mi_model_forward(m, prompt[i], logits, scratch);
    }

    /* ── Decode loop ── */
    int cur_token;
    /* First generated token comes from the prompt's last-position logits */
    {
        float *logits_copy = (float *)malloc(V * sizeof(float));
        MI_CHECK_OOM(logits_copy);
        memcpy(logits_copy, logits, V * sizeof(float));
        cur_token = mi_sampler_sample(s, logits_copy, V, rng);
        mi_sampler_accept(s, cur_token);
        free(logits_copy);

        out_tokens[generated++] = cur_token;
        if (cfg->on_token) {
            if (!cfg->on_token(cur_token, m->pos, cfg->callback_data))
                goto done;
        }
        if (cfg->eos_token >= 0 && cur_token == cfg->eos_token) goto done;
    }

    while (generated < cfg->max_tokens) {
        mi_model_forward(m, cur_token, logits, scratch);

        float *logits_copy = (float *)malloc(V * sizeof(float));
        MI_CHECK_OOM(logits_copy);
        memcpy(logits_copy, logits, V * sizeof(float));
        cur_token = mi_sampler_sample(s, logits_copy, V, rng);
        mi_sampler_accept(s, cur_token);
        free(logits_copy);

        out_tokens[generated++] = cur_token;

        if (cfg->on_token) {
            if (!cfg->on_token(cur_token, m->pos, cfg->callback_data))
                break;
        }
        if (cfg->eos_token >= 0 && cur_token == cfg->eos_token) break;
    }

done:
    free(scratch);
    free(logits);
    return generated;
}

/* ════════════ Benchmarking ════════════ */

MiGenStats mi_generate_bench(MiGenerateConfig *cfg,
                             const int *prompt, int prompt_len,
                             int *out_tokens) {
    MiGenStats stats;
    memset(&stats, 0, sizeof(stats));

    MiModel  *m  = cfg->model;
    MiSampler *s = cfg->sampler;
    MiRng    *rng = cfg->rng;
    int V = m->cfg.vocab_size;

    size_t scratch_bytes = mi_model_scratch_size(&m->cfg);
    float *scratch = (float *)malloc(scratch_bytes);
    float *logits  = (float *)malloc(V * sizeof(float));
    MI_CHECK_OOM(scratch); MI_CHECK_OOM(logits);

    /* ── Prefill benchmark ── */
    MiTimer timer;
    mi_timer_start(&timer);
    for (int i = 0; i < prompt_len; i++)
        mi_model_forward(m, prompt[i], logits, scratch);
    stats.prefill_s = mi_timer_elapsed_s(&timer);
    stats.prefill_tokens = prompt_len;

    /* ── Decode benchmark ── */
    int generated = 0;
    int cur_token;
    {
        float *lc = (float *)malloc(V * sizeof(float));
        MI_CHECK_OOM(lc);
        memcpy(lc, logits, V * sizeof(float));
        cur_token = mi_sampler_sample(s, lc, V, rng);
        mi_sampler_accept(s, cur_token);
        free(lc);
        out_tokens[generated++] = cur_token;
    }

    mi_timer_start(&timer);
    while (generated < cfg->max_tokens) {
        mi_model_forward(m, cur_token, logits, scratch);
        float *lc = (float *)malloc(V * sizeof(float));
        MI_CHECK_OOM(lc);
        memcpy(lc, logits, V * sizeof(float));
        cur_token = mi_sampler_sample(s, lc, V, rng);
        mi_sampler_accept(s, cur_token);
        free(lc);
        out_tokens[generated++] = cur_token;
        if (cfg->eos_token >= 0 && cur_token == cfg->eos_token) break;
    }
    stats.decode_s = mi_timer_elapsed_s(&timer);
    stats.decode_tokens = generated;

    stats.prefill_tok_s = (stats.prefill_s > 0)
        ? stats.prefill_tokens / stats.prefill_s : 0;
    stats.decode_tok_s  = (stats.decode_s > 0)
        ? stats.decode_tokens / stats.decode_s : 0;

    free(scratch);
    free(logits);
    return stats;
}

void mi_gen_stats_print(const MiGenStats *s) {
    printf("╔══════════════ Generation Stats ══════════════╗\n");
    printf("║ Prefill: %d tokens in %.4f s  (%.1f tok/s) ║\n",
           s->prefill_tokens, s->prefill_s, s->prefill_tok_s);
    printf("║ Decode:  %d tokens in %.4f s  (%.1f tok/s) ║\n",
           s->decode_tokens, s->decode_s, s->decode_tok_s);
    printf("╚══════════════════════════════════════════════╝\n");
}
