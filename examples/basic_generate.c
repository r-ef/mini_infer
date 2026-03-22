
#include "mi.h"

static bool print_token(int token_id, int pos, void *data) {
    MiTokenizer *tok = (MiTokenizer *)data;
    printf("%s ", mi_tokenizer_token(tok, token_id));
    fflush(stdout);
    MI_UNUSED(pos);
    return true;
}

int main(void) {
    mi_log_level = MI_LOG_WARN;


    MiModelConfig cfg = {
        .d_model     = 64,
        .n_heads     = 4,
        .n_kv_heads  = 2,
        .d_head      = 16,
        .d_ff        = 128,
        .n_layers    = 2,
        .vocab_size  = 32,
        .max_seq_len = 256,
        .norm_eps    = 1e-5f,
        .rope_theta  = 10000.0f,
        .ffn_type    = MI_FFN_SWIGLU,
    };


    MiRng rng = mi_rng_create(42);
    MiModel model = mi_model_create(cfg);
    mi_model_init_random(&model, &rng);


    MiTokenizer tok = mi_tokenizer_create(cfg.vocab_size);
    const char *words[] = {
        "the","cat","sat","on","mat","a","big","small","red","blue",
        "dog","ran","fast","slow","and","or","but","with","in","at",
        "to","from","up","down","left","right","is","was","be","go",
        "end","<eos>"
    };
    for (int i = 0; i < cfg.vocab_size; i++)
        mi_tokenizer_set(&tok, i, words[i], 0.0f);


    int prompt[] = {0, 1, 2, 3, 0, 4};
    int prompt_len = 6;
    int max_gen = 20;
    int *out = (int *)malloc(max_gen * sizeof(int));


    printf("═══ Prompt: ");
    for (int i = 0; i < prompt_len; i++)
        printf("%s ", mi_tokenizer_token(&tok, prompt[i]));
    printf("═══\n\n");

    struct { const char *name; MiSampler s; } samplers[] = {
        {"greedy",       mi_sampler_greedy()},
        {"top_k(5,0.8)", mi_sampler_top_k(5, 0.8f)},
        {"top_p(0.9,1)", mi_sampler_top_p(0.9f, 1.0f)},
        {"min_p(0.05,1)",mi_sampler_min_p(0.05f, 1.0f)},
        {"typical(0.95,1)", mi_sampler_typical(0.95f, 1.0f)},
        {"mirostat(5,0.1)", mi_sampler_mirostat_v2(5.0f, 0.1f)},
    };
    int n_samplers = MI_ARRAY_LEN(samplers);

    for (int si = 0; si < n_samplers; si++) {
        mi_model_reset(&model);
        mi_sampler_reset(&samplers[si].s);
        MiRng gen_rng = mi_rng_create(123);

        MiGenerateConfig gen = {
            .model     = &model,
            .sampler   = &samplers[si].s,
            .rng       = &gen_rng,
            .max_tokens = max_gen,
            .eos_token  = 31,
            .on_token   = print_token,
            .callback_data = &tok,
        };

        printf("[%-16s] ", samplers[si].name);
        int n = mi_generate(&gen, prompt, prompt_len, out);
        printf(" (%d tokens)\n", n);
    }


    printf("\n═══ Benchmark ═══\n");
    mi_model_reset(&model);
    MiSampler greedy = mi_sampler_greedy();
    MiRng bench_rng = mi_rng_create(0);
    MiGenerateConfig bench_cfg = {
        .model = &model, .sampler = &greedy, .rng = &bench_rng,
        .max_tokens = max_gen, .eos_token = -1,
    };
    MiGenStats stats = mi_generate_bench(&bench_cfg, prompt, prompt_len, out);
    mi_gen_stats_print(&stats);


    for (int i = 0; i < n_samplers; i++)
        mi_sampler_destroy(&samplers[i].s);
    mi_sampler_destroy(&greedy);
    free(out);
    mi_tokenizer_free(&tok);
    mi_model_free(&model);

    return 0;
}
