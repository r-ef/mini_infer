
#include "mi.h"

typedef struct {
    MiTokenizer *tok;
    int          count;
} StreamCtx;

static bool stream_token(int token_id, int pos, void *data) {
    StreamCtx *ctx = (StreamCtx *)data;
    MI_UNUSED(pos);
    const char *s = mi_tokenizer_token(ctx->tok, token_id);
    printf("%s", s);
    fflush(stdout);
    ctx->count++;
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [\"prompt\"] [max_tokens]\n", argv[0]);
        return 1;
    }

    const char *model_dir  = argv[1];
    const char *prompt     = (argc > 2) ? argv[2] : "The meaning of life is";
    int         max_tokens = (argc > 3) ? atoi(argv[3]) : 64;

    mi_log_level = MI_LOG_INFO;


    char model_path[512], tok_path[512];
    snprintf(model_path, sizeof(model_path), "%s/model.bin", model_dir);
    snprintf(tok_path,   sizeof(tok_path),   "%s/tokenizer.bin", model_dir);


    MI_INFO("loading model from %s", model_path);
    MiModel model = mi_model_load_file(model_path);


    mi_model_set_attention(&model, mi_attention_flash());


    MI_INFO("loading tokenizer from %s", tok_path);
    MiTokenizer tok = mi_tokenizer_load_bpe(tok_path);


    int prompt_tokens[1024];
    prompt_tokens[0] = tok.bos_token;
    int n_prompt = mi_tokenizer_encode(&tok, prompt,
                                        prompt_tokens + 1, 1023) + 1;
    MI_INFO("prompt: \"%s\" → %d tokens (with BOS)", prompt, n_prompt);


    MiRng rng = mi_rng_create(42);


    MiSampler chain_items[2] = {
        mi_sampler_repetition(1.1f, 64),
        mi_sampler_top_p(0.9f, 0.7f),
    };
    MiSampler sampler = mi_sampler_chain(chain_items, 2);

    int *out_tokens = (int *)malloc(max_tokens * sizeof(int));
    MI_CHECK_OOM(out_tokens);

    StreamCtx stream_ctx = { .tok = &tok, .count = 0 };

    MiGenerateConfig cfg = {
        .model       = &model,
        .sampler     = &sampler,
        .rng         = &rng,
        .max_tokens  = max_tokens,
        .eos_token   = tok.eos_token,
        .on_token    = stream_token,
        .callback_data = &stream_ctx,
    };


    printf("\n─── prompt: \"%s\" ───\n\n", prompt);

    MiGenStats stats = mi_generate_bench(&cfg, prompt_tokens, n_prompt,
                                          out_tokens);


    char *prompt_text = mi_tokenizer_decode(&tok, prompt_tokens, n_prompt);
    char *gen_text    = mi_tokenizer_decode(&tok, out_tokens, stats.decode_tokens);
    printf("%s%s\n\n", prompt_text, gen_text);
    free(prompt_text);
    free(gen_text);

    mi_gen_stats_print(&stats);


    free(out_tokens);
    mi_sampler_destroy(&sampler);
    mi_tokenizer_free(&tok);
    mi_model_free(&model);

    return 0;
}
