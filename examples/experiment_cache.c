/* experiment_cache.c — measure compressed-cache quality degradation
 *
 * Runs the same prompt through dense (ground truth) and compressed caches
 * with varying fresh_count.  Uses GREEDY sampling so divergence is purely
 * caused by int8 quantisation of old KV entries.
 *
 * Metrics per fresh_count:
 *   - first divergent token position
 *   - argmax match rate (%)
 *   - mean logit cosine similarity
 *   - memory savings
 *
 * Usage:
 *   ./examples/experiment_cache ./models/smollm [gen_tokens]
 */
#include "mi.h"

#define MAX_GEN 256

/* Run generation with a specific cache, capturing logits + tokens. */
static void run_generation(MiModel *model, MiCache cache,
                           const int *prompt, int prompt_len,
                           int gen_tokens,
                           int *out_tokens, float *out_logits /* [gen*V] */) {
    int V = model->cfg.vocab_size;

    /* Destroy old cache, swap in new one, reset position */
    mi_cache_destroy(&model->cache);
    model->cache = cache;
    model->pos   = 0;

    size_t scratch_bytes = mi_model_scratch_size(&model->cfg);
    float *scratch = (float *)malloc(scratch_bytes);
    float *logits  = (float *)malloc(V * sizeof(float));
    MI_CHECK_OOM(scratch); MI_CHECK_OOM(logits);

    /* Prefill */
    for (int i = 0; i < prompt_len; i++)
        mi_model_forward(model, prompt[i], logits, scratch);

    /* First token from prompt logits */
    out_tokens[0] = mi_argmax(logits, V);
    memcpy(out_logits, logits, V * sizeof(float));

    /* Decode */
    for (int i = 1; i < gen_tokens; i++) {
        mi_model_forward(model, out_tokens[i-1], logits, scratch);
        out_tokens[i] = mi_argmax(logits, V);
        memcpy(out_logits + (size_t)i * V, logits, V * sizeof(float));
    }

    free(scratch);
    free(logits);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [gen_tokens]\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    int gen_tokens = (argc > 2) ? atoi(argv[2]) : 128;
    if (gen_tokens > MAX_GEN) gen_tokens = MAX_GEN;

    mi_log_level = MI_LOG_INFO;

    /* ── Load model ── */
    char path[512];
    snprintf(path, sizeof(path), "%s/model.bin", model_dir);
    MiModel model = mi_model_load_file(path);
    mi_model_set_attention(&model, mi_attention_flash());

    snprintf(path, sizeof(path), "%s/tokenizer.bin", model_dir);
    MiTokenizer tok = mi_tokenizer_load_bpe(path);

    mi_log_level = MI_LOG_WARN;

    int V = model.cfg.vocab_size;
    int L = model.cfg.n_layers;
    int kv_h = model.cfg.n_kv_heads;
    int dh = model.cfg.d_head;

    /* ── Encode prompt ── */
    const char *prompt_text = "In a groundbreaking discovery, researchers at the "
        "institute have found that the fundamental nature of consciousness may be "
        "linked to quantum processes occurring within neural microtubules. This "
        "theory suggests that";
    int prompt_tokens[512];
    prompt_tokens[0] = tok.bos_token;
    int prompt_len = mi_tokenizer_encode(&tok, prompt_text,
                                          prompt_tokens + 1, 511) + 1;

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  KV Cache Compression Experiment                             ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Model:   d=%d  L=%d  kv_heads=%d  d_head=%d                 ║\n",
           model.cfg.d_model, L, kv_h, dh);
    printf("║  Prompt:  %d tokens                                          ║\n",
           prompt_len);
    printf("║  Generate: %d tokens (greedy / argmax)                       ║\n",
           gen_tokens);
    printf("║  Total context: %d tokens                                    ║\n",
           prompt_len + gen_tokens);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* ── Allocate result buffers ── */
    int   *dense_tokens = (int *)malloc(gen_tokens * sizeof(int));
    float *dense_logits = (float *)malloc((size_t)gen_tokens * V * sizeof(float));
    int   *comp_tokens  = (int *)malloc(gen_tokens * sizeof(int));
    float *comp_logits  = (float *)malloc((size_t)gen_tokens * V * sizeof(float));
    MI_CHECK_OOM(dense_tokens); MI_CHECK_OOM(dense_logits);
    MI_CHECK_OOM(comp_tokens);  MI_CHECK_OOM(comp_logits);

    /* ── Run dense baseline ── */
    int max_seq = prompt_len + gen_tokens + 16;
    MiCache dense_cache = mi_cache_dense(L, kv_h, dh, max_seq);

    printf("Running dense cache (ground truth)...\n");
    MiTimer timer;
    mi_timer_start(&timer);
    run_generation(&model, dense_cache, prompt_tokens, prompt_len,
                   gen_tokens, dense_tokens, dense_logits);
    double dense_time = mi_timer_elapsed_s(&timer);
    printf("  %.2f s (%.1f tok/s)\n\n", dense_time,
           gen_tokens / dense_time);

    /* Print dense output */
    char *dense_text = mi_tokenizer_decode(&tok, dense_tokens, gen_tokens);
    printf("Dense output:\n  %.120s...\n\n", dense_text);
    free(dense_text);

    /* model now owns dense_cache — run_generation will destroy it on swap */

    /* ── Run compressed cache with various fresh_count ── */
    int fresh_values[] = {4, 8, 16, 32, 64, 128};
    int n_fresh = MI_ARRAY_LEN(fresh_values);

    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  fresh_count │ 1st diverge │ match%%  │ mean cos_sim │ mem ratio\n");
    printf("──────────────┼─────────────┼─────────┼──────────────┼──────────\n");

    for (int fi = 0; fi < n_fresh; fi++) {
        int fresh = fresh_values[fi];
        MiCache ccache = mi_cache_compressed(L, kv_h, dh, max_seq, fresh);

        run_generation(&model, ccache, prompt_tokens, prompt_len,
                       gen_tokens, comp_tokens, comp_logits);

        /* ── Compute metrics ── */
        int first_diverge = gen_tokens; /* means no divergence */
        int matches = 0;
        double cos_sum = 0.0;

        for (int i = 0; i < gen_tokens; i++) {
            if (dense_tokens[i] == comp_tokens[i])
                matches++;
            else if (first_diverge == gen_tokens)
                first_diverge = i;

            /* Cosine similarity between logit vectors */
            float *dl = dense_logits + (size_t)i * V;
            float *cl = comp_logits  + (size_t)i * V;
            cos_sum += mi_vec_cosine(dl, cl, V);
        }

        double match_pct = 100.0 * matches / gen_tokens;
        double mean_cos  = cos_sum / gen_tokens;

        /* Memory: compressed uses int8 for old entries, fp32 for fresh */
        int total_ctx = prompt_len + gen_tokens;
        int compressed_entries = MI_MAX(0, total_ctx - fresh);
        int kv_dim = kv_h * dh;
        double dense_bytes = (double)total_ctx * kv_dim * 4.0 * 2 * L;
        double comp_bytes  = (double)compressed_entries * kv_dim * 1.0 * 2 * L  /* int8 */
                           + (double)MI_MIN(fresh, total_ctx) * kv_dim * 4.0 * 2 * L; /* fp32 */
        double mem_ratio = comp_bytes / dense_bytes;

        char div_str[32];
        if (first_diverge == gen_tokens)
            snprintf(div_str, sizeof(div_str), "  never   ");
        else
            snprintf(div_str, sizeof(div_str), "  tok %3d  ", first_diverge);

        printf("  %9d  │%s│ %5.1f%%  │   %8.6f   │  %.2fx\n",
               fresh, div_str, match_pct, mean_cos, mem_ratio);
    }

    printf("──────────────┴─────────────┴─────────┴──────────────┴──────────\n\n");

    /* ── Detailed divergence analysis for smallest fresh_count ── */
    {
        int fresh = fresh_values[0];
        MiCache ccache = mi_cache_compressed(L, kv_h, dh, max_seq, fresh);
        run_generation(&model, ccache, prompt_tokens, prompt_len,
                       gen_tokens, comp_tokens, comp_logits);

        printf("Divergence detail (fresh=%d):\n", fresh);
        int shown = 0;
        for (int i = 0; i < gen_tokens && shown < 10; i++) {
            if (dense_tokens[i] != comp_tokens[i]) {
                printf("  pos %3d: dense=\"%s\"  compressed=\"%s\"  cos=%.4f\n",
                       i,
                       mi_tokenizer_token(&tok, dense_tokens[i]),
                       mi_tokenizer_token(&tok, comp_tokens[i]),
                       mi_vec_cosine(dense_logits + (size_t)i * V,
                                     comp_logits  + (size_t)i * V, V));
                shown++;
            }
        }
        if (shown == 0) printf("  (no divergence)\n");
    }

    printf("\n");

    /* ── Also compare sliding window cache ── */
    printf("Bonus: sliding window cache comparison:\n");
    printf("  window_size │ 1st diverge │ match%%  │ mean cos_sim\n");
    printf("──────────────┼─────────────┼─────────┼─────────────\n");

    int windows[] = {32, 64, 128, 256};
    for (int wi = 0; wi < MI_ARRAY_LEN(windows); wi++) {
        int w = windows[wi];
        MiCache scache = mi_cache_sliding(L, kv_h, dh, w);

        run_generation(&model, scache, prompt_tokens, prompt_len,
                       gen_tokens, comp_tokens, comp_logits);

        int first_div = gen_tokens;
        int matches = 0;
        double cos_sum = 0.0;
        for (int i = 0; i < gen_tokens; i++) {
            if (dense_tokens[i] == comp_tokens[i]) matches++;
            else if (first_div == gen_tokens) first_div = i;
            cos_sum += mi_vec_cosine(dense_logits + (size_t)i * V,
                                     comp_logits  + (size_t)i * V, V);
        }

        char div_str[32];
        if (first_div == gen_tokens)
            snprintf(div_str, sizeof(div_str), "  never   ");
        else
            snprintf(div_str, sizeof(div_str), "  tok %3d  ", first_div);

        printf("  %9d  │%s│ %5.1f%%  │   %8.6f\n",
               w, div_str, 100.0 * matches / gen_tokens,
               cos_sum / gen_tokens);
    }
    printf("──────────────┴─────────────┴─────────┴─────────────\n");

    /* Cleanup */
    free(dense_tokens); free(dense_logits);
    free(comp_tokens);  free(comp_logits);
    mi_tokenizer_free(&tok);
    mi_model_free(&model);

    return 0;
}
