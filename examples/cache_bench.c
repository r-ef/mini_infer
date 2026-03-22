
#include "mi.h"

static void bench_cache(const char *name, MiCache cache,
                        int n_layers, int kv_dim, int n_tokens) {
    MiRng rng = mi_rng_create(42);
    float *kv = (float *)malloc(kv_dim * sizeof(float));

    printf("── %s ──\n", name);


    MiTimer t;
    mi_timer_start(&t);
    for (int i = 0; i < n_tokens; i++) {
        for (int d = 0; d < kv_dim; d++) kv[d] = mi_rng_float(&rng);
        for (int l = 0; l < n_layers; l++)
            mi_cache_append(&cache, l, kv, kv);
    }
    double append_s = mi_timer_elapsed_s(&t);


    mi_timer_start(&t);
    for (int l = 0; l < n_layers; l++) {
        int len;
        const float *K = mi_cache_keys(&cache, l, &len);
        const float *V = mi_cache_values(&cache, l, &len);
        MI_UNUSED(K); MI_UNUSED(V);
    }
    double get_s = mi_timer_elapsed_s(&t);


    char stats[256];
    if (cache.vt->stats) cache.vt->stats(&cache, stats, sizeof(stats));
    else snprintf(stats, sizeof(stats), "(no stats)");

    printf("  Appended %d tokens: %.4f s (%.0f tok/s)\n",
           n_tokens, append_s, n_tokens / append_s);
    printf("  Get all layers: %.6f s\n", get_s);
    printf("  Size: %d  |  %s\n\n", mi_cache_size(&cache), stats);


    int half = mi_cache_size(&cache) / 2;
    mi_cache_truncate(&cache, half);
    printf("  After truncate to %d: size=%d\n\n", half, mi_cache_size(&cache));

    mi_cache_destroy(&cache);
    free(kv);
}

int main(void) {
    mi_log_level = MI_LOG_WARN;

    int n_layers  = 4;
    int n_kv_heads = 2;
    int d_head    = 32;
    int kv_dim    = n_kv_heads * d_head;
    int n_tokens  = 512;

    printf("═══════════════════════════════════════\n");
    printf("  KV Cache Benchmark\n");
    printf("  layers=%d  kv_heads=%d  d_head=%d  tokens=%d\n",
           n_layers, n_kv_heads, d_head, n_tokens);
    printf("═══════════════════════════════════════\n\n");

    bench_cache("Dense",
        mi_cache_dense(n_layers, n_kv_heads, d_head, 1024),
        n_layers, kv_dim, n_tokens);

    bench_cache("Paged (page_size=16)",
        mi_cache_paged(n_layers, n_kv_heads, d_head, 16, 256),
        n_layers, kv_dim, n_tokens);

    bench_cache("Sliding (window=128)",
        mi_cache_sliding(n_layers, n_kv_heads, d_head, 128),
        n_layers, kv_dim, n_tokens);

    bench_cache("Compressed (fresh=64)",
        mi_cache_compressed(n_layers, n_kv_heads, d_head, 1024, 64),
        n_layers, kv_dim, n_tokens);

    return 0;
}
