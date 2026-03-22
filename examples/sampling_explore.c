
#include "mi.h"

static void print_probs(const float *logits, int n, const char *name) {
    float probs[n];
    memcpy(probs, logits, n * sizeof(float));
    mi_softmax(probs, n);
    printf("  %-20s → ", name);
    for (int i = 0; i < n; i++) printf("%.3f ", probs[i]);
    printf("\n");
}

int main(void) {
    mi_log_level = MI_LOG_NONE;

    int V = 10;

    float base_logits[10] = {3.0f, 2.5f, 1.0f, 0.5f, 0.2f,
                              -0.5f, -1.0f, -2.0f, -3.0f, -4.0f};

    printf("═══════════════════════════════════════════════\n");
    printf("  Sampling Method Comparison (V=%d)\n", V);
    printf("═══════════════════════════════════════════════\n\n");

    printf("Base logits: ");
    for (int i = 0; i < V; i++) printf("%.1f ", base_logits[i]);
    printf("\n");
    print_probs(base_logits, V, "base probs");
    printf("\n");


    struct { const char *name; MiSampler s; } samplers[] = {
        {"greedy",          mi_sampler_greedy()},
        {"top_k(3, 1.0)",   mi_sampler_top_k(3, 1.0f)},
        {"top_k(3, 0.5)",   mi_sampler_top_k(3, 0.5f)},
        {"top_p(0.8, 1.0)", mi_sampler_top_p(0.8f, 1.0f)},
        {"min_p(0.1, 1.0)", mi_sampler_min_p(0.1f, 1.0f)},
        {"typical(0.9,1.0)",mi_sampler_typical(0.9f, 1.0f)},
        {"mirostat(3,0.1)", mi_sampler_mirostat_v2(3.0f, 0.1f)},
    };
    int n_samplers = MI_ARRAY_LEN(samplers);

    int N = 1000;
    MiRng rng = mi_rng_create(42);
    float logits_buf[10];

    printf("Token frequency over %d samples:\n", N);
    printf("%-20s  ", "Sampler");
    for (int i = 0; i < V; i++) printf(" T%-3d", i);
    printf("  entropy\n");
    printf("────────────────────  ");
    for (int i = 0; i < V; i++) printf("─────");
    printf("  ───────\n");

    for (int si = 0; si < n_samplers; si++) {
        int counts[10] = {0};
        mi_sampler_reset(&samplers[si].s);

        for (int trial = 0; trial < N; trial++) {
            memcpy(logits_buf, base_logits, V * sizeof(float));
            int tok = mi_sampler_sample(&samplers[si].s, logits_buf, V, &rng);
            if (tok >= 0 && tok < V) counts[tok]++;
            mi_sampler_accept(&samplers[si].s, tok);
        }


        printf("%-20s  ", samplers[si].name);
        float entropy = 0.0f;
        for (int i = 0; i < V; i++) {
            float freq = (float)counts[i] / (float)N;
            printf("%.3f ", freq);
            if (freq > 0) entropy -= freq * log2f(freq);
        }
        printf("  %.2f bits\n", entropy);
    }

    printf("\n");
    for (int i = 0; i < n_samplers; i++)
        mi_sampler_destroy(&samplers[i].s);


    printf("═══ Chain: repetition(1.3, 5) → top_p(0.9, 0.8) ═══\n");
    MiSampler chain_items[2] = {
        mi_sampler_repetition(1.3f, 5),
        mi_sampler_top_p(0.9f, 0.8f),
    };
    MiSampler chain = mi_sampler_chain(chain_items, 2);

    printf("Sequence: ");
    for (int i = 0; i < 30; i++) {
        memcpy(logits_buf, base_logits, V * sizeof(float));
        int tok = mi_sampler_sample(&chain, logits_buf, V, &rng);
        mi_sampler_accept(&chain, tok);
        printf("%d ", tok);
    }
    printf("\n");

    mi_sampler_destroy(&chain);
    return 0;
}
