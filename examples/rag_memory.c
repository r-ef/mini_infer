/* rag_memory.c — demonstrate retrieval-augmented memory, attention sinks,
 *                  H2O eviction, and token merging */
#include "mi.h"

int main(void) {
    mi_log_level = MI_LOG_WARN;

    int kv_dim = 16;
    int max_seq = 32;

    MiRng rng = mi_rng_create(77);

    /* ════════════ Vector Store + RAG ════════════ */
    printf("═══════════════════════════════════════\n");
    printf("  RAG Memory Demo (dim=%d)\n", kv_dim);
    printf("═══════════════════════════════════════\n\n");

    MiVectorStore store = mi_vstore_create(kv_dim, 100);

    /* Add some "documents" */
    for (int doc = 0; doc < 20; doc++) {
        float emb[kv_dim];
        for (int d = 0; d < kv_dim; d++)
            emb[d] = mi_rng_normal(&rng);
        mi_vstore_add(&store, emb, doc);
    }

    /* Query */
    float query[kv_dim];
    memcpy(query, store.embeddings, kv_dim * sizeof(float));
    /* Perturb slightly */
    for (int d = 0; d < kv_dim; d++)
        query[d] += 0.1f * mi_rng_normal(&rng);

    int indices[5];
    float scores[5];
    int found = mi_vstore_search(&store, query, 5, indices, scores);
    printf("Top-%d search results:\n", found);
    for (int i = 0; i < found; i++)
        printf("  doc_id=%d  score=%.4f\n", store.doc_ids[indices[i]], scores[i]);

    /* RAG augmentation */
    float K[max_seq * kv_dim];
    float V[max_seq * kv_dim];
    memset(K, 0, sizeof(K));
    memset(V, 0, sizeof(V));
    int seq_len = 5;  /* pretend we have 5 existing KV entries */
    for (int i = 0; i < seq_len * kv_dim; i++) {
        K[i] = mi_rng_float(&rng);
        V[i] = mi_rng_float(&rng);
    }

    MiRAGConfig rag = {
        .store = &store,
        .n_retrieve = 3,
        .gate_threshold = -1.0f,  /* accept all */
    };
    int injected = mi_rag_augment(&rag, query, K, V, &seq_len, kv_dim, max_seq);
    printf("\nRAG: injected %d entries, seq_len now %d\n\n", injected, seq_len);

    /* ════════════ Attention Sink ════════════ */
    printf("═══ Attention Sink Demo ═══\n");
    int sink_len = 20;
    float sK[32 * kv_dim], sV[32 * kv_dim];
    for (int i = 0; i < sink_len * kv_dim; i++) {
        sK[i] = (float)(i % kv_dim);
        sV[i] = (float)(i % kv_dim) * 0.5f;
    }
    printf("Before: seq_len=%d\n", sink_len);
    MiSinkConfig sink = { .sink_size = 2, .window_size = 4 };
    mi_sink_evict(sK, sV, &sink_len, kv_dim, &sink);
    printf("After sink evict (sink=2, window=4): seq_len=%d\n\n", sink_len);

    /* ════════════ H2O ════════════ */
    printf("═══ H2O Eviction Demo ═══\n");
    MiH2O h2o = mi_h2o_create(64, 8);
    /* Simulate some attention patterns */
    float attn_weights[] = {0.3f, 0.01f, 0.02f, 0.15f, 0.01f,
                            0.01f, 0.2f, 0.05f, 0.1f, 0.15f};
    mi_h2o_update(&h2o, attn_weights, 10);

    /* Add more attention to positions 0, 3, 6, 9 (heavy hitters) */
    float attn2[] = {0.25f, 0.02f, 0.03f, 0.2f, 0.02f,
                     0.03f, 0.18f, 0.02f, 0.05f, 0.2f};
    mi_h2o_update(&h2o, attn2, 10);

    int keep[8];
    int n_keep = mi_h2o_select(&h2o, 10, keep);
    printf("H2O selected %d positions (budget=8): ", n_keep);
    for (int i = 0; i < n_keep; i++) printf("%d ", keep[i]);
    printf("\n\n");
    mi_h2o_free(&h2o);

    /* ════════════ Token Merge ════════════ */
    printf("═══ Token Merge Demo ═══\n");
    float mK[8 * kv_dim], mV[8 * kv_dim];
    for (int i = 0; i < 8; i++) {
        for (int d = 0; d < kv_dim; d++) {
            mK[i * kv_dim + d] = mi_rng_normal(&rng);
            mV[i * kv_dim + d] = mi_rng_normal(&rng);
        }
    }
    /* Make entries 2 and 3 very similar */
    memcpy(mK + 3 * kv_dim, mK + 2 * kv_dim, kv_dim * sizeof(float));
    mK[3 * kv_dim] += 0.01f;  /* tiny perturbation */
    memcpy(mV + 3 * kv_dim, mV + 2 * kv_dim, kv_dim * sizeof(float));

    int merge_len = 8;
    MiMergeConfig merge = { .threshold = 0.99f, .min_sep = 0 };
    int new_len = mi_token_merge(mK, mV, merge_len, kv_dim, &merge);
    printf("Merged %d → %d entries (threshold=0.99)\n\n", merge_len, new_len);

    mi_vstore_free(&store);
    printf("Done.\n");
    return 0;
}
