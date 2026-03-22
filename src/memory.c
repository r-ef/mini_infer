
#include "mi/memory.h"
#include "mi/ops.h"

void mi_sink_evict(float *K, float *V, int *seq_len,
                   int kv_dim, const MiSinkConfig *cfg) {
    int n = *seq_len;
    int keep_total = cfg->sink_size + cfg->window_size;
    if (n <= keep_total) return;


    int window_start = n - cfg->window_size;


    memmove(K + (size_t)cfg->sink_size * kv_dim,
            K + (size_t)window_start * kv_dim,
            (size_t)cfg->window_size * kv_dim * sizeof(float));
    memmove(V + (size_t)cfg->sink_size * kv_dim,
            V + (size_t)window_start * kv_dim,
            (size_t)cfg->window_size * kv_dim * sizeof(float));

    *seq_len = keep_total;
}

MiH2O mi_h2o_create(int max_seq, int budget) {
    MiH2O h;
    h.max_seq   = max_seq;
    h.budget    = budget;
    h.n_current = 0;
    h.cum_attn  = (float *)calloc(max_seq, sizeof(float));
    MI_CHECK_OOM(h.cum_attn);
    return h;
}

void mi_h2o_free(MiH2O *h) {
    free(h->cum_attn);
    h->cum_attn = NULL;
}

void mi_h2o_update(MiH2O *h, const float *weights, int seq_len) {
    for (int i = 0; i < seq_len && i < h->max_seq; i++)
        h->cum_attn[i] += weights[i];
    h->n_current = seq_len;
}

int mi_h2o_select(const MiH2O *h, int seq_len,
                  int *keep_indices) {
    int n_keep = MI_MIN(seq_len, h->budget);


    int *indices = (int *)malloc(seq_len * sizeof(int));
    MI_CHECK_OOM(indices);
    for (int i = 0; i < seq_len; i++) indices[i] = i;


    for (int i = 0; i < n_keep; i++) {
        int best = i;
        for (int j = i + 1; j < seq_len; j++)
            if (h->cum_attn[indices[j]] > h->cum_attn[indices[best]])
                best = j;
        int tmp = indices[i]; indices[i] = indices[best]; indices[best] = tmp;
    }


    for (int i = 1; i < n_keep; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && indices[j] > key) {
            indices[j + 1] = indices[j]; j--;
        }
        indices[j + 1] = key;
    }

    memcpy(keep_indices, indices, n_keep * sizeof(int));
    free(indices);
    return n_keep;
}

void mi_h2o_compact(float *K, float *V, int *seq_len, int kv_dim,
                    const int *keep_indices, int n_keep) {
    for (int i = 0; i < n_keep; i++) {
        int src = keep_indices[i];
        if (src != i) {
            memcpy(K + (size_t)i * kv_dim,
                   K + (size_t)src * kv_dim,
                   kv_dim * sizeof(float));
            memcpy(V + (size_t)i * kv_dim,
                   V + (size_t)src * kv_dim,
                   kv_dim * sizeof(float));
        }
    }
    *seq_len = n_keep;
}

int mi_token_merge(float *K, float *V, int seq_len, int kv_dim,
                   const MiMergeConfig *cfg) {
    if (seq_len < 2) return seq_len;

    int write = 0;
    int i = 0;
    while (i < seq_len) {
        if (i + 1 < seq_len && (i + 1 - write) >= cfg->min_sep) {
            float sim = mi_vec_cosine(K + (size_t)i * kv_dim,
                                      K + (size_t)(i + 1) * kv_dim,
                                      kv_dim);
            if (sim > cfg->threshold) {

                float *k_dst = K + (size_t)write * kv_dim;
                float *v_dst = V + (size_t)write * kv_dim;
                const float *k0 = K + (size_t)i * kv_dim;
                const float *k1 = K + (size_t)(i + 1) * kv_dim;
                const float *v0 = V + (size_t)i * kv_dim;
                const float *v1 = V + (size_t)(i + 1) * kv_dim;
                for (int d = 0; d < kv_dim; d++) {
                    k_dst[d] = 0.5f * (k0[d] + k1[d]);
                    v_dst[d] = 0.5f * (v0[d] + v1[d]);
                }
                write++;
                i += 2;
                continue;
            }
        }

        if (write != i) {
            memcpy(K + (size_t)write * kv_dim,
                   K + (size_t)i * kv_dim, kv_dim * sizeof(float));
            memcpy(V + (size_t)write * kv_dim,
                   V + (size_t)i * kv_dim, kv_dim * sizeof(float));
        }
        write++;
        i++;
    }
    return write;
}

MiVectorStore mi_vstore_create(int dim, int capacity) {
    MiVectorStore vs;
    vs.dim        = dim;
    vs.size       = 0;
    vs.capacity   = capacity;
    vs.embeddings = (float *)calloc((size_t)capacity * dim, sizeof(float));
    vs.doc_ids    = (int *)calloc(capacity, sizeof(int));
    MI_CHECK_OOM(vs.embeddings); MI_CHECK_OOM(vs.doc_ids);
    return vs;
}

void mi_vstore_free(MiVectorStore *vs) {
    free(vs->embeddings); free(vs->doc_ids);
    vs->embeddings = NULL; vs->doc_ids = NULL;
}

void mi_vstore_add(MiVectorStore *vs, const float *emb, int doc_id) {
    MI_ASSERT(vs->size < vs->capacity, "vector store full");
    memcpy(vs->embeddings + (size_t)vs->size * vs->dim,
           emb, vs->dim * sizeof(float));
    vs->doc_ids[vs->size] = doc_id;
    vs->size++;
}

int mi_vstore_search(const MiVectorStore *vs, const float *query,
                     int k, int *out_indices, float *out_scores) {
    int n = vs->size;
    if (n == 0) return 0;
    k = MI_MIN(k, n);


    float *sims = (float *)malloc(n * sizeof(float));
    int *indices = (int *)malloc(n * sizeof(int));
    MI_CHECK_OOM(sims); MI_CHECK_OOM(indices);

    for (int i = 0; i < n; i++) {
        sims[i] = mi_vec_cosine(query,
                                vs->embeddings + (size_t)i * vs->dim,
                                vs->dim);
        indices[i] = i;
    }


    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (sims[indices[j]] > sims[indices[best]]) best = j;
        int tmp = indices[i]; indices[i] = indices[best]; indices[best] = tmp;
    }

    for (int i = 0; i < k; i++) {
        out_indices[i] = indices[i];
        out_scores[i]  = sims[indices[i]];
    }

    free(sims); free(indices);
    return k;
}

int mi_rag_augment(const MiRAGConfig *cfg, const float *query,
                   float *K, float *V, int *seq_len,
                   int kv_dim, int max_seq_len) {
    if (!cfg->store || cfg->store->size == 0) return 0;

    int *indices = (int *)malloc(cfg->n_retrieve * sizeof(int));
    float *scores = (float *)malloc(cfg->n_retrieve * sizeof(float));
    MI_CHECK_OOM(indices); MI_CHECK_OOM(scores);

    int found = mi_vstore_search(cfg->store, query,
                                 cfg->n_retrieve, indices, scores);


    int inject = 0;
    for (int i = 0; i < found; i++)
        if (scores[i] >= cfg->gate_threshold)
            inject++;

    if (inject == 0 || *seq_len + inject > max_seq_len) {
        free(indices); free(scores);
        return 0;
    }


    memmove(K + (size_t)inject * kv_dim, K,
            (size_t)(*seq_len) * kv_dim * sizeof(float));
    memmove(V + (size_t)inject * kv_dim, V,
            (size_t)(*seq_len) * kv_dim * sizeof(float));


    int written = 0;
    for (int i = 0; i < found && written < inject; i++) {
        if (scores[i] >= cfg->gate_threshold) {
            const float *emb = cfg->store->embeddings
                             + (size_t)indices[i] * cfg->store->dim;

            int copy_dim = MI_MIN(cfg->store->dim, kv_dim);
            memcpy(K + (size_t)written * kv_dim, emb, copy_dim * sizeof(float));
            memcpy(V + (size_t)written * kv_dim, emb, copy_dim * sizeof(float));

            if (kv_dim > copy_dim) {
                memset(K + (size_t)written * kv_dim + copy_dim, 0,
                       (kv_dim - copy_dim) * sizeof(float));
                memset(V + (size_t)written * kv_dim + copy_dim, 0,
                       (kv_dim - copy_dim) * sizeof(float));
            }
            written++;
        }
    }

    *seq_len += written;
    free(indices); free(scores);
    return written;
}
