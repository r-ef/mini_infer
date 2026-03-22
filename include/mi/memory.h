
#ifndef MI_MEMORY_H
#define MI_MEMORY_H

#include "tensor.h"
#include "cache.h"

typedef struct {
    int sink_size;
    int window_size;
} MiSinkConfig;

void mi_sink_evict(float *K, float *V, int *seq_len,
                   int kv_dim, const MiSinkConfig *cfg);

typedef struct {
    float *cum_attn;
    int    max_seq;
    int    budget;
    int    n_current;
} MiH2O;

MiH2O mi_h2o_create(int max_seq, int budget);
void  mi_h2o_free(MiH2O *h);

void mi_h2o_update(MiH2O *h, const float *weights, int seq_len);

int  mi_h2o_select(const MiH2O *h, int seq_len,
                   int *keep_indices);

void mi_h2o_compact(float *K, float *V, int *seq_len, int kv_dim,
                    const int *keep_indices, int n_keep);

typedef struct {
    float threshold;
    int   min_sep;
} MiMergeConfig;

int mi_token_merge(float *K, float *V, int seq_len, int kv_dim,
                   const MiMergeConfig *cfg);

typedef struct {
    float *embeddings;
    int   *doc_ids;
    int    dim;
    int    size;
    int    capacity;
} MiVectorStore;

MiVectorStore mi_vstore_create(int dim, int capacity);
void          mi_vstore_free(MiVectorStore *vs);

void mi_vstore_add(MiVectorStore *vs, const float *emb, int doc_id);

int mi_vstore_search(const MiVectorStore *vs, const float *query,
                     int k, int *out_indices, float *out_scores);

typedef struct {
    MiVectorStore *store;
    int   n_retrieve;
    float gate_threshold;
} MiRAGConfig;

int mi_rag_augment(const MiRAGConfig *cfg, const float *query,
                   float *K, float *V, int *seq_len,
                   int kv_dim, int max_seq_len);

#endif
