/* ═══════════════════════════════════════════════════════════════════
 * memory.h — KV memory management & retrieval-augmented memory
 *
 * Compression / eviction
 *   Attention Sink  — StreamingLLM: keep first K + last N tokens
 *   H2O             — Heavy-Hitter Oracle: evict least-attended
 *   Token Merge     — merge similar adjacent KV entries
 *
 * Retrieval
 *   VectorStore     — brute-force cosine-similarity store
 *   RAG Augment     — inject retrieved vectors into KV sequence
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_MEMORY_H
#define MI_MEMORY_H

#include "tensor.h"
#include "cache.h"

/* ════════════ Attention Sink (StreamingLLM) ════════════ */

typedef struct {
    int sink_size;      /* initial "sink" tokens always kept */
    int window_size;    /* recent tokens always kept */
} MiSinkConfig;

/* Compact K, V arrays in-place.
 * *seq_len is updated to the new (shorter) length.
 * K, V: [*seq_len, kv_dim]  →  [new_len, kv_dim]  */
void mi_sink_evict(float *K, float *V, int *seq_len,
                   int kv_dim, const MiSinkConfig *cfg);

/* ════════════ H2O — Heavy Hitter Oracle ════════════ */

typedef struct {
    float *cum_attn;      /* [max_seq] cumulative attention received */
    int    max_seq;
    int    budget;        /* max KV entries to keep */
    int    n_current;
} MiH2O;

MiH2O mi_h2o_create(int max_seq, int budget);
void  mi_h2o_free(MiH2O *h);

/* Accumulate attention weights from the latest decode step.
 * weights: [seq_len] (softmax output for one head; caller may
 * average across heads first). */
void mi_h2o_update(MiH2O *h, const float *weights, int seq_len);

/* Select which indices to keep (writes up to budget indices).
 * Returns n_keep (≤ budget). */
int  mi_h2o_select(const MiH2O *h, int seq_len,
                   int *keep_indices);

/* Compact K, V using keep_indices (from mi_h2o_select). */
void mi_h2o_compact(float *K, float *V, int *seq_len, int kv_dim,
                    const int *keep_indices, int n_keep);

/* ════════════ Token Merge ════════════ */

typedef struct {
    float threshold;   /* cosine similarity threshold for merging */
    int   min_sep;     /* minimum gap between merge candidates */
} MiMergeConfig;

/* Merge similar adjacent entries in K, V.  Returns new seq_len. */
int mi_token_merge(float *K, float *V, int seq_len, int kv_dim,
                   const MiMergeConfig *cfg);

/* ════════════ Vector Store ════════════ */

typedef struct {
    float *embeddings;    /* [capacity * dim] flat */
    int   *doc_ids;
    int    dim;
    int    size;
    int    capacity;
} MiVectorStore;

MiVectorStore mi_vstore_create(int dim, int capacity);
void          mi_vstore_free(MiVectorStore *vs);

void mi_vstore_add(MiVectorStore *vs, const float *emb, int doc_id);

/* k-nearest-neighbour search.  Returns count found.
 * out_indices, out_scores must hold k elements. */
int mi_vstore_search(const MiVectorStore *vs, const float *query,
                     int k, int *out_indices, float *out_scores);

/* ════════════ RAG Memory Augmentation ════════════ */

typedef struct {
    MiVectorStore *store;
    int   n_retrieve;        /* how many neighbours to inject */
    float gate_threshold;    /* min cosine-sim to include */
} MiRAGConfig;

/* Retrieve from store and prepend to K, V.
 * K, V: [*seq_len, kv_dim]  (must have room for n_retrieve extra rows
 *        up to max_seq_len).
 * Returns number of entries injected. */
int mi_rag_augment(const MiRAGConfig *cfg, const float *query,
                   float *K, float *V, int *seq_len,
                   int kv_dim, int max_seq_len);

#endif /* MI_MEMORY_H */
