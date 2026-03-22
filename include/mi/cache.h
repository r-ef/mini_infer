/* ═══════════════════════════════════════════════════════════════════
 * cache.h — pluggable KV-cache layouts
 *
 * All implementations store K/V for every layer internally.
 * Vectors are laid out as [n_kv_heads * d_head] per position.
 *
 * Implementations:
 *   Dense      — contiguous pre-allocated arrays (baseline)
 *   Paged      — block-allocated pages (vLLM-style)
 *   Sliding    — ring buffer, last N tokens (Mistral-style)
 *   Compressed — quantise entries older than a threshold to int8
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_CACHE_H
#define MI_CACHE_H

#include "tensor.h"

typedef struct MiCache MiCache;

typedef struct {
    const char *name;

    /* Append k,v of shape [kv_dim] for one layer at the current tail. */
    void (*append)(MiCache *c, int layer, const float *k, const float *v);

    /* Return contiguous K data for a layer and write *seq_len.
     * Pointer valid until the next mutating call on this cache.
     * Layout: [seq_len, kv_dim] row-major. */
    const float *(*get_keys)(MiCache *c, int layer, int *seq_len);
    const float *(*get_values)(MiCache *c, int layer, int *seq_len);

    /* How many positions are stored. */
    int (*size)(MiCache *c);

    /* Truncate to `new_size` positions (for speculative rollback). */
    void (*truncate)(MiCache *c, int new_size);

    /* Clear everything. */
    void (*clear)(MiCache *c);

    /* Release resources. */
    void (*destroy)(MiCache *c);

    /* Print stats into buf (optional). */
    void (*stats)(MiCache *c, char *buf, int buf_len);

} MiCacheVT;

struct MiCache {
    const MiCacheVT *vt;
    void *ctx;
    int n_layers;
    int n_kv_heads;
    int d_head;
    int kv_dim;            /* = n_kv_heads * d_head, cached for convenience */
};

/* ── Generic dispatch (inline) ── */
static inline void mi_cache_append(MiCache *c, int layer,
                                    const float *k, const float *v) {
    c->vt->append(c, layer, k, v);
}
static inline const float *mi_cache_keys(MiCache *c, int layer, int *len) {
    return c->vt->get_keys(c, layer, len);
}
static inline const float *mi_cache_values(MiCache *c, int layer, int *len) {
    return c->vt->get_values(c, layer, len);
}
static inline int  mi_cache_size(MiCache *c)      { return c->vt->size(c);    }
static inline void mi_cache_truncate(MiCache *c, int n) { c->vt->truncate(c,n); }
static inline void mi_cache_clear(MiCache *c)     { c->vt->clear(c);   }
static inline void mi_cache_destroy(MiCache *c)   { c->vt->destroy(c); }

/* ── Constructors ── */

/* Dense: pre-allocated [max_seq_len, kv_dim] per layer */
MiCache mi_cache_dense(int n_layers, int n_kv_heads, int d_head,
                       int max_seq_len);

/* Paged: blocks of page_size entries, allocated on demand */
MiCache mi_cache_paged(int n_layers, int n_kv_heads, int d_head,
                       int page_size, int max_pages);

/* Sliding window: ring buffer of the last window_size tokens */
MiCache mi_cache_sliding(int n_layers, int n_kv_heads, int d_head,
                         int window_size);

/* Compressed: entries older than fresh_count are quantised to int8 */
MiCache mi_cache_compressed(int n_layers, int n_kv_heads, int d_head,
                            int max_seq_len, int fresh_count);

#endif /* MI_CACHE_H */
