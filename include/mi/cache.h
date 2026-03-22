
#ifndef MI_CACHE_H
#define MI_CACHE_H

#include "tensor.h"

typedef struct MiCache MiCache;

typedef struct {
    const char *name;


    void (*append)(MiCache *c, int layer, const float *k, const float *v);


    const float *(*get_keys)(MiCache *c, int layer, int *seq_len);
    const float *(*get_values)(MiCache *c, int layer, int *seq_len);


    int (*size)(MiCache *c);


    void (*truncate)(MiCache *c, int new_size);


    void (*clear)(MiCache *c);


    void (*destroy)(MiCache *c);


    void (*stats)(MiCache *c, char *buf, int buf_len);

} MiCacheVT;

struct MiCache {
    const MiCacheVT *vt;
    void *ctx;
    int n_layers;
    int n_kv_heads;
    int d_head;
    int kv_dim;
};

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

MiCache mi_cache_dense(int n_layers, int n_kv_heads, int d_head,
                       int max_seq_len);

MiCache mi_cache_paged(int n_layers, int n_kv_heads, int d_head,
                       int page_size, int max_pages);

MiCache mi_cache_sliding(int n_layers, int n_kv_heads, int d_head,
                         int window_size);

MiCache mi_cache_compressed(int n_layers, int n_kv_heads, int d_head,
                            int max_seq_len, int fresh_count);

#endif
