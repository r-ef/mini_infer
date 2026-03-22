/* cache.c — four KV-cache layouts: dense, paged, sliding, compressed */
#include "mi/cache.h"
#include "mi/ops.h"

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  1. DENSE CACHE — contiguous pre-allocated arrays (baseline)    ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float *K;          /* [n_layers * max_seq * kv_dim] */
    float *V;
    int    max_seq;
    int    size;       /* tokens cached (same for all layers) */
    int    n_layers;
    int    kv_dim;
} DenseCtx;

static float *dense_kv(float *base, int layer, int max_seq, int kv_dim) {
    return base + (size_t)layer * max_seq * kv_dim;
}

static void dense_append(MiCache *c, int layer, const float *k, const float *v) {
    DenseCtx *d = (DenseCtx *)c->ctx;
    /* Increment on first layer so all layers see the current position
     * in get_keys/get_values (needed for self-attention). */
    if (layer == 0) {
        MI_ASSERT(d->size < d->max_seq, "dense cache full (%d)", d->max_seq);
        d->size++;
    }
    int pos = d->size - 1;
    int off = pos * d->kv_dim;
    memcpy(dense_kv(d->K, layer, d->max_seq, d->kv_dim) + off, k, d->kv_dim * sizeof(float));
    memcpy(dense_kv(d->V, layer, d->max_seq, d->kv_dim) + off, v, d->kv_dim * sizeof(float));
}

static const float *dense_keys(MiCache *c, int layer, int *len) {
    DenseCtx *d = (DenseCtx *)c->ctx;
    *len = d->size;
    return dense_kv(d->K, layer, d->max_seq, d->kv_dim);
}
static const float *dense_values(MiCache *c, int layer, int *len) {
    DenseCtx *d = (DenseCtx *)c->ctx;
    *len = d->size;
    return dense_kv(d->V, layer, d->max_seq, d->kv_dim);
}
static int dense_size(MiCache *c) { return ((DenseCtx *)c->ctx)->size; }
static void dense_truncate(MiCache *c, int n) {
    DenseCtx *d = (DenseCtx *)c->ctx;
    if (n < d->size) d->size = n;
}
static void dense_clear(MiCache *c) { ((DenseCtx *)c->ctx)->size = 0; }
static void dense_destroy(MiCache *c) {
    DenseCtx *d = (DenseCtx *)c->ctx;
    free(d->K); free(d->V); free(d);
}
static void dense_stats(MiCache *c, char *buf, int len) {
    DenseCtx *d = (DenseCtx *)c->ctx;
    snprintf(buf, len, "dense: %d/%d slots, %.1f KB",
             d->size, d->max_seq,
             2.0f * d->n_layers * d->max_seq * d->kv_dim * 4.0f / 1024.0f);
}

static const MiCacheVT dense_vt = {
    .name       = "dense",
    .append     = dense_append,
    .get_keys   = dense_keys,
    .get_values = dense_values,
    .size       = dense_size,
    .truncate   = dense_truncate,
    .clear      = dense_clear,
    .destroy    = dense_destroy,
    .stats      = dense_stats,
};

MiCache mi_cache_dense(int n_layers, int n_kv_heads, int d_head, int max_seq) {
    int kv_dim = n_kv_heads * d_head;
    DenseCtx *d = (DenseCtx *)calloc(1, sizeof(DenseCtx));
    MI_CHECK_OOM(d);
    size_t total = (size_t)n_layers * max_seq * kv_dim * sizeof(float);
    d->K = (float *)calloc(1, total); MI_CHECK_OOM(d->K);
    d->V = (float *)calloc(1, total); MI_CHECK_OOM(d->V);
    d->max_seq  = max_seq;
    d->n_layers = n_layers;
    d->kv_dim   = kv_dim;
    d->size     = 0;
    return (MiCache){
        .vt = &dense_vt, .ctx = d,
        .n_layers = n_layers, .n_kv_heads = n_kv_heads,
        .d_head = d_head, .kv_dim = kv_dim };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  2. PAGED CACHE — block-allocated pages (vLLM-style)            ║
 * ║                                                                   ║
 * ║  Physical pages are pooled.  Each layer maintains a page table   ║
 * ║  mapping logical block index → physical page.  Pages hold        ║
 * ║  page_size entries of [kv_dim] floats.                            ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    /* Page pool: each page = page_size * kv_dim floats */
    float *k_pool;       /* [max_pages * page_size * kv_dim] */
    float *v_pool;
    int    page_size;
    int    max_pages;
    int    kv_dim;
    int    n_layers;

    /* Free list (stack) */
    int   *free_stack;
    int    n_free;

    /* Per-layer page tables: page_table[layer][block_idx] = phys page id */
    int  **page_table;
    int   *n_blocks;       /* [n_layers] blocks allocated per layer */

    int    size;           /* current token count */

    /* Staging: used by get_keys/get_values to return contiguous data */
    float *staging;
    int    staging_cap;    /* max tokens for staging */
} PagedCtx;

static float *paged_page(float *pool, int page_id, int page_size, int kv_dim) {
    return pool + (size_t)page_id * page_size * kv_dim;
}

static int paged_alloc_page(PagedCtx *p) {
    MI_ASSERT(p->n_free > 0, "paged cache: no free pages");
    return p->free_stack[--p->n_free];
}

static void paged_free_page(PagedCtx *p, int page_id) {
    p->free_stack[p->n_free++] = page_id;
}

static void paged_append(MiCache *c, int layer, const float *k, const float *v) {
    PagedCtx *p = (PagedCtx *)c->ctx;
    if (layer == 0) p->size++;
    int pos = p->size - 1;
    int pos_in_block = pos % p->page_size;

    /* Need a new page for this layer? */
    if (pos_in_block == 0) {
        int pid = paged_alloc_page(p);
        p->page_table[layer][p->n_blocks[layer]++] = pid;
    }

    int block_idx = p->n_blocks[layer] - 1;
    int pid = p->page_table[layer][block_idx];
    float *kp = paged_page(p->k_pool, pid, p->page_size, p->kv_dim)
                + pos_in_block * p->kv_dim;
    float *vp = paged_page(p->v_pool, pid, p->page_size, p->kv_dim)
                + pos_in_block * p->kv_dim;
    memcpy(kp, k, p->kv_dim * sizeof(float));
    memcpy(vp, v, p->kv_dim * sizeof(float));
}

/* Gather pages into contiguous staging buffer */
static const float *paged_gather(PagedCtx *p, float *pool,
                                  int layer, int *seq_len) {
    *seq_len = p->size;
    if (p->size == 0) return p->staging;

    /* Ensure staging is large enough */
    int needed = p->size * p->kv_dim;
    if (needed > p->staging_cap * p->kv_dim) {
        p->staging = (float *)realloc(p->staging, (size_t)needed * sizeof(float));
        MI_CHECK_OOM(p->staging);
        p->staging_cap = p->size;
    }

    int copied = 0;
    for (int b = 0; b < p->n_blocks[layer]; b++) {
        int pid = p->page_table[layer][b];
        float *src = paged_page(pool, pid, p->page_size, p->kv_dim);
        int count = MI_MIN(p->page_size, p->size - copied);
        memcpy(p->staging + (size_t)copied * p->kv_dim, src,
               (size_t)count * p->kv_dim * sizeof(float));
        copied += count;
    }
    return p->staging;
}

static const float *paged_keys(MiCache *c, int layer, int *len) {
    PagedCtx *p = (PagedCtx *)c->ctx;
    return paged_gather(p, p->k_pool, layer, len);
}
static const float *paged_values(MiCache *c, int layer, int *len) {
    PagedCtx *p = (PagedCtx *)c->ctx;
    return paged_gather(p, p->v_pool, layer, len);
}

static int paged_size(MiCache *c) { return ((PagedCtx *)c->ctx)->size; }

static void paged_truncate(MiCache *c, int n) {
    PagedCtx *p = (PagedCtx *)c->ctx;
    if (n >= p->size) return;

    /* Free pages that are entirely beyond position n */
    int keep_blocks = (n + p->page_size - 1) / p->page_size;
    for (int l = 0; l < p->n_layers; l++) {
        for (int b = keep_blocks; b < p->n_blocks[l]; b++)
            paged_free_page(p, p->page_table[l][b]);
        p->n_blocks[l] = keep_blocks;
    }
    p->size = n;
}

static void paged_clear(MiCache *c) { paged_truncate(c, 0); }

static void paged_destroy(MiCache *c) {
    PagedCtx *p = (PagedCtx *)c->ctx;
    free(p->k_pool); free(p->v_pool);
    for (int l = 0; l < p->n_layers; l++) free(p->page_table[l]);
    free(p->page_table); free(p->n_blocks);
    free(p->free_stack); free(p->staging); free(p);
}

static void paged_stats(MiCache *c, char *buf, int len) {
    PagedCtx *p = (PagedCtx *)c->ctx;
    int used = p->max_pages - p->n_free;
    snprintf(buf, len, "paged: %d tokens, %d/%d pages used (page_size=%d)",
             p->size, used, p->max_pages, p->page_size);
}

static const MiCacheVT paged_vt = {
    .name       = "paged",
    .append     = paged_append,
    .get_keys   = paged_keys,
    .get_values = paged_values,
    .size       = paged_size,
    .truncate   = paged_truncate,
    .clear      = paged_clear,
    .destroy    = paged_destroy,
    .stats      = paged_stats,
};

MiCache mi_cache_paged(int n_layers, int n_kv_heads, int d_head,
                       int page_size, int max_pages) {
    int kv_dim = n_kv_heads * d_head;
    PagedCtx *p = (PagedCtx *)calloc(1, sizeof(PagedCtx));
    MI_CHECK_OOM(p);
    p->page_size = page_size;
    p->max_pages = max_pages;
    p->kv_dim    = kv_dim;
    p->n_layers  = n_layers;

    size_t page_bytes = (size_t)max_pages * page_size * kv_dim * sizeof(float);
    p->k_pool = (float *)calloc(1, page_bytes); MI_CHECK_OOM(p->k_pool);
    p->v_pool = (float *)calloc(1, page_bytes); MI_CHECK_OOM(p->v_pool);

    p->free_stack = (int *)malloc(max_pages * sizeof(int));
    MI_CHECK_OOM(p->free_stack);
    p->n_free = max_pages;
    for (int i = 0; i < max_pages; i++) p->free_stack[i] = i;

    int max_blocks_per_layer = max_pages;  /* conservative upper bound */
    p->page_table = (int **)calloc(n_layers, sizeof(int *));
    p->n_blocks   = (int *)calloc(n_layers, sizeof(int));
    MI_CHECK_OOM(p->page_table); MI_CHECK_OOM(p->n_blocks);
    for (int l = 0; l < n_layers; l++) {
        p->page_table[l] = (int *)malloc(max_blocks_per_layer * sizeof(int));
        MI_CHECK_OOM(p->page_table[l]);
    }

    p->staging     = NULL;
    p->staging_cap = 0;
    p->size = 0;

    return (MiCache){
        .vt = &paged_vt, .ctx = p,
        .n_layers = n_layers, .n_kv_heads = n_kv_heads,
        .d_head = d_head, .kv_dim = kv_dim };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  3. SLIDING WINDOW — ring buffer (Mistral-style)                ║
 * ║                                                                   ║
 * ║  Attention only sees the last `window` tokens.  Memory is O(W)  ║
 * ║  regardless of total sequence length.                             ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float *K;           /* [n_layers * window * kv_dim] */
    float *V;
    int    window;
    int    kv_dim;
    int    n_layers;
    int    total_written; /* monotonically increasing */

    /* Staging for linearised get (ring → contiguous) */
    float *staging;
} SlidingCtx;

static void sliding_append(MiCache *c, int layer, const float *k, const float *v) {
    SlidingCtx *s = (SlidingCtx *)c->ctx;
    if (layer == 0) s->total_written++;
    int pos = s->total_written - 1;
    int slot = pos % s->window;
    size_t base = (size_t)layer * s->window * s->kv_dim + slot * s->kv_dim;
    memcpy(s->K + base, k, s->kv_dim * sizeof(float));
    memcpy(s->V + base, v, s->kv_dim * sizeof(float));
}

/* Linearise ring buffer into staging for the given layer and pool */
static const float *sliding_linearise(SlidingCtx *s, float *pool,
                                       int layer, int *seq_len) {
    int len = MI_MIN(s->total_written, s->window);
    *seq_len = len;
    if (len == 0) return s->staging;

    float *base = pool + (size_t)layer * s->window * s->kv_dim;

    if (s->total_written <= s->window) {
        /* Haven't wrapped yet — already contiguous */
        return base;
    }

    /* Wrapped: oldest entry is at slot (total_written % window).
     * Copy [oldest..window) then [0..oldest) */
    int oldest = s->total_written % s->window;
    int part1 = s->window - oldest;
    memcpy(s->staging,
           base + (size_t)oldest * s->kv_dim,
           (size_t)part1 * s->kv_dim * sizeof(float));
    memcpy(s->staging + (size_t)part1 * s->kv_dim,
           base,
           (size_t)oldest * s->kv_dim * sizeof(float));
    return s->staging;
}

static const float *sliding_keys(MiCache *c, int layer, int *len) {
    return sliding_linearise((SlidingCtx *)c->ctx,
                              ((SlidingCtx *)c->ctx)->K, layer, len);
}
static const float *sliding_values(MiCache *c, int layer, int *len) {
    return sliding_linearise((SlidingCtx *)c->ctx,
                              ((SlidingCtx *)c->ctx)->V, layer, len);
}
static int sliding_size(MiCache *c) {
    SlidingCtx *s = (SlidingCtx *)c->ctx;
    return MI_MIN(s->total_written, s->window);
}
static void sliding_truncate(MiCache *c, int n) {
    SlidingCtx *s = (SlidingCtx *)c->ctx;
    if (n < s->total_written) s->total_written = n;
}
static void sliding_clear(MiCache *c) {
    ((SlidingCtx *)c->ctx)->total_written = 0;
}
static void sliding_destroy(MiCache *c) {
    SlidingCtx *s = (SlidingCtx *)c->ctx;
    free(s->K); free(s->V); free(s->staging); free(s);
}
static void sliding_stats(MiCache *c, char *buf, int len) {
    SlidingCtx *s = (SlidingCtx *)c->ctx;
    snprintf(buf, len, "sliding: window=%d, total_written=%d, active=%d",
             s->window, s->total_written,
             MI_MIN(s->total_written, s->window));
}

static const MiCacheVT sliding_vt = {
    .name       = "sliding",
    .append     = sliding_append,
    .get_keys   = sliding_keys,
    .get_values = sliding_values,
    .size       = sliding_size,
    .truncate   = sliding_truncate,
    .clear      = sliding_clear,
    .destroy    = sliding_destroy,
    .stats      = sliding_stats,
};

MiCache mi_cache_sliding(int n_layers, int n_kv_heads, int d_head,
                         int window_size) {
    int kv_dim = n_kv_heads * d_head;
    SlidingCtx *s = (SlidingCtx *)calloc(1, sizeof(SlidingCtx));
    MI_CHECK_OOM(s);
    s->window   = window_size;
    s->kv_dim   = kv_dim;
    s->n_layers = n_layers;
    size_t total = (size_t)n_layers * window_size * kv_dim * sizeof(float);
    s->K = (float *)calloc(1, total); MI_CHECK_OOM(s->K);
    s->V = (float *)calloc(1, total); MI_CHECK_OOM(s->V);
    s->staging = (float *)malloc((size_t)window_size * kv_dim * sizeof(float));
    MI_CHECK_OOM(s->staging);
    return (MiCache){
        .vt = &sliding_vt, .ctx = s,
        .n_layers = n_layers, .n_kv_heads = n_kv_heads,
        .d_head = d_head, .kv_dim = kv_dim };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  4. COMPRESSED CACHE — int8 for old entries, fp32 for recent    ║
 * ║                                                                   ║
 * ║  Entries 0 .. size-fresh_count-1 are stored as int8 + scale.    ║
 * ║  Entries size-fresh_count .. size-1 remain fp32.                 ║
 * ║  On get_keys/values, int8 entries are dequantised to staging.   ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    /* Fresh (fp32) storage — [n_layers * max_seq * kv_dim] */
    float  *K_f32;
    float  *V_f32;

    /* Compressed (int8) storage */
    int8_t *K_i8;          /* [n_layers * max_seq * kv_dim] */
    int8_t *V_i8;
    float  *K_scales;      /* [n_layers * max_seq] one scale per position */
    float  *V_scales;

    int     max_seq;
    int     kv_dim;
    int     n_layers;
    int     fresh_count;   /* how many recent entries stay fp32 */
    int     size;
    int     compressed_up_to; /* entries [0, compressed_up_to) are in int8 */

    /* Staging for dequantised + fresh concatenation */
    float  *staging;
} CompressedCtx;

/* Quantise a single kv vector to int8 absmax */
static void compress_vec(const float *src, int8_t *dst, float *scale, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    *scale = amax / 127.0f;
    if (amax < 1e-10f) {
        memset(dst, 0, n);
        return;
    }
    float inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        int v = (int)roundf(src[i] * inv);
        dst[i] = (int8_t)MI_CLAMP(v, -127, 127);
    }
}

static void decompress_vec(const int8_t *src, float scale, float *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = (float)src[i] * scale;
}

static void compressed_maybe_compress(CompressedCtx *cc) {
    /* Compress entries that are no longer "fresh" */
    int target = cc->size - cc->fresh_count;
    if (target <= cc->compressed_up_to) return;

    for (int t = cc->compressed_up_to; t < target; t++) {
        for (int l = 0; l < cc->n_layers; l++) {
            size_t off_vec   = (size_t)l * cc->max_seq * cc->kv_dim + t * cc->kv_dim;
            size_t off_scale = (size_t)l * cc->max_seq + t;
            compress_vec(cc->K_f32 + off_vec, cc->K_i8 + off_vec,
                         cc->K_scales + off_scale, cc->kv_dim);
            compress_vec(cc->V_f32 + off_vec, cc->V_i8 + off_vec,
                         cc->V_scales + off_scale, cc->kv_dim);
        }
    }
    cc->compressed_up_to = target;
}

static void compressed_append(MiCache *c, int layer, const float *k, const float *v) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    if (layer == 0) {
        MI_ASSERT(cc->size < cc->max_seq, "compressed cache full");
        cc->size++;
    }
    int pos = cc->size - 1;
    size_t off = (size_t)layer * cc->max_seq * cc->kv_dim + pos * cc->kv_dim;
    memcpy(cc->K_f32 + off, k, cc->kv_dim * sizeof(float));
    memcpy(cc->V_f32 + off, v, cc->kv_dim * sizeof(float));
    if (layer == cc->n_layers - 1)
        compressed_maybe_compress(cc);
}

static const float *compressed_get(CompressedCtx *cc, float *f32, int8_t *i8,
                                    float *scales, int layer, int *seq_len) {
    *seq_len = cc->size;
    if (cc->size == 0) return cc->staging;

    float *out = cc->staging;

    /* Dequantise compressed portion */
    for (int t = 0; t < cc->compressed_up_to; t++) {
        size_t off_vec   = (size_t)layer * cc->max_seq * cc->kv_dim + t * cc->kv_dim;
        size_t off_scale = (size_t)layer * cc->max_seq + t;
        decompress_vec(i8 + off_vec, scales[off_scale],
                       out + t * cc->kv_dim, cc->kv_dim);
    }
    /* Copy fresh portion */
    int fresh_start = MI_MAX(0, cc->compressed_up_to);
    if (fresh_start < cc->size) {
        size_t off = (size_t)layer * cc->max_seq * cc->kv_dim + fresh_start * cc->kv_dim;
        int n = cc->size - fresh_start;
        memcpy(out + (size_t)fresh_start * cc->kv_dim, f32 + off,
               (size_t)n * cc->kv_dim * sizeof(float));
    }
    return out;
}

static const float *compressed_keys(MiCache *c, int layer, int *len) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    return compressed_get(cc, cc->K_f32, cc->K_i8, cc->K_scales, layer, len);
}
static const float *compressed_values(MiCache *c, int layer, int *len) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    return compressed_get(cc, cc->V_f32, cc->V_i8, cc->V_scales, layer, len);
}
static int compressed_size(MiCache *c) { return ((CompressedCtx *)c->ctx)->size; }
static void compressed_truncate(MiCache *c, int n) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    if (n < cc->size) {
        cc->size = n;
        if (n < cc->compressed_up_to) cc->compressed_up_to = n;
    }
}
static void compressed_clear(MiCache *c) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    cc->size = 0; cc->compressed_up_to = 0;
}
static void compressed_destroy(MiCache *c) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    free(cc->K_f32); free(cc->V_f32);
    free(cc->K_i8);  free(cc->V_i8);
    free(cc->K_scales); free(cc->V_scales);
    free(cc->staging); free(cc);
}
static void compressed_stats(MiCache *c, char *buf, int len) {
    CompressedCtx *cc = (CompressedCtx *)c->ctx;
    float fp32_kb = (float)(cc->size - cc->compressed_up_to) * cc->kv_dim * 4.0f / 1024.0f;
    float i8_kb   = (float)cc->compressed_up_to * cc->kv_dim * 1.0f / 1024.0f;
    snprintf(buf, len, "compressed: %d entries (%d int8 + %d fp32), "
             "%.1f KB i8 + %.1f KB f32 per layer",
             cc->size, cc->compressed_up_to,
             cc->size - cc->compressed_up_to,
             i8_kb, fp32_kb);
}

static const MiCacheVT compressed_vt = {
    .name       = "compressed",
    .append     = compressed_append,
    .get_keys   = compressed_keys,
    .get_values = compressed_values,
    .size       = compressed_size,
    .truncate   = compressed_truncate,
    .clear      = compressed_clear,
    .destroy    = compressed_destroy,
    .stats      = compressed_stats,
};

MiCache mi_cache_compressed(int n_layers, int n_kv_heads, int d_head,
                            int max_seq, int fresh_count) {
    int kv_dim = n_kv_heads * d_head;
    CompressedCtx *cc = (CompressedCtx *)calloc(1, sizeof(CompressedCtx));
    MI_CHECK_OOM(cc);
    cc->max_seq     = max_seq;
    cc->kv_dim      = kv_dim;
    cc->n_layers    = n_layers;
    cc->fresh_count = fresh_count;

    size_t f32_total = (size_t)n_layers * max_seq * kv_dim * sizeof(float);
    size_t i8_total  = (size_t)n_layers * max_seq * kv_dim;
    size_t sc_total  = (size_t)n_layers * max_seq * sizeof(float);

    cc->K_f32    = (float *)calloc(1, f32_total); MI_CHECK_OOM(cc->K_f32);
    cc->V_f32    = (float *)calloc(1, f32_total); MI_CHECK_OOM(cc->V_f32);
    cc->K_i8     = (int8_t *)calloc(1, i8_total);  MI_CHECK_OOM(cc->K_i8);
    cc->V_i8     = (int8_t *)calloc(1, i8_total);  MI_CHECK_OOM(cc->V_i8);
    cc->K_scales = (float *)calloc(1, sc_total);  MI_CHECK_OOM(cc->K_scales);
    cc->V_scales = (float *)calloc(1, sc_total);  MI_CHECK_OOM(cc->V_scales);

    cc->staging = (float *)malloc((size_t)max_seq * kv_dim * sizeof(float));
    MI_CHECK_OOM(cc->staging);

    return (MiCache){
        .vt = &compressed_vt, .ctx = cc,
        .n_layers = n_layers, .n_kv_heads = n_kv_heads,
        .d_head = d_head, .kv_dim = kv_dim };
}
