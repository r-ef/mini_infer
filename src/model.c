
#include "mi/model.h"
#include "mi/ops.h"

static void alloc_layer(MiLayerWeights *lw, const MiModelConfig *cfg) {
    int q_dim  = cfg->n_heads    * cfg->d_head;
    int kv_dim = cfg->n_kv_heads * cfg->d_head;
    int d      = cfg->d_model;
    int ff     = cfg->d_ff;

    lw->Wq = mi_tensor_zeros(q_dim,  d);
    lw->Wk = mi_tensor_zeros(kv_dim, d);
    lw->Wv = mi_tensor_zeros(kv_dim, d);
    lw->Wo = mi_tensor_zeros(d,      q_dim);
    lw->W1 = mi_tensor_zeros(ff,     d);
    lw->W2 = mi_tensor_zeros(d,      ff);
    if (cfg->ffn_type == MI_FFN_SWIGLU)
        lw->W3 = mi_tensor_zeros(ff, d);
    else
        lw->W3 = (MiTensor){0};

    lw->rms_att = (float *)calloc(d, sizeof(float));
    lw->rms_ffn = (float *)calloc(d, sizeof(float));
    MI_CHECK_OOM(lw->rms_att); MI_CHECK_OOM(lw->rms_ffn);
    for (int i = 0; i < d; i++) { lw->rms_att[i] = 1.0f; lw->rms_ffn[i] = 1.0f; }
}

static void free_layer(MiLayerWeights *lw) {
    mi_tensor_free(&lw->Wq); mi_tensor_free(&lw->Wk);
    mi_tensor_free(&lw->Wv); mi_tensor_free(&lw->Wo);
    mi_tensor_free(&lw->W1); mi_tensor_free(&lw->W2);
    mi_tensor_free(&lw->W3);
    free(lw->rms_att); free(lw->rms_ffn);
    lw->rms_att = NULL; lw->rms_ffn = NULL;
}

MiModel mi_model_create(MiModelConfig cfg) {
    MiModel m;
    memset(&m, 0, sizeof(m));
    m.cfg = cfg;


    m.w.tok_emb  = mi_tensor_zeros(cfg.vocab_size, cfg.d_model);
    m.w.out_proj = mi_tensor_zeros(cfg.vocab_size, cfg.d_model);
    m.w.rms_final = (float *)calloc(cfg.d_model, sizeof(float));
    MI_CHECK_OOM(m.w.rms_final);
    for (int i = 0; i < cfg.d_model; i++) m.w.rms_final[i] = 1.0f;

    m.w.layers = (MiLayerWeights *)calloc(cfg.n_layers, sizeof(MiLayerWeights));
    MI_CHECK_OOM(m.w.layers);
    for (int l = 0; l < cfg.n_layers; l++)
        alloc_layer(&m.w.layers[l], &cfg);


    m.cache = mi_cache_dense(cfg.n_layers, cfg.n_kv_heads, cfg.d_head,
                             cfg.max_seq_len);
    m.attn  = mi_attention_standard();
    m.rope  = mi_rope_standard(cfg.rope_theta > 0 ? cfg.rope_theta : 10000.0f);
    m.pos   = 0;
    return m;
}

void mi_model_init_random(MiModel *m, MiRng *rng) {
    MiModelConfig *c = &m->cfg;
    float std = 0.02f;

    mi_tensor_rand_normal(&m->w.tok_emb, rng, 0.0f, std);
    mi_tensor_rand_normal(&m->w.out_proj, rng, 0.0f, std);

    for (int l = 0; l < c->n_layers; l++) {
        MiLayerWeights *lw = &m->w.layers[l];
        mi_tensor_rand_normal(&lw->Wq, rng, 0.0f, std);
        mi_tensor_rand_normal(&lw->Wk, rng, 0.0f, std);
        mi_tensor_rand_normal(&lw->Wv, rng, 0.0f, std);
        mi_tensor_rand_normal(&lw->Wo, rng, 0.0f, std);
        mi_tensor_rand_normal(&lw->W1, rng, 0.0f, std);
        mi_tensor_rand_normal(&lw->W2, rng, 0.0f, std);
        if (c->ffn_type == MI_FFN_SWIGLU)
            mi_tensor_rand_normal(&lw->W3, rng, 0.0f, std);
    }
}

void mi_model_free(MiModel *m) {
    mi_tensor_free(&m->w.tok_emb);
    mi_tensor_free(&m->w.out_proj);
    free(m->w.rms_final); m->w.rms_final = NULL;
    for (int l = 0; l < m->cfg.n_layers; l++)
        free_layer(&m->w.layers[l]);
    free(m->w.layers); m->w.layers = NULL;
    mi_cache_destroy(&m->cache);
    mi_rope_destroy(&m->rope);
    if (m->attn.vt && m->attn.vt->destroy)
        m->attn.vt->destroy(&m->attn);
}

void mi_model_set_attention(MiModel *m, MiAttention attn) { m->attn = attn; }
void mi_model_set_cache(MiModel *m, MiCache cache) { m->cache = cache; }
void mi_model_set_rope(MiModel *m, MiRoPE rope) { m->rope = rope; }

void mi_model_reset(MiModel *m) {
    mi_cache_clear(&m->cache);
    m->pos = 0;
}

size_t mi_model_scratch_size(const MiModelConfig *cfg) {
    int d      = cfg->d_model;
    int q_dim  = cfg->n_heads * cfg->d_head;
    int kv_dim = cfg->n_kv_heads * cfg->d_head;
    int ff     = cfg->d_ff;


    size_t per_token = (size_t)(d * 3 + q_dim * 2 + kv_dim * 2 + ff * 2);

    per_token += cfg->max_seq_len;

    per_token += 256;
    return per_token * sizeof(float);
}

void mi_model_forward(MiModel *m, int token, float *logits, float *scratch) {
    MiModelConfig *cfg = &m->cfg;
    int d       = cfg->d_model;
    int q_dim   = cfg->n_heads * cfg->d_head;
    int kv_dim  = cfg->n_kv_heads * cfg->d_head;
    int ff      = cfg->d_ff;
    int pos     = m->pos;


    float *ptr = scratch;
    #define SALLOC(n) (ptr += (n), ptr - (n))
    float *x         = SALLOC(d);
    float *x_norm    = SALLOC(d);
    float *q         = SALLOC(q_dim);
    float *k         = SALLOC(kv_dim);
    float *v         = SALLOC(kv_dim);
    float *attn_out  = SALLOC(q_dim);
    float *o_proj    = SALLOC(d);
    float *ffn_out   = SALLOC(d);
    float *ffn_scratch = SALLOC(ff * 2);
    float *attn_scratch = SALLOC(cfg->max_seq_len);
    #undef SALLOC


    mi_vec_copy(mi_tensor_row(&m->w.tok_emb, token), x, d);


    for (int l = 0; l < cfg->n_layers; l++) {
        MiLayerWeights *lw = &m->w.layers[l];


        mi_rmsnorm(x, lw->rms_att, x_norm, d, cfg->norm_eps);


        mi_matvec(&lw->Wq, x_norm, q);
        mi_matvec(&lw->Wk, x_norm, k);
        mi_matvec(&lw->Wv, x_norm, v);


        mi_rope_apply(&m->rope, q, pos, cfg->n_heads, cfg->d_head);
        mi_rope_apply(&m->rope, k, pos, cfg->n_kv_heads, cfg->d_head);


        mi_cache_append(&m->cache, l, k, v);
        int seq_len;
        const float *K = mi_cache_keys(&m->cache, l, &seq_len);
        const float *V = mi_cache_values(&m->cache, l, &seq_len);


        mi_attention_decode(&m->attn, q, K, V, attn_out,
                           cfg->n_heads, cfg->n_kv_heads, cfg->d_head,
                           seq_len, pos, attn_scratch);


        mi_matvec(&lw->Wo, attn_out, o_proj);
        mi_vec_add(x, o_proj, x, d);


        mi_rmsnorm(x, lw->rms_ffn, x_norm, d, cfg->norm_eps);


        if (cfg->ffn_type == MI_FFN_SWIGLU)
            mi_swiglu_ffn(&lw->W1, &lw->W3, &lw->W2,
                          x_norm, ffn_out, ffn_scratch);
        else
            mi_relu_ffn(&lw->W1, &lw->W2,
                        x_norm, ffn_out, ffn_scratch);

        mi_vec_add(x, ffn_out, x, d);
    }


    mi_rmsnorm(x, m->w.rms_final, x_norm, d, cfg->norm_eps);
    mi_matvec(&m->w.out_proj, x_norm, logits);

    m->pos++;
}

void mi_model_forward_batch(MiModel *m, const int *tokens, int n,
                            float *logits, float *scratch) {

    for (int i = 0; i < n; i++) {
        mi_model_forward(m, tokens[i],
                         logits + (size_t)i * m->cfg.vocab_size,
                         scratch);
    }
}

#define MI_MODEL_MAGIC 0x4D494E49
#define MI_MODEL_VERSION 2

static MiStatus write_raw(FILE *f, const void *data, size_t bytes) {
    return (fwrite(data, 1, bytes, f) == bytes) ? MI_OK : MI_ERR_IO;
}
static MiStatus read_raw(FILE *f, void *data, size_t bytes) {
    return (fread(data, 1, bytes, f) == bytes) ? MI_OK : MI_ERR_IO;
}
static MiStatus write_i32(FILE *f, int32_t v) { return write_raw(f, &v, 4); }
static MiStatus write_f32(FILE *f, float v)   { return write_raw(f, &v, 4); }
static int32_t  read_i32(FILE *f)  { int32_t v = 0; read_raw(f, &v, 4); return v; }
static float    read_f32(FILE *f)  { float v = 0;   read_raw(f, &v, 4); return v; }

static MiStatus write_tensor(FILE *f, const MiTensor *t) {
    return write_raw(f, t->data, (size_t)t->rows * t->cols * sizeof(float));
}
static MiStatus read_tensor(FILE *f, MiTensor *t) {
    return read_raw(f, t->data, (size_t)t->rows * t->cols * sizeof(float));
}

static MiStatus write_header(FILE *f, const MiModelConfig *c) {
    MiStatus s = MI_OK;
    uint32_t magic = MI_MODEL_MAGIC;
    s = write_raw(f, &magic, 4);      if (s) return s;
    s = write_i32(f, MI_MODEL_VERSION); if (s) return s;
    s = write_i32(f, c->d_model);      if (s) return s;
    s = write_i32(f, c->n_heads);      if (s) return s;
    s = write_i32(f, c->n_kv_heads);   if (s) return s;
    s = write_i32(f, c->d_head);       if (s) return s;
    s = write_i32(f, c->d_ff);         if (s) return s;
    s = write_i32(f, c->n_layers);     if (s) return s;
    s = write_i32(f, c->vocab_size);   if (s) return s;
    s = write_i32(f, c->max_seq_len);  if (s) return s;
    s = write_f32(f, c->norm_eps);     if (s) return s;
    s = write_f32(f, c->rope_theta);   if (s) return s;
    s = write_i32(f, c->ffn_type);     if (s) return s;
    return MI_OK;
}

static MiModelConfig read_header(FILE *f) {
    MiModelConfig c;
    memset(&c, 0, sizeof(c));
    uint32_t magic = 0;
    read_raw(f, &magic, 4);
    MI_ASSERT(magic == MI_MODEL_MAGIC, "bad model magic: 0x%08X", magic);
    int version     = read_i32(f);
    MI_ASSERT(version == MI_MODEL_VERSION,
              "unsupported model version %d (expected %d)", version, MI_MODEL_VERSION);
    c.d_model       = read_i32(f);
    c.n_heads       = read_i32(f);
    c.n_kv_heads    = read_i32(f);
    c.d_head        = read_i32(f);
    c.d_ff          = read_i32(f);
    c.n_layers      = read_i32(f);
    c.vocab_size    = read_i32(f);
    c.max_seq_len   = read_i32(f);
    c.norm_eps      = read_f32(f);
    c.rope_theta    = read_f32(f);
    c.ffn_type      = read_i32(f);
    return c;
}

MiStatus mi_model_save(const MiModel *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return MI_ERR_IO;

    MiStatus s = write_header(f, &m->cfg);
    if (s) goto fail;
    s = write_tensor(f, &m->w.tok_emb); if (s) goto fail;

    for (int l = 0; l < m->cfg.n_layers; l++) {
        const MiLayerWeights *lw = &m->w.layers[l];
        s = write_tensor(f, &lw->Wq); if (s) goto fail;
        s = write_tensor(f, &lw->Wk); if (s) goto fail;
        s = write_tensor(f, &lw->Wv); if (s) goto fail;
        s = write_tensor(f, &lw->Wo); if (s) goto fail;
        s = write_tensor(f, &lw->W1); if (s) goto fail;
        s = write_tensor(f, &lw->W2); if (s) goto fail;
        if (m->cfg.ffn_type == MI_FFN_SWIGLU) {
            s = write_tensor(f, &lw->W3); if (s) goto fail;
        }
        s = write_raw(f, lw->rms_att, m->cfg.d_model * sizeof(float)); if (s) goto fail;
        s = write_raw(f, lw->rms_ffn, m->cfg.d_model * sizeof(float)); if (s) goto fail;
    }
    s = write_raw(f, m->w.rms_final, m->cfg.d_model * sizeof(float)); if (s) goto fail;
    s = write_tensor(f, &m->w.out_proj);
fail:
    fclose(f);
    return s;
}

MiStatus mi_model_load(MiModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return MI_ERR_IO;

    MiModelConfig c = read_header(f);
    MI_ASSERT(c.d_model == m->cfg.d_model && c.n_layers == m->cfg.n_layers,
              "config mismatch: file has d=%d L=%d, model has d=%d L=%d",
              c.d_model, c.n_layers, m->cfg.d_model, m->cfg.n_layers);

    MiStatus s = read_tensor(f, &m->w.tok_emb); if (s) goto fail;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        MiLayerWeights *lw = &m->w.layers[l];
        s = read_tensor(f, &lw->Wq); if (s) goto fail;
        s = read_tensor(f, &lw->Wk); if (s) goto fail;
        s = read_tensor(f, &lw->Wv); if (s) goto fail;
        s = read_tensor(f, &lw->Wo); if (s) goto fail;
        s = read_tensor(f, &lw->W1); if (s) goto fail;
        s = read_tensor(f, &lw->W2); if (s) goto fail;
        if (m->cfg.ffn_type == MI_FFN_SWIGLU) {
            s = read_tensor(f, &lw->W3); if (s) goto fail;
        }
        s = read_raw(f, lw->rms_att, m->cfg.d_model * sizeof(float)); if (s) goto fail;
        s = read_raw(f, lw->rms_ffn, m->cfg.d_model * sizeof(float)); if (s) goto fail;
    }
    s = read_raw(f, m->w.rms_final, m->cfg.d_model * sizeof(float)); if (s) goto fail;
    s = read_tensor(f, &m->w.out_proj);
fail:
    fclose(f);
    return s;
}

MiModel mi_model_load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    MI_ASSERT(f != NULL, "cannot open model: %s", path);
    MiModelConfig c = read_header(f);
    fclose(f);

    MI_INFO("loading model: d=%d h=%d kv=%d dh=%d ff=%d L=%d V=%d seq=%d %s",
            c.d_model, c.n_heads, c.n_kv_heads, c.d_head, c.d_ff,
            c.n_layers, c.vocab_size, c.max_seq_len,
            c.ffn_type == MI_FFN_SWIGLU ? "SwiGLU" : "ReLU");

    MiModel m = mi_model_create(c);
    MiStatus s = mi_model_load(&m, path);
    MI_ASSERT(s == MI_OK, "failed to load weights from %s", path);
    MI_INFO("model loaded (%.1f MB)",
            (double)mi_tensor_numel(&m.w.tok_emb) * 4.0 / 1e6 * 2
            + (double)c.n_layers * c.d_model * c.d_model * 4.0 * 7 / 1e6);
    return m;
}
