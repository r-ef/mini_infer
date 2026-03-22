/* rope.c — RoPE variants + ALiBi */
#include "mi/rope.h"
#include <math.h>

/* ── Common rotation kernel ── */
static void rope_rotate(float *vec, int n_heads, int d_head,
                        int pos, float theta) {
    for (int h = 0; h < n_heads; h++) {
        float *head = vec + h * d_head;
        for (int i = 0; i < d_head; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)d_head);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = head[i];
            float x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  1. Standard RoPE                                                 ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct { float theta; } StdCtx;

static void std_apply(MiRoPE *r, float *vec,
                      int pos, int n_heads, int d_head) {
    StdCtx *c = (StdCtx *)r->ctx;
    rope_rotate(vec, n_heads, d_head, pos, c->theta);
}

static void rope_ctx_destroy(MiRoPE *r) { free(r->ctx); }

static const MiRoPEVT std_vt = {
    .name = "standard", .apply = std_apply,
    .bias = NULL, .destroy = rope_ctx_destroy,
};

MiRoPE mi_rope_standard(float theta) {
    StdCtx *c = (StdCtx *)malloc(sizeof(StdCtx));
    MI_CHECK_OOM(c);
    c->theta = theta;
    return (MiRoPE){ .vt = &std_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  2. NTK-Aware — scale θ for longer contexts                     ║
 * ║                                                                   ║
 * ║  θ' = θ · α^(d/(d-2))  where α = scale_factor                  ║
 * ║  This adjusts all frequency bands proportionally.                ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct { float theta_scaled; } NTKCtx;

static void ntk_apply(MiRoPE *r, float *vec,
                      int pos, int n_heads, int d_head) {
    NTKCtx *c = (NTKCtx *)r->ctx;
    rope_rotate(vec, n_heads, d_head, pos, c->theta_scaled);
}

static const MiRoPEVT ntk_vt = {
    .name = "ntk", .apply = ntk_apply,
    .bias = NULL, .destroy = rope_ctx_destroy,
};

MiRoPE mi_rope_ntk(float theta, float scale_factor) {
    NTKCtx *c = (NTKCtx *)malloc(sizeof(NTKCtx));
    MI_CHECK_OOM(c);
    /* For typical d_head=128: exponent ≈ d/(d-2) ≈ 1.016.
     * We use a representative d_head=128 for the exponent. */
    float d = 128.0f;
    c->theta_scaled = theta * powf(scale_factor, d / (d - 2.0f));
    return (MiRoPE){ .vt = &ntk_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  3. YaRN — NTK + frequency interpolation + attention scaling    ║
 * ║                                                                   ║
 * ║  Low frequencies: interpolated (divide pos by scale).            ║
 * ║  High frequencies: kept at original scale.                       ║
 * ║  Middle: linearly blended.                                       ║
 * ║  Plus an attention temperature correction ~ sqrt(1 + ln(s)/ln(L))║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float theta;
    float scale;
    int   orig_max;
    float attn_factor; /* multiplicative scaling for attention scores */
} YaRNCtx;

static void yarn_apply(MiRoPE *r, float *vec,
                       int pos, int n_heads, int d_head) {
    YaRNCtx *c = (YaRNCtx *)r->ctx;
    float alpha = c->scale;

    /* Wavelength thresholds for interpolation boundaries.
     * β_fast = 32, β_slow = 1 (from YaRN paper defaults) */
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float low_freq_factor  = (float)c->orig_max / beta_slow;
    float high_freq_factor = (float)c->orig_max / beta_fast;

    for (int h = 0; h < n_heads; h++) {
        float *head = vec + h * d_head;
        for (int i = 0; i < d_head; i += 2) {
            float freq = 1.0f / powf(c->theta, (float)i / (float)d_head);
            float wavelength = 2.0f * (float)M_PI / freq;

            float effective_pos;
            if (wavelength < high_freq_factor) {
                /* High frequency: no interpolation */
                effective_pos = (float)pos;
            } else if (wavelength > low_freq_factor) {
                /* Low frequency: full interpolation */
                effective_pos = (float)pos / alpha;
            } else {
                /* Middle: linear ramp */
                float t = (wavelength - high_freq_factor)
                        / (low_freq_factor - high_freq_factor);
                float interp_pos = (float)pos / alpha;
                effective_pos = (1.0f - t) * (float)pos + t * interp_pos;
            }

            float angle = effective_pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = head[i];
            float x1 = head[i + 1];
            head[i]     = x0 * cos_a - x1 * sin_a;
            head[i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

static const MiRoPEVT yarn_vt = {
    .name = "yarn", .apply = yarn_apply,
    .bias = NULL, .destroy = rope_ctx_destroy,
};

MiRoPE mi_rope_yarn(float theta, float scale_factor, int orig_max) {
    YaRNCtx *c = (YaRNCtx *)malloc(sizeof(YaRNCtx));
    MI_CHECK_OOM(c);
    c->theta    = theta;
    c->scale    = scale_factor;
    c->orig_max = orig_max;
    c->attn_factor = sqrtf(1.0f + logf(scale_factor) / logf((float)orig_max));
    return (MiRoPE){ .vt = &yarn_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  4. Dynamic NTK — auto-scale θ when position exceeds training   ║
 * ║                                                                   ║
 * ║  At position p > L:  θ' = θ · (α)^(d/(d-2))                    ║
 * ║  where α = p / L (dynamic per-position).                        ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float theta;
    int   orig_max;
} DynCtx;

static void dyn_apply(MiRoPE *r, float *vec,
                      int pos, int n_heads, int d_head) {
    DynCtx *c = (DynCtx *)r->ctx;
    float theta_eff = c->theta;

    if (pos >= c->orig_max) {
        float alpha = (float)(pos + 1) / (float)c->orig_max;
        float d = (float)d_head;
        theta_eff = c->theta * powf(alpha, d / (d - 2.0f));
    }

    rope_rotate(vec, n_heads, d_head, pos, theta_eff);
}

static const MiRoPEVT dyn_vt = {
    .name = "dynamic", .apply = dyn_apply,
    .bias = NULL, .destroy = rope_ctx_destroy,
};

MiRoPE mi_rope_dynamic(float theta, int orig_max) {
    DynCtx *c = (DynCtx *)malloc(sizeof(DynCtx));
    MI_CHECK_OOM(c);
    c->theta    = theta;
    c->orig_max = orig_max;
    return (MiRoPE){ .vt = &dyn_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  5. ALiBi — Attention with Linear Biases                        ║
 * ║                                                                   ║
 * ║  No rotation.  Instead, bias(h, q, k) = slope_h · (k − q).     ║
 * ║  slope_h = 2^(−8·h/H) for head h ∈ [0, H).                     ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

typedef struct {
    float *slopes;    /* [n_heads] */
    int    n_heads;
} ALiBiCtx;

static void alibi_apply(MiRoPE *r, float *vec,
                        int pos, int n_heads, int d_head) {
    /* ALiBi does not modify the vectors */
    MI_UNUSED(r); MI_UNUSED(vec);
    MI_UNUSED(pos); MI_UNUSED(n_heads); MI_UNUSED(d_head);
}

static float alibi_bias(MiRoPE *r, int head, int q_pos, int k_pos) {
    ALiBiCtx *c = (ALiBiCtx *)r->ctx;
    if (head < 0 || head >= c->n_heads) return 0.0f;
    return c->slopes[head] * (float)(k_pos - q_pos);
}

static void alibi_destroy(MiRoPE *r) {
    ALiBiCtx *c = (ALiBiCtx *)r->ctx;
    free(c->slopes); free(c);
}

static const MiRoPEVT alibi_vt = {
    .name = "alibi", .apply = alibi_apply,
    .bias = alibi_bias, .destroy = alibi_destroy,
};

MiRoPE mi_rope_alibi(int n_heads) {
    ALiBiCtx *c = (ALiBiCtx *)malloc(sizeof(ALiBiCtx));
    MI_CHECK_OOM(c);
    c->n_heads = n_heads;
    c->slopes  = (float *)malloc(n_heads * sizeof(float));
    MI_CHECK_OOM(c->slopes);
    /* Geometric slopes: 2^(−8/H), 2^(−16/H), … */
    float base = powf(2.0f, -8.0f / (float)n_heads);
    for (int h = 0; h < n_heads; h++)
        c->slopes[h] = powf(base, (float)(h + 1));
    return (MiRoPE){ .vt = &alibi_vt, .ctx = c };
}

/* ╔═══════════════════════════════════════════════════════════════════╗
 * ║  6. None — identity, for ablation / testing                      ║
 * ╚═══════════════════════════════════════════════════════════════════╝ */

static void none_apply(MiRoPE *r, float *vec,
                       int pos, int n_heads, int d_head) {
    MI_UNUSED(r); MI_UNUSED(vec);
    MI_UNUSED(pos); MI_UNUSED(n_heads); MI_UNUSED(d_head);
}
static void none_destroy(MiRoPE *r) { MI_UNUSED(r); }

static const MiRoPEVT none_vt = {
    .name = "none", .apply = none_apply,
    .bias = NULL, .destroy = none_destroy,
};

MiRoPE mi_rope_none(void) {
    return (MiRoPE){ .vt = &none_vt, .ctx = NULL };
}
