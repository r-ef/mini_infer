
#include "mi/attention.h"
#include "mi/ops.h"

static inline int kv_head_for(int q_head, int n_heads, int n_kv_heads) {
    return q_head * n_kv_heads / n_heads;
}

static void std_decode(MiAttention *a,
                       const float *q, const float *K, const float *V,
                       float *out,
                       int n_heads, int n_kv_heads, int d_head,
                       int seq_len, int pos,
                       float *scratch) {
    float scale = 1.0f / sqrtf((float)d_head);
    int kv_dim = n_kv_heads * d_head;
    float *scores = scratch;

    for (int h = 0; h < n_heads; h++) {
        int kvh = kv_head_for(h, n_heads, n_kv_heads);
        const float *qh = q + h * d_head;
        float *oh = out + h * d_head;


        int active = MI_MIN(seq_len, pos + 1);
        for (int t = 0; t < active; t++) {
            const float *kt = K + (size_t)t * kv_dim + kvh * d_head;
            scores[t] = mi_dot(qh, kt, d_head) * scale;
        }

        for (int t = active; t < seq_len; t++) scores[t] = -1e9f;


        if (a->alibi_slopes) {
            float slope = a->alibi_slopes[h];
            for (int t = 0; t < active; t++)
                scores[t] += slope * (float)(t - pos);
        }

        mi_softmax(scores, active);


        memset(oh, 0, d_head * sizeof(float));
        for (int t = 0; t < active; t++) {
            const float *vt = V + (size_t)t * kv_dim + kvh * d_head;
            for (int d = 0; d < d_head; d++)
                oh[d] += scores[t] * vt[d];
        }
    }
}

static void std_prefill(MiAttention *a,
                        const float *Q, const float *K, const float *V,
                        float *out,
                        int n_heads, int n_kv_heads, int d_head,
                        int n_pos, int seq_len,
                        float *scratch) {

    int q_stride = n_heads * d_head;
    for (int p = 0; p < n_pos; p++) {
        std_decode(a,
                   Q + p * q_stride, K, V,
                   out + p * q_stride,
                   n_heads, n_kv_heads, d_head,
                   seq_len, p,
                   scratch);
    }
}

static void std_destroy(MiAttention *a) {
    if (a->alibi_slopes) { free(a->alibi_slopes); a->alibi_slopes = NULL; }
}

static const MiAttentionVT std_vt = {
    .name    = "standard",
    .decode  = std_decode,
    .prefill = std_prefill,
    .destroy = std_destroy,
};

MiAttention mi_attention_standard(void) {
    return (MiAttention){ .vt = &std_vt, .ctx = NULL, .alibi_slopes = NULL };
}

static void flash_decode(MiAttention *a,
                         const float *q, const float *K, const float *V,
                         float *out,
                         int n_heads, int n_kv_heads, int d_head,
                         int seq_len, int pos,
                         float *scratch) {
    MI_UNUSED(scratch);
    float scale = 1.0f / sqrtf((float)d_head);
    int kv_dim = n_kv_heads * d_head;
    int active = MI_MIN(seq_len, pos + 1);

    for (int h = 0; h < n_heads; h++) {
        int kvh = kv_head_for(h, n_heads, n_kv_heads);
        const float *qh = q + h * d_head;
        float *oh = out + h * d_head;

        float m = -FLT_MAX;
        float l = 0.0f;
        memset(oh, 0, d_head * sizeof(float));

        for (int t = 0; t < active; t++) {
            const float *kt = K + (size_t)t * kv_dim + kvh * d_head;
            float s = mi_dot(qh, kt, d_head) * scale;


            if (a->alibi_slopes)
                s += a->alibi_slopes[h] * (float)(t - pos);

            if (s > m) {

                float correction = expf(m - s);
                for (int d = 0; d < d_head; d++) oh[d] *= correction;
                l *= correction;
                m = s;
            }

            float w = expf(s - m);
            l += w;

            const float *vt = V + (size_t)t * kv_dim + kvh * d_head;
            for (int d = 0; d < d_head; d++)
                oh[d] += w * vt[d];
        }

        if (l > 0.0f) {
            float inv = 1.0f / l;
            for (int d = 0; d < d_head; d++) oh[d] *= inv;
        }
    }
}

static void flash_prefill(MiAttention *a,
                          const float *Q, const float *K, const float *V,
                          float *out,
                          int n_heads, int n_kv_heads, int d_head,
                          int n_pos, int seq_len,
                          float *scratch) {
    int q_stride = n_heads * d_head;
    for (int p = 0; p < n_pos; p++) {
        flash_decode(a,
                     Q + p * q_stride, K, V,
                     out + p * q_stride,
                     n_heads, n_kv_heads, d_head,
                     seq_len, p,
                     scratch);
    }
}

static void flash_destroy(MiAttention *a) {
    if (a->alibi_slopes) { free(a->alibi_slopes); a->alibi_slopes = NULL; }
}

static const MiAttentionVT flash_vt = {
    .name    = "flash",
    .decode  = flash_decode,
    .prefill = flash_prefill,
    .destroy = flash_destroy,
};

MiAttention mi_attention_flash(void) {
    return (MiAttention){ .vt = &flash_vt, .ctx = NULL, .alibi_slopes = NULL };
}

static inline float elu_plus_1(float x) {
    return x > 0.0f ? x + 1.0f : expf(x);
}

static void linear_decode(MiAttention *a,
                          const float *q, const float *K, const float *V,
                          float *out,
                          int n_heads, int n_kv_heads, int d_head,
                          int seq_len, int pos,
                          float *scratch) {
    MI_UNUSED(a);
    int kv_dim = n_kv_heads * d_head;
    int active = MI_MIN(seq_len, pos + 1);


    for (int h = 0; h < n_heads; h++) {
        int kvh = kv_head_for(h, n_heads, n_kv_heads);
        const float *qh = q + h * d_head;
        float *oh = out + h * d_head;

        float *S = scratch;
        float *z = scratch + d_head * d_head;
        float *phi_q = z + d_head;


        memset(S, 0, d_head * d_head * sizeof(float));
        memset(z, 0, d_head * sizeof(float));

        for (int t = 0; t < active; t++) {
            const float *kt = K + (size_t)t * kv_dim + kvh * d_head;
            const float *vt = V + (size_t)t * kv_dim + kvh * d_head;

            for (int i = 0; i < d_head; i++) {
                float phi_k_i = elu_plus_1(kt[i]);
                z[i] += phi_k_i;
                for (int j = 0; j < d_head; j++)
                    S[i * d_head + j] += phi_k_i * vt[j];
            }
        }


        for (int i = 0; i < d_head; i++)
            phi_q[i] = elu_plus_1(qh[i]);


        float denom = mi_dot(phi_q, z, d_head);
        if (denom < 1e-12f) denom = 1e-12f;

        for (int j = 0; j < d_head; j++) {
            float num = 0.0f;
            for (int i = 0; i < d_head; i++)
                num += phi_q[i] * S[i * d_head + j];
            oh[j] = num / denom;
        }
    }
}

static void linear_prefill(MiAttention *a,
                           const float *Q, const float *K, const float *V,
                           float *out,
                           int n_heads, int n_kv_heads, int d_head,
                           int n_pos, int seq_len,
                           float *scratch) {
    int q_stride = n_heads * d_head;
    for (int p = 0; p < n_pos; p++) {
        linear_decode(a,
                      Q + p * q_stride, K, V,
                      out + p * q_stride,
                      n_heads, n_kv_heads, d_head,
                      seq_len, p,
                      scratch);
    }
}

static void linear_destroy(MiAttention *a) {
    if (a->alibi_slopes) { free(a->alibi_slopes); a->alibi_slopes = NULL; }
}

static const MiAttentionVT linear_vt = {
    .name    = "linear",
    .decode  = linear_decode,
    .prefill = linear_prefill,
    .destroy = linear_destroy,
};

MiAttention mi_attention_linear(void) {
    return (MiAttention){ .vt = &linear_vt, .ctx = NULL, .alibi_slopes = NULL };
}

int mi_attention_scratch_size(const MiAttention *a,
                              int n_heads, int seq_len) {
    MI_UNUSED(n_heads);
    if (a->vt == &linear_vt) {

        return seq_len * seq_len + seq_len * 4;
    }
    if (a->vt == &flash_vt) return 0;
    return seq_len;
}
