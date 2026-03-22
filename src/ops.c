
#include "mi/ops.h"

#if !defined(MI_NO_SIMD)
  #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define MI_NEON 1
  #elif defined(__AVX2__) && defined(__FMA__)
    #include <immintrin.h>
    #define MI_AVX2 1
  #elif defined(__SSE3__)
    #include <immintrin.h>
    #define MI_SSE 1
  #endif
#endif

#if MI_AVX2
static inline float hsum_avx(__m256 v) {
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

float mi_dot(const float *a, const float *b, int n) {
#if MI_NEON
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a+i),    vld1q_f32(b+i));
        s1 = vfmaq_f32(s1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        s2 = vfmaq_f32(s2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        s3 = vfmaq_f32(s3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0,s1), vaddq_f32(s2,s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#elif MI_AVX2
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i),   s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), s1);
    }
    float s = hsum_avx(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

void mi_vec_add(const float *a, const float *b, float *out, int n) {
#if MI_NEON
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vst1q_f32(out+i,    vaddq_f32(vld1q_f32(a+i),    vld1q_f32(b+i)));
        vst1q_f32(out+i+4,  vaddq_f32(vld1q_f32(a+i+4),  vld1q_f32(b+i+4)));
        vst1q_f32(out+i+8,  vaddq_f32(vld1q_f32(a+i+8),  vld1q_f32(b+i+8)));
        vst1q_f32(out+i+12, vaddq_f32(vld1q_f32(a+i+12), vld1q_f32(b+i+12)));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
#elif MI_AVX2
    int i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_ps(out+i,   _mm256_add_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i)));
        _mm256_storeu_ps(out+i+8, _mm256_add_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8)));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

void mi_vec_sub(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void mi_vec_mul(const float *a, const float *b, float *out, int n) {
#if MI_NEON
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vst1q_f32(out+i,    vmulq_f32(vld1q_f32(a+i),    vld1q_f32(b+i)));
        vst1q_f32(out+i+4,  vmulq_f32(vld1q_f32(a+i+4),  vld1q_f32(b+i+4)));
        vst1q_f32(out+i+8,  vmulq_f32(vld1q_f32(a+i+8),  vld1q_f32(b+i+8)));
        vst1q_f32(out+i+12, vmulq_f32(vld1q_f32(a+i+12), vld1q_f32(b+i+12)));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
#elif MI_AVX2
    int i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_ps(out+i,   _mm256_mul_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i)));
        _mm256_storeu_ps(out+i+8, _mm256_mul_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8)));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

void mi_vec_scale(const float *x, float s, float *out, int n) {
#if MI_NEON
    float32x4_t sv = vdupq_n_f32(s);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vst1q_f32(out+i,    vmulq_f32(vld1q_f32(x+i),    sv));
        vst1q_f32(out+i+4,  vmulq_f32(vld1q_f32(x+i+4),  sv));
        vst1q_f32(out+i+8,  vmulq_f32(vld1q_f32(x+i+8),  sv));
        vst1q_f32(out+i+12, vmulq_f32(vld1q_f32(x+i+12), sv));
    }
    for (; i < n; i++) out[i] = x[i] * s;
#elif MI_AVX2
    __m256 sv = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_ps(out+i,   _mm256_mul_ps(_mm256_loadu_ps(x+i),   sv));
        _mm256_storeu_ps(out+i+8, _mm256_mul_ps(_mm256_loadu_ps(x+i+8), sv));
    }
    for (; i < n; i++) out[i] = x[i] * s;
#else
    for (int i = 0; i < n; i++) out[i] = x[i] * s;
#endif
}

void mi_vec_add_scaled(const float *a, const float *b, float s,
                       float *out, int n) {
#if MI_NEON
    float32x4_t sv = vdupq_n_f32(s);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(out+i, vfmaq_f32(vld1q_f32(a+i), vld1q_f32(b+i), sv));
    for (; i < n; i++) out[i] = a[i] + s * b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] + s * b[i];
#endif
}

void mi_vec_copy(const float *src, float *dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}
void mi_vec_fill(float *x, int n, float val) {
    for (int i = 0; i < n; i++) x[i] = val;
}
float mi_vec_max(const float *x, int n) {
#if MI_NEON
    float32x4_t mx = vdupq_n_f32(-FLT_MAX);
    int i = 0;
    for (; i + 3 < n; i += 4)
        mx = vmaxq_f32(mx, vld1q_f32(x+i));
    float m = vmaxvq_f32(mx);
    for (; i < n; i++) if (x[i] > m) m = x[i];
    return m;
#else
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    return m;
#endif
}
float mi_vec_min(const float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] < m) m = x[i];
    return m;
}
float mi_vec_sum(const float *x, int n) {
#if MI_NEON
    float32x4_t s = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < n; i += 4) s = vaddq_f32(s, vld1q_f32(x+i));
    float r = vaddvq_f32(s);
    for (; i < n; i++) r += x[i];
    return r;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
#endif
}
float mi_vec_norm2(const float *x, int n) {
    return sqrtf(mi_dot(x, x, n));
}
float mi_vec_cosine(const float *a, const float *b, int n) {
    float d  = mi_dot(a, b, n);
    float na = mi_vec_norm2(a, n);
    float nb = mi_vec_norm2(b, n);
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return d / (na * nb);
}

void mi_matvec(const MiTensor *W, const float *x, float *out) {
    const int rows = W->rows;
    const int cols = W->cols;
    int r = 0;

#if MI_NEON

    for (; r + 3 < rows; r += 4) {
        const float *r0 = W->data + (r+0) * cols;
        const float *r1 = W->data + (r+1) * cols;
        const float *r2 = W->data + (r+2) * cols;
        const float *r3 = W->data + (r+3) * cols;

        float32x4_t a0 = vdupq_n_f32(0), b0 = vdupq_n_f32(0);
        float32x4_t a1 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), b2 = vdupq_n_f32(0);
        float32x4_t a3 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);

        int c = 0;
        for (; c + 7 < cols; c += 8) {
            float32x4_t xa = vld1q_f32(x + c);
            float32x4_t xb = vld1q_f32(x + c + 4);
            a0 = vfmaq_f32(a0, vld1q_f32(r0+c),   xa);
            b0 = vfmaq_f32(b0, vld1q_f32(r0+c+4), xb);
            a1 = vfmaq_f32(a1, vld1q_f32(r1+c),   xa);
            b1 = vfmaq_f32(b1, vld1q_f32(r1+c+4), xb);
            a2 = vfmaq_f32(a2, vld1q_f32(r2+c),   xa);
            b2 = vfmaq_f32(b2, vld1q_f32(r2+c+4), xb);
            a3 = vfmaq_f32(a3, vld1q_f32(r3+c),   xa);
            b3 = vfmaq_f32(b3, vld1q_f32(r3+c+4), xb);
        }

        out[r+0] = vaddvq_f32(vaddq_f32(a0, b0));
        out[r+1] = vaddvq_f32(vaddq_f32(a1, b1));
        out[r+2] = vaddvq_f32(vaddq_f32(a2, b2));
        out[r+3] = vaddvq_f32(vaddq_f32(a3, b3));

        for (; c < cols; c++) {
            float xc = x[c];
            out[r+0] += r0[c] * xc;
            out[r+1] += r1[c] * xc;
            out[r+2] += r2[c] * xc;
            out[r+3] += r3[c] * xc;
        }
    }

    for (; r < rows; r++) {
        const float *row = W->data + r * cols;
        float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            s0 = vfmaq_f32(s0, vld1q_f32(row+c),   vld1q_f32(x+c));
            s1 = vfmaq_f32(s1, vld1q_f32(row+c+4), vld1q_f32(x+c+4));
        }
        float s = vaddvq_f32(vaddq_f32(s0, s1));
        for (; c < cols; c++) s += row[c] * x[c];
        out[r] = s;
    }

#elif MI_AVX2

    for (; r + 3 < rows; r += 4) {
        const float *r0 = W->data + (r+0) * cols;
        const float *r1 = W->data + (r+1) * cols;
        const float *r2 = W->data + (r+2) * cols;
        const float *r3 = W->data + (r+3) * cols;

        __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();

        int c = 0;
        for (; c + 15 < cols; c += 16) {
            __m256 xa = _mm256_loadu_ps(x + c);
            __m256 xb = _mm256_loadu_ps(x + c + 8);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0+c),   xa, a0);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0+c+8), xb, b0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1+c),   xa, a1);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1+c+8), xb, b1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2+c),   xa, a2);
            b2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2+c+8), xb, b2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3+c),   xa, a3);
            b3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3+c+8), xb, b3);
        }

        out[r+0] = hsum_avx(_mm256_add_ps(a0, b0));
        out[r+1] = hsum_avx(_mm256_add_ps(a1, b1));
        out[r+2] = hsum_avx(_mm256_add_ps(a2, b2));
        out[r+3] = hsum_avx(_mm256_add_ps(a3, b3));

        for (; c < cols; c++) {
            float xc = x[c];
            out[r+0] += r0[c] * xc;
            out[r+1] += r1[c] * xc;
            out[r+2] += r2[c] * xc;
            out[r+3] += r3[c] * xc;
        }
    }
    for (; r < rows; r++) {
        const float *row = W->data + r * cols;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        int c = 0;
        for (; c + 15 < cols; c += 16) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(row+c),   _mm256_loadu_ps(x+c),   s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(row+c+8), _mm256_loadu_ps(x+c+8), s1);
        }
        float s = hsum_avx(_mm256_add_ps(s0, s1));
        for (; c < cols; c++) s += row[c] * x[c];
        out[r] = s;
    }

#else

    for (; r < rows; r++) {
        const float *row = W->data + r * cols;
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += row[c] * x[c];
        out[r] = s;
    }
#endif
}

void mi_relu(float *x, int n) {
#if MI_NEON
    float32x4_t z = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(x+i, vmaxq_f32(vld1q_f32(x+i), z));
    for (; i < n; i++) if (x[i] < 0) x[i] = 0;
#else
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#endif
}

void mi_silu(float *x, int n) {

    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

void mi_gelu(float *x, int n) {
    const float c = 0.7978845608f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(c * (v + 0.044715f * v * v * v));
        x[i] = 0.5f * v * (1.0f + t);
    }
}

void mi_rmsnorm(const float *x, const float *w, float *out,
                int n, float eps) {

    float ss = mi_dot(x, x, n);
    float scale = 1.0f / sqrtf(ss / (float)n + eps);

#if MI_NEON
    float32x4_t sv = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(out+i, vmulq_f32(vmulq_f32(vld1q_f32(x+i), sv),
                                     vld1q_f32(w+i)));
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
#elif MI_AVX2
    __m256 sv = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(out+i,
            _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x+i), sv),
                          _mm256_loadu_ps(w+i)));
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
#else
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
#endif
}

void mi_layernorm(const float *x, const float *gamma, const float *beta,
                  float *out, int n, float eps) {
    float mean = mi_vec_sum(x, n) / (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float scale = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++)
        out[i] = (x[i] - mean) * scale * gamma[i] + beta[i];
}

void mi_softmax(float *x, int n) {
    float m = mi_vec_max(x, n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = 1.0f / s;
    mi_vec_scale(x, inv, x, n);
}

void mi_log_softmax(const float *x, float *out, int n) {
    float m = mi_vec_max(x, n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += expf(x[i] - m);
    float lse = m + logf(s);
    for (int i = 0; i < n; i++) out[i] = x[i] - lse;
}

int mi_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}
int mi_argmin(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] < x[best]) best = i;
    return best;
}

void mi_swiglu_ffn(const MiTensor *W_gate, const MiTensor *W_up,
                   const MiTensor *W_down,
                   const float *x, float *out, float *scratch) {
    int d_ff = W_gate->rows;
    float *gate = scratch;
    float *up   = scratch + d_ff;

    mi_matvec(W_gate, x, gate);
    mi_silu(gate, d_ff);
    mi_matvec(W_up, x, up);
    mi_vec_mul(gate, up, gate, d_ff);
    mi_matvec(W_down, gate, out);
}

void mi_relu_ffn(const MiTensor *W1, const MiTensor *W2,
                 const float *x, float *out, float *scratch) {
    int d_ff = W1->rows;
    float *hidden = scratch;
    mi_matvec(W1, x, hidden);
    mi_relu(hidden, d_ff);
    mi_matvec(W2, hidden, out);
}
