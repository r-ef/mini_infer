
#include "mi/quant.h"
#include "mi/ops.h"

MiQInt8 mi_quant_int8_absmax(const float *data, int n) {
    MiQInt8 q;
    q.n    = n;
    q.data = (int8_t *)malloc(n);
    MI_CHECK_OOM(q.data);

    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > amax) amax = a;
    }
    q.scale = amax / 127.0f;

    if (amax < 1e-10f) {
        memset(q.data, 0, n);
    } else {
        float inv = 127.0f / amax;
        for (int i = 0; i < n; i++) {
            int v = (int)roundf(data[i] * inv);
            q.data[i] = (int8_t)MI_CLAMP(v, -127, 127);
        }
    }
    return q;
}

void mi_dequant_int8_absmax(const MiQInt8 *q, float *out) {
    for (int i = 0; i < q->n; i++)
        out[i] = (float)q->data[i] * q->scale;
}

void mi_quant_int8_free(MiQInt8 *q) {
    free(q->data); q->data = NULL;
}

MiQInt8ZP mi_quant_int8_zp(const float *data, int n) {
    MiQInt8ZP q;
    q.n    = n;
    q.data = (int8_t *)malloc(n);
    MI_CHECK_OOM(q.data);

    float fmin = data[0], fmax = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < fmin) fmin = data[i];
        if (data[i] > fmax) fmax = data[i];
    }
    q.scale = (fmax - fmin) / 254.0f;
    float mid = (fmax + fmin) * 0.5f;
    q.zero_point = (int8_t)roundf(-mid / q.scale);

    if (q.scale < 1e-10f) {
        memset(q.data, 0, n);
        q.scale = 1e-10f;
    } else {
        float inv = 1.0f / q.scale;
        for (int i = 0; i < n; i++) {
            int v = (int)roundf(data[i] * inv) + q.zero_point;
            q.data[i] = (int8_t)MI_CLAMP(v, -127, 127);
        }
    }
    return q;
}

void mi_dequant_int8_zp(const MiQInt8ZP *q, float *out) {
    for (int i = 0; i < q->n; i++)
        out[i] = ((float)q->data[i] - (float)q->zero_point) * q->scale;
}

void mi_quant_int8_zp_free(MiQInt8ZP *q) {
    free(q->data); q->data = NULL;
}

MiQInt4 mi_quant_int4_group(const float *data, int n, int group_size) {
    MiQInt4 q;
    q.n          = n;
    q.group_size = group_size;
    q.n_groups   = (n + group_size - 1) / group_size;
    q.data       = (uint8_t *)calloc((n + 1) / 2, 1);
    q.scales     = (float *)calloc(q.n_groups, sizeof(float));
    q.zeros      = NULL;
    MI_CHECK_OOM(q.data); MI_CHECK_OOM(q.scales);

    for (int g = 0; g < q.n_groups; g++) {
        int start = g * group_size;
        int end   = MI_MIN(start + group_size, n);

        float amax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = fabsf(data[i]);
            if (a > amax) amax = a;
        }
        q.scales[g] = amax / 7.0f;

        float inv = (amax > 1e-10f) ? 7.0f / amax : 0.0f;
        for (int i = start; i < end; i++) {
            int v = (int)roundf(data[i] * inv);
            v = MI_CLAMP(v, -8, 7);
            uint8_t nibble = (uint8_t)(v + 8);
            int byte_idx = i / 2;
            if (i % 2 == 0)
                q.data[byte_idx] = (q.data[byte_idx] & 0xF0) | (nibble & 0x0F);
            else
                q.data[byte_idx] = (q.data[byte_idx] & 0x0F) | (nibble << 4);
        }
    }
    return q;
}

void mi_dequant_int4_group(const MiQInt4 *q, float *out) {
    for (int g = 0; g < q->n_groups; g++) {
        int start = g * q->group_size;
        int end   = MI_MIN(start + q->group_size, q->n);
        float scale = q->scales[g];

        for (int i = start; i < end; i++) {
            int byte_idx = i / 2;
            uint8_t nibble;
            if (i % 2 == 0)
                nibble = q->data[byte_idx] & 0x0F;
            else
                nibble = (q->data[byte_idx] >> 4) & 0x0F;
            int val = (int)nibble - 8;
            out[i] = (float)val * scale;
        }
    }
}

void mi_quant_int4_free(MiQInt4 *q) {
    free(q->data); free(q->scales); free(q->zeros);
    q->data = NULL; q->scales = NULL; q->zeros = NULL;
}

int mi_quant_q4_0_nblocks(int n) {
    return (n + MI_Q4_0_BLOCK - 1) / MI_Q4_0_BLOCK;
}

void mi_quant_q4_0(const float *data, MiBlockQ4_0 *blocks, int n) {
    int nb = mi_quant_q4_0_nblocks(n);
    for (int b = 0; b < nb; b++) {
        int start = b * MI_Q4_0_BLOCK;
        int end   = MI_MIN(start + MI_Q4_0_BLOCK, n);

        float amax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = fabsf(data[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 8.0f;
        blocks[b].scale = mi_f32_to_f16(scale);

        float inv = (amax > 1e-10f) ? 8.0f / amax : 0.0f;
        memset(blocks[b].qs, 0, MI_Q4_0_BLOCK / 2);
        for (int i = start; i < end; i++) {
            int j = i - start;
            int v = (int)roundf(data[i] * inv) + 8;
            v = MI_CLAMP(v, 0, 15);
            int byte_idx = j / 2;
            if (j % 2 == 0)
                blocks[b].qs[byte_idx] |= (uint8_t)(v & 0x0F);
            else
                blocks[b].qs[byte_idx] |= (uint8_t)(v << 4);
        }
    }
}

void mi_dequant_q4_0(const MiBlockQ4_0 *blocks, float *out, int n) {
    int nb = mi_quant_q4_0_nblocks(n);
    for (int b = 0; b < nb; b++) {
        float scale = mi_f16_to_f32(blocks[b].scale);
        int start = b * MI_Q4_0_BLOCK;
        int end   = MI_MIN(start + MI_Q4_0_BLOCK, n);
        for (int i = start; i < end; i++) {
            int j = i - start;
            int byte_idx = j / 2;
            uint8_t nibble;
            if (j % 2 == 0)
                nibble = blocks[b].qs[byte_idx] & 0x0F;
            else
                nibble = (blocks[b].qs[byte_idx] >> 4) & 0x0F;
            out[i] = ((float)nibble - 8.0f) * scale;
        }
    }
}

int mi_quant_q8_0_nblocks(int n) {
    return (n + MI_Q8_0_BLOCK - 1) / MI_Q8_0_BLOCK;
}

void mi_quant_q8_0(const float *data, MiBlockQ8_0 *blocks, int n) {
    int nb = mi_quant_q8_0_nblocks(n);
    for (int b = 0; b < nb; b++) {
        int start = b * MI_Q8_0_BLOCK;
        int end   = MI_MIN(start + MI_Q8_0_BLOCK, n);

        float amax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = fabsf(data[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        blocks[b].scale = mi_f32_to_f16(scale);

        float inv = (amax > 1e-10f) ? 127.0f / amax : 0.0f;
        memset(blocks[b].qs, 0, MI_Q8_0_BLOCK);
        for (int i = start; i < end; i++) {
            int j = i - start;
            int v = (int)roundf(data[i] * inv);
            blocks[b].qs[j] = (int8_t)MI_CLAMP(v, -127, 127);
        }
    }
}

void mi_dequant_q8_0(const MiBlockQ8_0 *blocks, float *out, int n) {
    int nb = mi_quant_q8_0_nblocks(n);
    for (int b = 0; b < nb; b++) {
        float scale = mi_f16_to_f32(blocks[b].scale);
        int start = b * MI_Q8_0_BLOCK;
        int end   = MI_MIN(start + MI_Q8_0_BLOCK, n);
        for (int i = start; i < end; i++)
            out[i] = (float)blocks[b].qs[i - start] * scale;
    }
}

void mi_matvec_int8(const int8_t *W, const float *row_scales,
                    const float *x, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const int8_t *row = W + (size_t)r * cols;
        int32_t acc = 0;


        float sum = 0.0f;
        for (int c = 0; c < cols; c++)
            sum += (float)row[c] * x[c];
        out[r] = sum * row_scales[r];
        MI_UNUSED(acc);
    }
}

void mi_matvec_q4_0(const MiBlockQ4_0 *W, const float *x,
                    float *out, int rows, int cols) {
    int blocks_per_row = mi_quant_q4_0_nblocks(cols);
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const MiBlockQ4_0 *row_blocks = W + (size_t)r * blocks_per_row;
        int col = 0;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = mi_f16_to_f32(row_blocks[b].scale);
            for (int j = 0; j < MI_Q4_0_BLOCK / 2 && col + 1 < cols; j++) {
                uint8_t byte = row_blocks[b].qs[j];
                float w0 = ((float)(byte & 0x0F) - 8.0f) * scale;
                float w1 = ((float)((byte >> 4) & 0x0F) - 8.0f) * scale;
                sum += w0 * x[col] + w1 * x[col + 1];
                col += 2;
            }
        }
        out[r] = sum;
    }
}

uint16_t mi_f32_to_f16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);

    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127;
    uint32_t frac = u & 0x7FFFFF;

    if (exp > 15)  return sign | 0x7C00;
    if (exp < -14) return sign;

    uint16_t hexp  = (uint16_t)((exp + 15) << 10);
    uint16_t hfrac = (uint16_t)(frac >> 13);
    return sign | hexp | hfrac;
}

float mi_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    if (exp == 0) {
        if (frac == 0) {
            float f; uint32_t u = sign; memcpy(&f, &u, 4); return f;
        }

        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t u = sign | 0x7F800000 | (frac << 13);
        float f; memcpy(&f, &u, 4); return f;
    }

    uint32_t u = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float f; memcpy(&f, &u, 4);
    return f;
}

uint16_t mi_f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);

    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

float mi_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

MiQuantStats mi_quant_analyze(const float *orig, const float *deq, int n) {
    MiQuantStats s = {0};
    float sig_power = 0.0f;
    float err_power = 0.0f;
    float max_err   = 0.0f;

    for (int i = 0; i < n; i++) {
        float err = fabsf(orig[i] - deq[i]);
        if (err > max_err) max_err = err;
        err_power += err * err;
        sig_power += orig[i] * orig[i];
    }

    s.mse = err_power / (float)n;
    s.max_err = max_err;
    s.snr_db = (err_power > 1e-20f)
             ? 10.0f * log10f(sig_power / err_power)
             : 999.0f;
    s.cosine_sim = mi_vec_cosine(orig, deq, n);
    return s;
}

void mi_quant_stats_print(const MiQuantStats *s, const char *label) {
    printf("[%s] MSE=%.6e  MaxErr=%.6e  SNR=%.1f dB  Cosine=%.6f\n",
           label, s->mse, s->max_err, s->snr_db, s->cosine_sim);
}
