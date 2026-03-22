/* ═══════════════════════════════════════════════════════════════════
 * quant.h — quantisation research
 *
 * Schemes
 *   INT8 absmax   — symmetric per-tensor/per-row
 *   INT8 zp       — asymmetric with zero-point
 *   INT4 group    — GPTQ-style group quantisation (32/64/128)
 *   Q4_0 / Q8_0   — GGML-style block quantisation
 *
 * Utilities
 *   FP16 / BF16 conversion
 *   Quantised matvec
 *   Error analysis (MSE, SNR, cosine sim)
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_QUANT_H
#define MI_QUANT_H

#include "tensor.h"

/* ════════════ Quantised storage types ════════════ */

typedef struct {
    int8_t *data;
    float   scale;
    int     n;
} MiQInt8;

typedef struct {
    int8_t  *data;
    float    scale;
    int8_t   zero_point;
    int      n;
} MiQInt8ZP;

typedef struct {
    uint8_t *data;        /* packed: two int4 per byte */
    float   *scales;      /* [n_groups] */
    float   *zeros;       /* [n_groups] or NULL (symmetric) */
    int      n;
    int      group_size;
    int      n_groups;
} MiQInt4;

#define MI_Q4_0_BLOCK 32
typedef struct {
    uint16_t scale;                          /* stored as fp16 */
    uint8_t  qs[MI_Q4_0_BLOCK / 2];         /* packed nibbles */
} MiBlockQ4_0;

#define MI_Q8_0_BLOCK 32
typedef struct {
    uint16_t scale;
    int8_t   qs[MI_Q8_0_BLOCK];
} MiBlockQ8_0;

/* ════════════ Quantise / Dequantise ════════════ */

MiQInt8   mi_quant_int8_absmax(const float *data, int n);
void      mi_dequant_int8_absmax(const MiQInt8 *q, float *out);
void      mi_quant_int8_free(MiQInt8 *q);

MiQInt8ZP mi_quant_int8_zp(const float *data, int n);
void      mi_dequant_int8_zp(const MiQInt8ZP *q, float *out);
void      mi_quant_int8_zp_free(MiQInt8ZP *q);

MiQInt4   mi_quant_int4_group(const float *data, int n, int group_size);
void      mi_dequant_int4_group(const MiQInt4 *q, float *out);
void      mi_quant_int4_free(MiQInt4 *q);

int       mi_quant_q4_0_nblocks(int n);
void      mi_quant_q4_0(const float *data, MiBlockQ4_0 *blocks, int n);
void      mi_dequant_q4_0(const MiBlockQ4_0 *blocks, float *out, int n);

int       mi_quant_q8_0_nblocks(int n);
void      mi_quant_q8_0(const float *data, MiBlockQ8_0 *blocks, int n);
void      mi_dequant_q8_0(const MiBlockQ8_0 *blocks, float *out, int n);

/* ════════════ Quantised matvec ════════════ */

/* W is stored row-major in int8, one scale per row.
 * x fp32 [cols], out fp32 [rows]. */
void mi_matvec_int8(const int8_t *W, const float *row_scales,
                    const float *x, float *out, int rows, int cols);

/* W stored as groups of int4 values per row. */
void mi_matvec_q4_0(const MiBlockQ4_0 *W, const float *x,
                    float *out, int rows, int cols);

/* ════════════ FP16 / BF16 helpers ════════════ */

uint16_t mi_f32_to_f16(float x);
float    mi_f16_to_f32(uint16_t h);
uint16_t mi_f32_to_bf16(float x);
float    mi_bf16_to_f32(uint16_t b);

/* ════════════ Error analysis ════════════ */

typedef struct {
    float mse;
    float max_err;
    float snr_db;
    float cosine_sim;
} MiQuantStats;

MiQuantStats mi_quant_analyze(const float *orig, const float *deq, int n);
void         mi_quant_stats_print(const MiQuantStats *s, const char *label);

#endif /* MI_QUANT_H */
