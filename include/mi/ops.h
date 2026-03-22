
#ifndef MI_OPS_H
#define MI_OPS_H

#include "tensor.h"

float mi_dot(const float *a, const float *b, int n);
void  mi_vec_add(const float *a, const float *b, float *out, int n);
void  mi_vec_sub(const float *a, const float *b, float *out, int n);
void  mi_vec_mul(const float *a, const float *b, float *out, int n);
void  mi_vec_scale(const float *x, float s, float *out, int n);
void  mi_vec_add_scaled(const float *a, const float *b, float s,
                        float *out, int n);
void  mi_vec_copy(const float *src, float *dst, int n);
void  mi_vec_fill(float *x, int n, float val);
float mi_vec_max(const float *x, int n);
float mi_vec_min(const float *x, int n);
float mi_vec_sum(const float *x, int n);
float mi_vec_norm2(const float *x, int n);
float mi_vec_cosine(const float *a, const float *b, int n);

void mi_matvec(const MiTensor *W, const float *x, float *out);

void mi_relu(float *x, int n);
void mi_silu(float *x, int n);
void mi_gelu(float *x, int n);

void mi_rmsnorm(const float *x, const float *w, float *out,
                int n, float eps);
void mi_layernorm(const float *x, const float *gamma, const float *beta,
                  float *out, int n, float eps);

void  mi_softmax(float *x, int n);
void  mi_log_softmax(const float *x, float *out, int n);

int   mi_argmax(const float *x, int n);
int   mi_argmin(const float *x, int n);

void mi_swiglu_ffn(const MiTensor *W_gate, const MiTensor *W_up,
                   const MiTensor *W_down,
                   const float *x, float *out, float *scratch);

void mi_relu_ffn(const MiTensor *W1, const MiTensor *W2,
                 const float *x, float *out, float *scratch);

#endif
