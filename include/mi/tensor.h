
#ifndef MI_TENSOR_H
#define MI_TENSOR_H

#include "base.h"

typedef struct {
    int   rows;
    int   cols;
    float *data;
    bool  owned;
} MiTensor;

MiTensor mi_tensor_create(int rows, int cols);
MiTensor mi_tensor_zeros(int rows, int cols);
MiTensor mi_tensor_view(float *data, int rows, int cols);
MiTensor mi_tensor_clone(const MiTensor *t);
void     mi_tensor_free(MiTensor *t);

static inline float mi_tensor_get(const MiTensor *t, int r, int c) {
    return t->data[r * t->cols + c];
}
static inline void mi_tensor_set(MiTensor *t, int r, int c, float v) {
    t->data[r * t->cols + c] = v;
}
static inline float *mi_tensor_row(const MiTensor *t, int r) {
    return t->data + r * t->cols;
}
static inline int mi_tensor_numel(const MiTensor *t) {
    return t->rows * t->cols;
}

static inline MiTensor mi_tensor_row_view(const MiTensor *t, int r) {
    return (MiTensor){ .rows = 1, .cols = t->cols,
                       .data = t->data + r * t->cols, .owned = false };
}
static inline MiTensor mi_tensor_slice_rows(const MiTensor *t,
                                             int start, int n) {
    return (MiTensor){ .rows = n, .cols = t->cols,
                       .data = t->data + start * t->cols, .owned = false };
}

void mi_tensor_fill(MiTensor *t, float val);
void mi_tensor_rand(MiTensor *t, MiRng *rng, float lo, float hi);
void mi_tensor_rand_normal(MiTensor *t, MiRng *rng, float mean, float std);
void mi_tensor_copy(const MiTensor *src, MiTensor *dst);
void mi_tensor_print(const MiTensor *t, const char *name);

void mi_print_vec(const float *x, int n, const char *name);

#endif
