/* tensor.c — MiTensor lifecycle & utilities */
#include "mi/tensor.h"

MiTensor mi_tensor_create(int rows, int cols) {
    MiTensor t;
    t.rows  = rows;
    t.cols  = cols;
    t.data  = (float *)malloc((size_t)rows * cols * sizeof(float));
    t.owned = true;
    MI_CHECK_OOM(t.data);
    return t;
}

MiTensor mi_tensor_zeros(int rows, int cols) {
    MiTensor t;
    t.rows  = rows;
    t.cols  = cols;
    t.data  = (float *)calloc((size_t)rows * cols, sizeof(float));
    t.owned = true;
    MI_CHECK_OOM(t.data);
    return t;
}

MiTensor mi_tensor_view(float *data, int rows, int cols) {
    return (MiTensor){ .rows = rows, .cols = cols,
                       .data = data, .owned = false };
}

MiTensor mi_tensor_clone(const MiTensor *t) {
    MiTensor c = mi_tensor_create(t->rows, t->cols);
    memcpy(c.data, t->data, (size_t)t->rows * t->cols * sizeof(float));
    return c;
}

void mi_tensor_free(MiTensor *t) {
    if (t->owned && t->data) {
        free(t->data);
    }
    t->data  = NULL;
    t->rows  = 0;
    t->cols  = 0;
    t->owned = false;
}

void mi_tensor_fill(MiTensor *t, float val) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++) t->data[i] = val;
}

void mi_tensor_rand(MiTensor *t, MiRng *rng, float lo, float hi) {
    int n = t->rows * t->cols;
    float range = hi - lo;
    for (int i = 0; i < n; i++)
        t->data[i] = lo + mi_rng_float(rng) * range;
}

void mi_tensor_rand_normal(MiTensor *t, MiRng *rng, float mean, float std) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++)
        t->data[i] = mean + mi_rng_normal(rng) * std;
}

void mi_tensor_copy(const MiTensor *src, MiTensor *dst) {
    MI_ASSERT(src->rows == dst->rows && src->cols == dst->cols,
              "tensor shape mismatch: (%d,%d) vs (%d,%d)",
              src->rows, src->cols, dst->rows, dst->cols);
    memcpy(dst->data, src->data,
           (size_t)src->rows * src->cols * sizeof(float));
}

void mi_tensor_print(const MiTensor *t, const char *name) {
    printf("%s [%d × %d]:\n", name, t->rows, t->cols);
    for (int r = 0; r < t->rows; r++) {
        printf("  ");
        for (int c = 0; c < t->cols; c++)
            printf("%8.4f ", mi_tensor_get(t, r, c));
        printf("\n");
    }
}

void mi_print_vec(const float *x, int n, const char *name) {
    printf("%s [%d]: ", name, n);
    for (int i = 0; i < n; i++) printf("%.4f ", x[i]);
    printf("\n");
}
