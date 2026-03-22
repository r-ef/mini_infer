
#include "mi/arena.h"

MiArena mi_arena_create(size_t cap) {
    MiArena a;
    a.buf  = (uint8_t *)malloc(cap);
    MI_CHECK_OOM(a.buf);
    a.cap  = cap;
    a.used = 0;
    return a;
}

void mi_arena_free(MiArena *a) {
    free(a->buf);
    a->buf  = NULL;
    a->cap  = 0;
    a->used = 0;
}

void *mi_arena_alloc(MiArena *a, size_t bytes) {

    size_t aligned = MI_ALIGN_UP(bytes, 16);
    MI_ASSERT(a->used + aligned <= a->cap,
              "arena OOM: need %zu, have %zu free",
              aligned, a->cap - a->used);
    void *ptr = a->buf + a->used;
    a->used += aligned;
    return ptr;
}

float *mi_arena_alloc_f32(MiArena *a, int n) {
    return (float *)mi_arena_alloc(a, (size_t)n * sizeof(float));
}

void mi_arena_reset(MiArena *a) {
    a->used = 0;
}

size_t mi_arena_used(const MiArena *a) { return a->used; }
size_t mi_arena_remaining(const MiArena *a) { return a->cap - a->used; }
