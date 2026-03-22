
#ifndef MI_ARENA_H
#define MI_ARENA_H

#include "base.h"

typedef struct {
    uint8_t *buf;
    size_t   cap;
    size_t   used;
} MiArena;

MiArena  mi_arena_create(size_t capacity_bytes);
void     mi_arena_free(MiArena *a);
void    *mi_arena_alloc(MiArena *a, size_t bytes);
float   *mi_arena_alloc_f32(MiArena *a, int count);
void     mi_arena_reset(MiArena *a);
size_t   mi_arena_used(const MiArena *a);
size_t   mi_arena_remaining(const MiArena *a);

#endif
