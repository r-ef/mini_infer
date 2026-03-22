
#ifndef MI_BASE_H
#define MI_BASE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

typedef enum {
    MI_OK = 0,
    MI_ERR_OOM,
    MI_ERR_INVALID,
    MI_ERR_OVERFLOW,
    MI_ERR_IO,
    MI_ERR_NOT_IMPL,
} MiStatus;

typedef enum {
    MI_LOG_TRACE = 0,
    MI_LOG_DEBUG = 1,
    MI_LOG_INFO  = 2,
    MI_LOG_WARN  = 3,
    MI_LOG_ERROR = 4,
    MI_LOG_NONE  = 5,
} MiLogLevel;

extern MiLogLevel mi_log_level;

#define MI_LOG(level, fmt, ...) do { \
    if ((level) >= mi_log_level) { \
        static const char *_names[] = {"TRACE","DEBUG","INFO ","WARN ","ERROR"}; \
        fprintf(stderr, "[mini_infer %s] " fmt "\n", _names[(level)], ##__VA_ARGS__); \
    } \
} while(0)

#define MI_TRACE(fmt, ...) MI_LOG(MI_LOG_TRACE, fmt, ##__VA_ARGS__)
#define MI_DEBUG(fmt, ...) MI_LOG(MI_LOG_DEBUG, fmt, ##__VA_ARGS__)
#define MI_INFO(fmt, ...)  MI_LOG(MI_LOG_INFO,  fmt, ##__VA_ARGS__)
#define MI_WARN(fmt, ...)  MI_LOG(MI_LOG_WARN,  fmt, ##__VA_ARGS__)
#define MI_ERROR(fmt, ...) MI_LOG(MI_LOG_ERROR, fmt, ##__VA_ARGS__)

#define MI_ASSERT(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "[ASSERT FAIL %s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        abort(); \
    } \
} while(0)

#define MI_CHECK_OOM(ptr) MI_ASSERT((ptr) != NULL, "out of memory")

#define MI_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MI_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MI_CLAMP(x, lo, hi) MI_MIN(MI_MAX((x), (lo)), (hi))
#define MI_ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define MI_ARRAY_LEN(arr) ((int)(sizeof(arr) / sizeof((arr)[0])))
#define MI_UNUSED(x) (void)(x)

typedef struct {
    struct timespec start;
    struct timespec end;
} MiTimer;

static inline void mi_timer_start(MiTimer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static inline double mi_timer_elapsed_s(MiTimer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    double s  = (double)(t->end.tv_sec  - t->start.tv_sec);
    double ns = (double)(t->end.tv_nsec - t->start.tv_nsec);
    return s + ns * 1e-9;
}

typedef struct {
    uint64_t s[4];
} MiRng;

MiRng   mi_rng_create(uint64_t seed);
uint64_t mi_rng_next(MiRng *rng);
float   mi_rng_float(MiRng *rng);
float   mi_rng_normal(MiRng *rng);
int     mi_rng_int(MiRng *rng, int lo, int hi);

#endif
