/* base.c — globals, RNG (xoshiro256**) */
#include "mi/base.h"

MiLogLevel mi_log_level = MI_LOG_INFO;

/* ── xoshiro256** ── */

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* SplitMix64 to seed the state from a single u64 */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

MiRng mi_rng_create(uint64_t seed) {
    MiRng rng;
    uint64_t sm = seed;
    rng.s[0] = splitmix64(&sm);
    rng.s[1] = splitmix64(&sm);
    rng.s[2] = splitmix64(&sm);
    rng.s[3] = splitmix64(&sm);
    return rng;
}

uint64_t mi_rng_next(MiRng *rng) {
    uint64_t *s = rng->s;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

float mi_rng_float(MiRng *rng) {
    /* Use upper 24 bits for a float in [0, 1) */
    return (float)(mi_rng_next(rng) >> 40) / (float)(1ULL << 24);
}

float mi_rng_normal(MiRng *rng) {
    /* Box-Muller */
    float u1 = mi_rng_float(rng);
    float u2 = mi_rng_float(rng);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

int mi_rng_int(MiRng *rng, int lo, int hi) {
    if (lo >= hi) return lo;
    uint64_t range = (uint64_t)(hi - lo);
    return lo + (int)(mi_rng_next(rng) % range);
}
