
#include "mi.h"
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-40s", #name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_passed++; printf("✓\n"); } while(0)
#define FAIL(msg) do { printf("✗ (%s)\n", msg); } while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        FAIL("expected " #a " ≈ " #b); return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { FAIL(#a " != " #b); return; } \
} while(0)

static void test_tensor_basics(void) {
    TEST(tensor_create_and_access);
    MiTensor t = mi_tensor_zeros(3, 4);
    mi_tensor_set(&t, 1, 2, 5.0f);
    ASSERT_NEAR(mi_tensor_get(&t, 1, 2), 5.0f, 1e-6f);
    ASSERT_EQ(mi_tensor_numel(&t), 12);
    mi_tensor_free(&t);
    PASS();
}

static void test_tensor_clone(void) {
    TEST(tensor_clone);
    MiTensor a = mi_tensor_create(2, 3);
    mi_tensor_fill(&a, 7.0f);
    MiTensor b = mi_tensor_clone(&a);
    ASSERT_NEAR(mi_tensor_get(&b, 1, 1), 7.0f, 1e-6f);
    mi_tensor_set(&b, 0, 0, 99.0f);
    ASSERT_NEAR(mi_tensor_get(&a, 0, 0), 7.0f, 1e-6f);
    mi_tensor_free(&a); mi_tensor_free(&b);
    PASS();
}

static void test_dot_product(void) {
    TEST(dot_product);
    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};
    ASSERT_NEAR(mi_dot(a, b, 3), 32.0f, 1e-5f);
    PASS();
}

static void test_softmax(void) {
    TEST(softmax);
    float x[] = {1, 2, 3};
    mi_softmax(x, 3);
    ASSERT_NEAR(x[0] + x[1] + x[2], 1.0f, 1e-5f);
    assert(x[2] > x[1] && x[1] > x[0]);
    PASS();
}

static void test_rmsnorm(void) {
    TEST(rmsnorm);
    float x[] = {3.0f, 4.0f};
    float w[] = {1.0f, 1.0f};
    float out[2];
    mi_rmsnorm(x, w, out, 2, 1e-5f);


    ASSERT_NEAR(out[0], 3.0f / sqrtf(12.5f), 1e-4f);
    PASS();
}

static void test_matvec(void) {
    TEST(matvec);
    MiTensor W = mi_tensor_create(2, 3);
    float data[] = {1,0,0, 0,1,0};
    memcpy(W.data, data, sizeof(data));
    float x[] = {5, 6, 7};
    float out[2];
    mi_matvec(&W, x, out);
    ASSERT_NEAR(out[0], 5.0f, 1e-5f);
    ASSERT_NEAR(out[1], 6.0f, 1e-5f);
    mi_tensor_free(&W);
    PASS();
}

static void test_silu(void) {
    TEST(silu);
    float x[] = {0.0f, 1.0f, -1.0f};
    mi_silu(x, 3);
    ASSERT_NEAR(x[0], 0.0f, 1e-5f);
    ASSERT_NEAR(x[1], 1.0f / (1.0f + expf(-1.0f)), 1e-4f);
    PASS();
}

static void test_cache_dense(void) {
    TEST(cache_dense);
    MiCache c = mi_cache_dense(2, 1, 4, 16);
    float k[] = {1, 2, 3, 4};
    float v[] = {5, 6, 7, 8};
    mi_cache_append(&c, 0, k, v);
    mi_cache_append(&c, 1, k, v);
    ASSERT_EQ(mi_cache_size(&c), 1);
    int len;
    const float *K = mi_cache_keys(&c, 0, &len);
    ASSERT_EQ(len, 1);
    ASSERT_NEAR(K[0], 1.0f, 1e-6f);
    mi_cache_truncate(&c, 0);
    ASSERT_EQ(mi_cache_size(&c), 0);
    mi_cache_destroy(&c);
    PASS();
}

static void test_cache_sliding(void) {
    TEST(cache_sliding);
    MiCache c = mi_cache_sliding(1, 1, 2, 4);
    float k[2], v[2];
    for (int i = 0; i < 8; i++) {
        k[0] = (float)i; k[1] = (float)(i * 10);
        v[0] = (float)(-i); v[1] = 0;
        mi_cache_append(&c, 0, k, v);
    }

    ASSERT_EQ(mi_cache_size(&c), 4);
    int len;
    const float *K = mi_cache_keys(&c, 0, &len);
    ASSERT_EQ(len, 4);

    ASSERT_NEAR(K[0], 4.0f, 1e-5f);
    mi_cache_destroy(&c);
    PASS();
}

static void test_cache_paged(void) {
    TEST(cache_paged);
    MiCache c = mi_cache_paged(1, 1, 4, 4, 32);
    float k[4] = {0}, v[4] = {0};
    for (int i = 0; i < 10; i++) {
        k[0] = (float)i;
        mi_cache_append(&c, 0, k, v);
    }
    ASSERT_EQ(mi_cache_size(&c), 10);
    int len;
    const float *K = mi_cache_keys(&c, 0, &len);
    ASSERT_NEAR(K[0], 0.0f, 1e-6f);
    ASSERT_NEAR(K[9 * 4], 9.0f, 1e-6f);
    mi_cache_truncate(&c, 5);
    ASSERT_EQ(mi_cache_size(&c), 5);
    mi_cache_destroy(&c);
    PASS();
}

static void test_sampler_greedy(void) {
    TEST(sampler_greedy);
    MiSampler s = mi_sampler_greedy();
    MiRng rng = mi_rng_create(0);
    float logits[] = {1, 5, 3, 2};
    ASSERT_EQ(mi_sampler_sample(&s, logits, 4, &rng), 1);
    mi_sampler_destroy(&s);
    PASS();
}

static void test_sampler_topk(void) {
    TEST(sampler_topk);
    MiSampler s = mi_sampler_top_k(2, 1.5f);
    MiRng rng = mi_rng_create(42);
    int counts[4] = {0};
    for (int i = 0; i < 1000; i++) {
        float logits[] = {1, 5, 3, 2};
        int tok = mi_sampler_sample(&s, logits, 4, &rng);
        counts[tok]++;
    }

    assert(counts[1] > 300);
    assert(counts[0] + counts[3] < 50);
    mi_sampler_destroy(&s);
    PASS();
}

static void test_sampler_mirostat(void) {
    TEST(sampler_mirostat);
    MiSampler s = mi_sampler_mirostat_v2(5.0f, 0.1f);
    MiRng rng = mi_rng_create(42);
    for (int i = 0; i < 100; i++) {
        float logits[] = {2, 1, 0, -1, -2};
        int tok = mi_sampler_sample(&s, logits, 5, &rng);
        assert(tok >= 0 && tok < 5);
        mi_sampler_accept(&s, tok);
    }
    mi_sampler_destroy(&s);
    PASS();
}

static void test_quant_int8(void) {
    TEST(quant_int8_roundtrip);
    float data[] = {-1.0f, 0.0f, 0.5f, 1.0f};
    MiQInt8 q = mi_quant_int8_absmax(data, 4);
    float deq[4];
    mi_dequant_int8_absmax(&q, deq);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(data[i], deq[i], 0.02f);
    mi_quant_int8_free(&q);
    PASS();
}

static void test_quant_q4_0(void) {
    TEST(quant_q4_0_roundtrip);
    MiRng rng = mi_rng_create(99);
    float data[64];
    for (int i = 0; i < 64; i++) data[i] = mi_rng_normal(&rng) * 0.5f;
    int nb = mi_quant_q4_0_nblocks(64);
    MiBlockQ4_0 *blocks = (MiBlockQ4_0 *)malloc(nb * sizeof(MiBlockQ4_0));
    mi_quant_q4_0(data, blocks, 64);
    float deq[64];
    mi_dequant_q4_0(blocks, deq, 64);
    MiQuantStats s = mi_quant_analyze(data, deq, 64);
    assert(s.cosine_sim > 0.9f);
    free(blocks);
    PASS();
}

static void test_fp16_roundtrip(void) {
    TEST(fp16_roundtrip);
    float vals[] = {0.0f, 1.0f, -1.0f, 0.001f, 65504.0f};
    for (int i = 0; i < 5; i++) {
        float rt = mi_f16_to_f32(mi_f32_to_f16(vals[i]));
        ASSERT_NEAR(vals[i], rt, fabsf(vals[i]) * 0.01f + 1e-4f);
    }
    PASS();
}

static void test_rope_standard(void) {
    TEST(rope_standard);
    MiRoPE rope = mi_rope_standard(10000.0f);
    float vec[] = {1, 0, 1, 0};
    mi_rope_apply(&rope, vec, 0, 1, 4);

    ASSERT_NEAR(vec[0], 1.0f, 1e-5f);
    ASSERT_NEAR(vec[1], 0.0f, 1e-5f);
    mi_rope_destroy(&rope);
    PASS();
}

static void test_rope_alibi(void) {
    TEST(rope_alibi_bias);
    MiRoPE alibi = mi_rope_alibi(4);

    float b = mi_rope_bias(&alibi, 0, 5, 3);
    assert(b < 0);
    float b0 = mi_rope_bias(&alibi, 0, 5, 5);
    ASSERT_NEAR(b0, 0.0f, 1e-6f);
    mi_rope_destroy(&alibi);
    PASS();
}

static void test_sink_evict(void) {
    TEST(sink_evict);
    int kv_dim = 4;
    float K[20 * 4], V[20 * 4];
    for (int i = 0; i < 20 * 4; i++) { K[i] = (float)i; V[i] = (float)i; }
    int seq_len = 20;
    MiSinkConfig cfg = { .sink_size = 2, .window_size = 3 };
    mi_sink_evict(K, V, &seq_len, kv_dim, &cfg);
    ASSERT_EQ(seq_len, 5);

    ASSERT_NEAR(K[0], 0.0f, 1e-5f);
    ASSERT_NEAR(K[4], 4.0f, 1e-5f);

    ASSERT_NEAR(K[8], 17.0f * 4, 1e-5f);
    PASS();
}

static void test_vstore(void) {
    TEST(vector_store_search);
    MiVectorStore vs = mi_vstore_create(4, 100);
    float e1[] = {1, 0, 0, 0};
    float e2[] = {0, 1, 0, 0};
    float e3[] = {0.9f, 0.1f, 0, 0};
    mi_vstore_add(&vs, e1, 0);
    mi_vstore_add(&vs, e2, 1);
    mi_vstore_add(&vs, e3, 2);

    float query[] = {1, 0, 0, 0};
    int idx[2]; float scores[2];
    mi_vstore_search(&vs, query, 2, idx, scores);

    ASSERT_EQ(idx[0], 0);
    ASSERT_NEAR(scores[0], 1.0f, 1e-5f);
    mi_vstore_free(&vs);
    PASS();
}

static void test_model_forward(void) {
    TEST(model_forward_basic);
    MiModelConfig cfg = {
        .d_model = 16, .n_heads = 2, .n_kv_heads = 1,
        .d_head = 8, .d_ff = 32, .n_layers = 1,
        .vocab_size = 10, .max_seq_len = 32,
        .norm_eps = 1e-5f, .rope_theta = 10000.0f,
        .ffn_type = MI_FFN_RELU,
    };
    MiRng rng = mi_rng_create(42);
    MiModel m = mi_model_create(cfg);
    mi_model_init_random(&m, &rng);

    size_t scratch_sz = mi_model_scratch_size(&cfg);
    float *scratch = (float *)malloc(scratch_sz);
    float logits[10];
    mi_model_forward(&m, 0, logits, scratch);

    for (int i = 0; i < 10; i++)
        assert(isfinite(logits[i]));
    free(scratch);
    mi_model_free(&m);
    PASS();
}

static void test_model_save_load(void) {
    TEST(model_save_load);
    MiModelConfig cfg = {
        .d_model = 8, .n_heads = 2, .n_kv_heads = 2,
        .d_head = 4, .d_ff = 16, .n_layers = 1,
        .vocab_size = 8, .max_seq_len = 16,
        .norm_eps = 1e-5f, .rope_theta = 10000.0f,
        .ffn_type = MI_FFN_RELU,
    };
    MiRng rng = mi_rng_create(99);
    MiModel m = mi_model_create(cfg);
    mi_model_init_random(&m, &rng);

    MiStatus s = mi_model_save(&m, "/tmp/mi_test_model.bin");
    ASSERT_EQ(s, MI_OK);

    MiModel m2 = mi_model_create(cfg);
    s = mi_model_load(&m2, "/tmp/mi_test_model.bin");
    ASSERT_EQ(s, MI_OK);


    ASSERT_NEAR(mi_tensor_get(&m.w.tok_emb, 0, 0),
                mi_tensor_get(&m2.w.tok_emb, 0, 0), 1e-6f);
    ASSERT_NEAR(m.w.layers[0].Wq.data[0],
                m2.w.layers[0].Wq.data[0], 1e-6f);

    mi_model_free(&m);
    mi_model_free(&m2);
    PASS();
}

static void test_arena(void) {
    TEST(arena_alloc_reset);
    MiArena a = mi_arena_create(4096);
    float *p1 = mi_arena_alloc_f32(&a, 100);
    float *p2 = mi_arena_alloc_f32(&a, 200);
    assert(p1 != NULL && p2 != NULL);
    assert(p2 > p1);
    assert(mi_arena_used(&a) > 0);
    mi_arena_reset(&a);
    ASSERT_EQ((int)mi_arena_used(&a), 0);
    mi_arena_free(&a);
    PASS();
}

int main(void) {
    mi_log_level = MI_LOG_NONE;

    printf("╔═══════════════════════════════════════╗\n");
    printf("║   mini_infer test suite                ║\n");
    printf("╚═══════════════════════════════════════╝\n\n");


    test_tensor_basics();
    test_tensor_clone();


    test_dot_product();
    test_softmax();
    test_rmsnorm();
    test_matvec();
    test_silu();


    test_cache_dense();
    test_cache_sliding();
    test_cache_paged();


    test_sampler_greedy();
    test_sampler_topk();
    test_sampler_mirostat();


    test_quant_int8();
    test_quant_q4_0();
    test_fp16_roundtrip();


    test_rope_standard();
    test_rope_alibi();


    test_sink_evict();
    test_vstore();


    test_model_forward();
    test_model_save_load();


    test_arena();

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d/%d passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
