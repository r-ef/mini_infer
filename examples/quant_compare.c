
#include "mi.h"

int main(void) {
    mi_log_level = MI_LOG_WARN;

    int rows = 64, cols = 128;
    int n = rows * cols;

    MiRng rng = mi_rng_create(1234);
    MiTensor W = mi_tensor_create(rows, cols);
    mi_tensor_rand_normal(&W, &rng, 0.0f, 0.5f);

    float *deq = (float *)malloc(n * sizeof(float));
    MI_CHECK_OOM(deq);

    printf("═══════════════════════════════════════════════\n");
    printf("  Quantisation Comparison  [%d × %d = %d params]\n", rows, cols, n);
    printf("═══════════════════════════════════════════════\n\n");


    {
        MiQInt8 q = mi_quant_int8_absmax(W.data, n);
        mi_dequant_int8_absmax(&q, deq);
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "INT8 absmax  ");
        mi_quant_int8_free(&q);
    }


    {
        MiQInt8ZP q = mi_quant_int8_zp(W.data, n);
        mi_dequant_int8_zp(&q, deq);
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "INT8 zp      ");
        mi_quant_int8_zp_free(&q);
    }


    {
        MiQInt4 q = mi_quant_int4_group(W.data, n, 32);
        mi_dequant_int4_group(&q, deq);
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "INT4 g32     ");
        mi_quant_int4_free(&q);
    }


    {
        MiQInt4 q = mi_quant_int4_group(W.data, n, 128);
        mi_dequant_int4_group(&q, deq);
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "INT4 g128    ");
        mi_quant_int4_free(&q);
    }


    {
        int nb = mi_quant_q4_0_nblocks(n);
        MiBlockQ4_0 *blocks = (MiBlockQ4_0 *)malloc(nb * sizeof(MiBlockQ4_0));
        mi_quant_q4_0(W.data, blocks, n);
        mi_dequant_q4_0(blocks, deq, n);
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "Q4_0 block   ");
        free(blocks);
    }


    {
        int nb = mi_quant_q8_0_nblocks(n);
        MiBlockQ8_0 *blocks = (MiBlockQ8_0 *)malloc(nb * sizeof(MiBlockQ8_0));
        mi_quant_q8_0(W.data, blocks, n);
        mi_dequant_q8_0(blocks, deq, n);
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "Q8_0 block   ");
        free(blocks);
    }


    {
        for (int i = 0; i < n; i++)
            deq[i] = mi_f16_to_f32(mi_f32_to_f16(W.data[i]));
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "FP16 round   ");
    }


    {
        for (int i = 0; i < n; i++)
            deq[i] = mi_bf16_to_f32(mi_f32_to_bf16(W.data[i]));
        MiQuantStats s = mi_quant_analyze(W.data, deq, n);
        mi_quant_stats_print(&s, "BF16 round   ");
    }

    printf("\n");
    free(deq);
    mi_tensor_free(&W);
    return 0;
}
