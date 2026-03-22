/* ═══════════════════════════════════════════════════════════════════
 * tokenizer.h — byte-level BPE tokeniser
 *
 * Two modes:
 *   1. Toy:  mi_tokenizer_create + mi_tokenizer_set  (word lookup)
 *   2. Real: mi_tokenizer_load_bpe("tokenizer.bin")  (full BPE)
 *
 * Binary format written by tools/convert_hf.py:
 *   "MITK"  magic
 *   int32   version (1)
 *   int32   vocab_size
 *   int32   bos_token
 *   int32   eos_token
 *   int32   n_merges
 *   int32[256]  byte_to_token mapping
 *   for each vocab entry:
 *     int32   byte_len
 *     bytes   raw token bytes
 *   for each merge:
 *     int32   pair_a, pair_b, result
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef MI_TOKENIZER_H
#define MI_TOKENIZER_H

#include "base.h"

/* Merge hash-map bucket */
typedef struct {
    uint64_t key;       /* (pair_a << 32) | pair_b */
    int      result;    /* merged token id */
    int      rank;      /* merge priority (lower = higher priority) */
} MiMergeBucket;

typedef struct {
    /* Vocabulary: raw byte sequences for each token */
    uint8_t **tokens;       /* [vocab_size] */
    int      *token_lens;   /* [vocab_size] */
    int       vocab_size;
    int       bos_token;
    int       eos_token;

    /* Byte-level: each byte value → its initial token id */
    int       byte_to_token[256];

    /* BPE merge table */
    int       n_merges;

    /* Merge hash map (open addressing, linear probe) */
    MiMergeBucket *merge_map;
    int            merge_map_cap;

    /* Legacy fields for toy tokenizer (non-BPE) mode */
    char    **vocab_strs;   /* [vocab_size] or NULL */
    float    *scores;       /* [vocab_size] or NULL */
    bool      is_bpe;
} MiTokenizer;

/* ── Toy mode (word-level, for examples/tests) ── */
MiTokenizer mi_tokenizer_create(int vocab_size);
void mi_tokenizer_set(MiTokenizer *t, int id, const char *text, float score);

/* ── BPE mode (load from binary) ── */
MiTokenizer mi_tokenizer_load_bpe(const char *path);

/* ── Common API ── */
void mi_tokenizer_free(MiTokenizer *t);

/* Encode text to token IDs. Returns number of tokens. */
int  mi_tokenizer_encode(const MiTokenizer *t, const char *text,
                         int *out, int max_tokens);

/* Decode token IDs to text. Caller must free() the result. */
char *mi_tokenizer_decode(const MiTokenizer *t,
                          const int *tokens, int n);

/* Get the string for a single token (for display). */
const char *mi_tokenizer_token(const MiTokenizer *t, int id);

#endif /* MI_TOKENIZER_H */
