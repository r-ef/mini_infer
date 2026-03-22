
#ifndef MI_TOKENIZER_H
#define MI_TOKENIZER_H

#include "base.h"

typedef struct {
    uint64_t key;
    int      result;
    int      rank;
} MiMergeBucket;

typedef struct {

    uint8_t **tokens;
    int      *token_lens;
    int       vocab_size;
    int       bos_token;
    int       eos_token;


    int       byte_to_token[256];


    int       n_merges;


    MiMergeBucket *merge_map;
    int            merge_map_cap;


    char    **vocab_strs;
    float    *scores;
    bool      is_bpe;
} MiTokenizer;

MiTokenizer mi_tokenizer_create(int vocab_size);
void mi_tokenizer_set(MiTokenizer *t, int id, const char *text, float score);

MiTokenizer mi_tokenizer_load_bpe(const char *path);

void mi_tokenizer_free(MiTokenizer *t);

int  mi_tokenizer_encode(const MiTokenizer *t, const char *text,
                         int *out, int max_tokens);

char *mi_tokenizer_decode(const MiTokenizer *t,
                          const int *tokens, int n);

const char *mi_tokenizer_token(const MiTokenizer *t, int id);

#endif
