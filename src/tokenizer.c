
#include "mi/tokenizer.h"

static uint64_t merge_key(int a, int b) {
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

static uint32_t merge_hash(uint64_t key, int cap) {
    key = (key ^ (key >> 33)) * 0xff51afd7ed558ccdULL;
    key = (key ^ (key >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)(key % (uint64_t)cap);
}

static void merge_map_insert(MiTokenizer *t, int a, int b,
                              int result, int rank) {
    uint64_t key = merge_key(a, b);
    uint32_t idx = merge_hash(key, t->merge_map_cap);
    while (t->merge_map[idx].rank >= 0) {
        idx = (idx + 1) % (uint32_t)t->merge_map_cap;
    }
    t->merge_map[idx].key    = key;
    t->merge_map[idx].result = result;
    t->merge_map[idx].rank   = rank;
}

static int merge_map_lookup(const MiTokenizer *t, int a, int b) {
    if (!t->merge_map) return -1;
    uint64_t key = merge_key(a, b);
    uint32_t idx = merge_hash(key, t->merge_map_cap);
    for (;;) {
        if (t->merge_map[idx].rank < 0) return -1;
        if (t->merge_map[idx].key == key) return t->merge_map[idx].rank;
        idx = (idx + 1) % (uint32_t)t->merge_map_cap;
    }
}

static int merge_map_result(const MiTokenizer *t, int a, int b) {
    if (!t->merge_map) return -1;
    uint64_t key = merge_key(a, b);
    uint32_t idx = merge_hash(key, t->merge_map_cap);
    for (;;) {
        if (t->merge_map[idx].rank < 0) return -1;
        if (t->merge_map[idx].key == key) return t->merge_map[idx].result;
        idx = (idx + 1) % (uint32_t)t->merge_map_cap;
    }
}

MiTokenizer mi_tokenizer_create(int vocab_size) {
    MiTokenizer t;
    memset(&t, 0, sizeof(t));
    t.vocab_size = vocab_size;
    t.bos_token  = 0;
    t.eos_token  = vocab_size - 1;
    t.is_bpe     = false;
    t.vocab_strs = (char **)calloc(vocab_size, sizeof(char *));
    t.scores     = (float *)calloc(vocab_size, sizeof(float));
    MI_CHECK_OOM(t.vocab_strs); MI_CHECK_OOM(t.scores);

    for (int i = 0; i < 256; i++) t.byte_to_token[i] = i % vocab_size;
    return t;
}

void mi_tokenizer_set(MiTokenizer *t, int id, const char *text, float score) {
    MI_ASSERT(id >= 0 && id < t->vocab_size, "token id %d out of range", id);
    if (t->vocab_strs) {
        free(t->vocab_strs[id]);
        t->vocab_strs[id] = strdup(text);
    }
    if (t->scores) t->scores[id] = score;
}

static int32_t read_i32(FILE *f) {
    int32_t v;
    if (fread(&v, 4, 1, f) != 1) return 0;
    return v;
}

MiTokenizer mi_tokenizer_load_bpe(const char *path) {
    MiTokenizer t;
    memset(&t, 0, sizeof(t));
    t.is_bpe = true;

    FILE *f = fopen(path, "rb");
    MI_ASSERT(f != NULL, "cannot open tokenizer: %s", path);


    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4D49544B) {
        MI_ASSERT(false, "bad tokenizer magic 0x%08X in %s", magic, path);
    }
    int version    = read_i32(f);
    MI_ASSERT(version == 1, "unsupported tokenizer version %d", version);
    t.vocab_size   = read_i32(f);
    t.bos_token    = read_i32(f);
    t.eos_token    = read_i32(f);
    t.n_merges     = read_i32(f);

    MI_INFO("loading tokenizer: vocab=%d  merges=%d  bos=%d  eos=%d",
            t.vocab_size, t.n_merges, t.bos_token, t.eos_token);


    for (int i = 0; i < 256; i++)
        t.byte_to_token[i] = read_i32(f);


    t.tokens     = (uint8_t **)calloc(t.vocab_size, sizeof(uint8_t *));
    t.token_lens = (int *)calloc(t.vocab_size, sizeof(int));
    MI_CHECK_OOM(t.tokens); MI_CHECK_OOM(t.token_lens);

    for (int i = 0; i < t.vocab_size; i++) {
        int len = read_i32(f);
        t.token_lens[i] = len;
        t.tokens[i] = (uint8_t *)malloc(len + 1);
        MI_CHECK_OOM(t.tokens[i]);
        if (len > 0) {
            if (fread(t.tokens[i], 1, len, f) != (size_t)len)
                MI_ASSERT(false, "truncated tokenizer at vocab entry %d", i);
        }
        t.tokens[i][len] = '\0';
    }


    t.merge_map_cap = t.n_merges * 3 + 1;
    t.merge_map = (MiMergeBucket *)malloc(
        (size_t)t.merge_map_cap * sizeof(MiMergeBucket));
    MI_CHECK_OOM(t.merge_map);
    for (int i = 0; i < t.merge_map_cap; i++)
        t.merge_map[i].rank = -1;

    for (int i = 0; i < t.n_merges; i++) {
        int a = read_i32(f);
        int b = read_i32(f);
        int r = read_i32(f);
        merge_map_insert(&t, a, b, r, i);
    }

    fclose(f);
    MI_INFO("tokenizer loaded");
    return t;
}

void mi_tokenizer_free(MiTokenizer *t) {
    if (t->tokens) {
        for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
        free(t->tokens);
    }
    free(t->token_lens);
    free(t->merge_map);
    if (t->vocab_strs) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab_strs[i]);
        free(t->vocab_strs);
    }
    free(t->scores);
    memset(t, 0, sizeof(*t));
}

int mi_tokenizer_encode(const MiTokenizer *t, const char *text,
                        int *out, int max_tokens) {
    if (!t->is_bpe) {

        int count = 0;
        const char *p = text;
        while (*p && count < max_tokens) {
            while (*p == ' ') p++;
            if (!*p) break;
            const char *start = p;
            while (*p && *p != ' ') p++;
            int len = (int)(p - start);
            int best = -1;
            if (t->vocab_strs) {
                for (int i = 0; i < t->vocab_size; i++) {
                    if (t->vocab_strs[i] &&
                        (int)strlen(t->vocab_strs[i]) == len &&
                        strncmp(t->vocab_strs[i], start, len) == 0) {
                        best = i; break;
                    }
                }
            }
            out[count++] = (best >= 0) ? best : 0;
        }
        return count;
    }


    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;


    int cap = text_len + 16;
    int *ids = (int *)malloc(cap * sizeof(int));
    MI_CHECK_OOM(ids);
    int n = 0;
    for (int i = 0; i < text_len; i++)
        ids[n++] = t->byte_to_token[(uint8_t)text[i]];


    while (n >= 2) {
        int best_rank = INT32_MAX;
        int best_pos  = -1;

        for (int i = 0; i < n - 1; i++) {
            int rank = merge_map_lookup(t, ids[i], ids[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos  = i;
            }
        }
        if (best_pos < 0) break;


        ids[best_pos] = merge_map_result(t, ids[best_pos], ids[best_pos + 1]);
        memmove(ids + best_pos + 1, ids + best_pos + 2,
                (n - best_pos - 2) * sizeof(int));
        n--;
    }

    int out_n = MI_MIN(n, max_tokens);
    memcpy(out, ids, out_n * sizeof(int));
    free(ids);
    return out_n;
}

char *mi_tokenizer_decode(const MiTokenizer *t,
                          const int *tokens, int n) {
    if (t->is_bpe && t->tokens) {

        size_t total = 0;
        for (int i = 0; i < n; i++) {
            int id = tokens[i];
            if (id >= 0 && id < t->vocab_size)
                total += t->token_lens[id];
        }
        char *buf = (char *)malloc(total + 1);
        MI_CHECK_OOM(buf);
        size_t pos = 0;
        for (int i = 0; i < n; i++) {
            int id = tokens[i];
            if (id >= 0 && id < t->vocab_size) {
                memcpy(buf + pos, t->tokens[id], t->token_lens[id]);
                pos += t->token_lens[id];
            }
        }
        buf[pos] = '\0';
        return buf;
    }


    size_t total = 1;
    for (int i = 0; i < n; i++)
        total += strlen(mi_tokenizer_token(t, tokens[i])) + 1;
    char *buf = (char *)malloc(total);
    MI_CHECK_OOM(buf);
    buf[0] = '\0';
    for (int i = 0; i < n; i++) {
        if (i > 0) strcat(buf, " ");
        strcat(buf, mi_tokenizer_token(t, tokens[i]));
    }
    return buf;
}

const char *mi_tokenizer_token(const MiTokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size) return "<unk>";
    if (t->is_bpe && t->tokens) {

        return (const char *)t->tokens[id];
    }
    if (t->vocab_strs && t->vocab_strs[id]) return t->vocab_strs[id];
    return "<unk>";
}
