CC      = clang
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iinclude
LDFLAGS = -lm

# ── SIMD: auto-detect architecture ──
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    CFLAGS += -mavx2 -mfma
endif
# ARM64 / aarch64: NEON is enabled by default, no flags needed.
# Define MI_NO_SIMD to force scalar: make CFLAGS+="-DMI_NO_SIMD"

# ── Source files ──
SRCS = src/base.c src/tensor.c src/arena.c src/ops.c \
       src/cache.c src/attention.c src/sampling.c src/rope.c \
       src/quant.c src/speculative.c src/memory.c \
       src/model.c src/generate.c src/tokenizer.c

OBJS = $(SRCS:.c=.o)

# ── Static library ──
LIB = libmi.a

# ── Examples ──
EXAMPLES = examples/basic_generate \
           examples/cache_bench \
           examples/quant_compare \
           examples/sampling_explore \
           examples/rag_memory \
           examples/generate_real \
           examples/experiment_cache

# ── Tests ──
TESTS = tests/test_runner

# ── Default target ──
all: $(LIB) $(EXAMPLES) $(TESTS)

$(LIB): $(OBJS)
	ar rcs $@ $^

# Pattern rule for source objects
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Examples
examples/%: examples/%.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmi $(LDFLAGS) -o $@

# Tests
tests/%: tests/%.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmi $(LDFLAGS) -o $@

# ── Convenience targets ──
test: $(TESTS)
	@echo "═══ Running tests ═══"
	@./tests/test_runner

run-examples: $(EXAMPLES)
	@echo "\n═══ basic_generate ═══" && ./examples/basic_generate
	@echo "\n═══ cache_bench ═══"    && ./examples/cache_bench
	@echo "\n═══ quant_compare ═══"  && ./examples/quant_compare
	@echo "\n═══ sampling_explore ═══" && ./examples/sampling_explore
	@echo "\n═══ rag_memory ═══"     && ./examples/rag_memory

clean:
	rm -f $(OBJS) $(LIB) $(EXAMPLES) $(TESTS)

# Debug build
debug: CFLAGS += -g -O0 -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: all

.PHONY: all test run-examples clean debug
