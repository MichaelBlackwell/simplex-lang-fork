// Standalone minimal runtime for bootstrap_mini
// Self-contained with no external dependencies

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <execinfo.h>
#include <sys/time.h>

// String representation matching simplex runtime
typedef struct {
    size_t len;
    size_t cap;
    char* data;
} SxString;

// Vector representation
typedef struct {
    void** items;
    size_t len;
    size_t cap;
} SxVec;

// Entry point wrapper
extern int64_t simplex_main(void);

static int program_argc;
static char** program_argv;

int main(int argc, char** argv) {
    program_argc = argc;
    program_argv = argv;
    return (int)simplex_main();
}

// Memory access helpers (work with 8-byte slots)
void* store_ptr(void* base, int64_t slot, void* value) {
    ((void**)base)[slot] = value;
    return base;
}

static int store_i64_count = 0;
void* store_i64(void* base, int64_t slot, int64_t value) {
    store_i64_count++;
    ((int64_t*)base)[slot] = value;
    return base;
}

void* load_ptr(void* base, int64_t slot) {
    return ((void**)base)[slot];
}

int64_t load_i64(void* base, int64_t slot) {
    return ((int64_t*)base)[slot];
}

// Debug trace function (can be called from LLVM IR if needed)
void debug_trace_i64(int64_t marker, int64_t value) {
    // No-op in release mode
}

// String functions
SxString* intrinsic_string_new(const char* data) {
    SxString* s = malloc(sizeof(SxString));
    if (!data) {
        s->len = 0;
        s->cap = 1;
        s->data = malloc(1);
        s->data[0] = '\0';
    } else {
        s->len = strlen(data);
        s->cap = s->len + 1;
        s->data = malloc(s->cap);
        memcpy(s->data, data, s->cap);
    }
    return s;
}

SxString* intrinsic_string_from_char(int64_t c) {
    SxString* s = malloc(sizeof(SxString));
    s->len = 1;
    s->cap = 2;
    s->data = malloc(2);
    s->data[0] = (char)c;
    s->data[1] = '\0';
    return s;
}

SxString* intrinsic_string_concat(SxString* a, SxString* b) {
    if (!a || !b) return intrinsic_string_new("");
    SxString* s = malloc(sizeof(SxString));
    s->len = a->len + b->len;
    s->cap = s->len + 1;
    s->data = malloc(s->cap);
    memcpy(s->data, a->data, a->len);
    memcpy(s->data + a->len, b->data, b->len);
    s->data[s->len] = '\0';
    return s;
}

int64_t intrinsic_string_len(SxString* str) {
    return str ? str->len : 0;
}

int8_t intrinsic_string_eq(SxString* a, SxString* b) {
    if (!a || !b) return a == b;
    if (a->len != b->len) return 0;
    return memcmp(a->data, b->data, a->len) == 0;
}

int64_t intrinsic_string_char_at(SxString* str, int64_t index) {
    if (!str || index < 0 || (size_t)index >= str->len) return -1;
    return (unsigned char)str->data[index];
}

SxString* intrinsic_string_slice(SxString* str, int64_t start, int64_t end) {
    if (!str || !str->data) return intrinsic_string_new("");
    if (start < 0) start = 0;
    if (end > (int64_t)str->len) end = str->len;
    if (start >= end) return intrinsic_string_new("");

    size_t len = end - start;
    SxString* result = malloc(sizeof(SxString));
    result->len = len;
    result->cap = len + 1;
    result->data = malloc(len + 1);
    memcpy(result->data, str->data + start, len);
    result->data[len] = '\0';
    return result;
}

int64_t intrinsic_string_to_int(SxString* str) {
    if (!str || !str->data) return 0;
    return strtoll(str->data, NULL, 10);
}

SxString* intrinsic_int_to_string(int64_t value) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long)value);
    return intrinsic_string_new(buf);
}

// Vector functions
SxVec* intrinsic_vec_new(void) {
    SxVec* v = malloc(sizeof(SxVec));
    v->items = NULL;
    v->len = 0;
    v->cap = 0;
    return v;
}

void intrinsic_vec_push(SxVec* vec, void* item) {
    if (!vec) return;
    if (vec->len >= vec->cap) {
        size_t new_cap = vec->cap == 0 ? 8 : vec->cap * 2;
        vec->items = realloc(vec->items, new_cap * sizeof(void*));
        vec->cap = new_cap;
    }
    vec->items[vec->len++] = item;
}

void* intrinsic_vec_get(SxVec* vec, int64_t index) {
    if (!vec || index < 0 || (size_t)index >= vec->len) {
        return NULL;
    }
    return vec->items[index];
}

int64_t intrinsic_vec_len(SxVec* vec) {
    return vec ? vec->len : 0;
}

// IO functions
void intrinsic_println(SxString* str) {
    if (str && str->data) {
        printf("%s\n", str->data);
    } else {
        printf("\n");
    }
}

SxString* intrinsic_read_file(SxString* path) {
    if (!path || !path->data) return NULL;

    FILE* f = fopen(path->data, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    SxString* result = malloc(sizeof(SxString));
    result->len = size;
    result->cap = size + 1;
    result->data = malloc(size + 1);
    fread(result->data, 1, size, f);
    result->data[size] = '\0';
    fclose(f);

    return result;
}

void intrinsic_write_file(SxString* path, SxString* content) {
    if (!path || !path->data || !content || !content->data) return;

    FILE* f = fopen(path->data, "wb");
    if (!f) return;

    fwrite(content->data, 1, content->len, f);
    fclose(f);
}

SxVec* intrinsic_get_args(void) {
    SxVec* args = intrinsic_vec_new();
    for (int i = 0; i < program_argc; i++) {
        intrinsic_vec_push(args, intrinsic_string_new(program_argv[i]));
    }
    return args;
}

// AI intrinsics (mock implementation for bootstrap)
SxString* intrinsic_ai_infer(SxString* model, SxString* prompt, int64_t temperature) {
    // Mock implementation: return prompt with "[MOCK AI RESPONSE]" prefix
    char buf[256];
    const char* model_name = model && model->data ? model->data : "default";
    const char* prompt_text = prompt && prompt->data ? prompt->data : "";
    snprintf(buf, sizeof(buf), "[AI:%s] Response to: %.100s", model_name, prompt_text);
    return intrinsic_string_new(buf);
}

double* intrinsic_ai_embed(SxString* text) {
    // Mock implementation: return zero vector
    double* vec = malloc(1536 * sizeof(double));
    for (int i = 0; i < 1536; i++) {
        vec[i] = 0.0;
    }
    // Set first element based on text length for basic similarity testing
    if (text && text->data) {
        vec[0] = (double)text->len / 100.0;
    }
    return vec;
}

// Timing intrinsics for performance measurement
int64_t intrinsic_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)tv.tv_usec / 1000;
}

int64_t intrinsic_get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + (int64_t)tv.tv_usec;
}

// Arena allocator for efficient memory management
typedef struct {
    char* base;
    int64_t capacity;
    int64_t offset;
} Arena;

Arena* intrinsic_arena_create(int64_t capacity) {
    Arena* a = malloc(sizeof(Arena));
    a->base = malloc(capacity);
    a->capacity = capacity;
    a->offset = 0;
    return a;
}

void* intrinsic_arena_alloc(Arena* a, int64_t size) {
    if (!a || a->offset + size > a->capacity) return NULL;
    void* ptr = a->base + a->offset;
    a->offset += size;
    // Align to 8 bytes
    a->offset = (a->offset + 7) & ~7;
    return ptr;
}

void intrinsic_arena_reset(Arena* a) {
    if (a) a->offset = 0;
}

void intrinsic_arena_free(Arena* a) {
    if (a) {
        free(a->base);
        free(a);
    }
}

int64_t intrinsic_arena_used(Arena* a) {
    return a ? a->offset : 0;
}
