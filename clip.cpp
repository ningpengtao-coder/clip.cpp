#include <cassert>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "clip.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#if defined(GGML_USE_CPU)
#include "ggml-cpu.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// #define CLIP_DEBUG

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

//
// key constants
//

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_HAS_TEXT_ENC "clip.has_text_encoder"
#define KEY_HAS_VIS_ENC "clip.has_vision_encoder"
#define KEY_USE_GELU "clip.use_gelu"
#define KEY_N_EMBD "clip.%s.embedding_length"
#define KEY_N_FF "clip.%s.feed_forward_length"
#define KEY_N_BLOCK "clip.%s.block_count"
#define KEY_N_HEAD "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_PROJ_DIM "clip.%s.projection_dim"
#define KEY_TOKENS "tokenizer.ggml.tokens"
#define KEY_N_POSITIONS "clip.text.context_length"
#define KEY_IMAGE_SIZE "clip.vision.image_size"
#define KEY_PATCH_SIZE "clip.vision.patch_size"
#define KEY_IMAGE_MEAN "clip.vision.image_mean"
#define KEY_IMAGE_STD "clip.vision.image_std"
#define KEY_DO_CENTER_CROP "clip.vision.do_center_crop"
#define KEY_TOKENIZER "clip.tokenizer"
#define KEY_TEXT_ARCH "clip.text.arch"
#define KEY_TEXT_POOLING "clip.text.pooling"
#define KEY_VISION_POOLING "clip.vision.pooling"
#define KEY_TEXT_USE_GELU "clip.text.use_gelu"
#define KEY_VISION_USE_GELU "clip.vision.use_gelu"
#define KEY_TEXT_PAD_LENGTH "clip.text.pad_length"

//
// tensor name constants
//

#define TN_TOKEN_EMBD "%s.token_embd.weight"
#define TN_TOKEN_TYPE_EMBD "t.token_type_embd.weight"
#define TN_POS_EMBD "%s.position_embd.weight"
#define TN_CLASS_EMBD "v.class_embd"
#define TN_PATCH_EMBD "v.patch_embd.weight"
#define TN_ATTN_K "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT "%s.blk.%d.attn_out.%s"
#define TN_FFN_DOWN "%s.blk.%d.ffn_down.%s"
#define TN_FFN_UP "%s.blk.%d.ffn_up.%s"
#define TN_LN_1 "%s.blk.%d.ln1.%s"
#define TN_LN_2 "%s.blk.%d.ln2.%s"
#define TN_LN_PRE "%s.pre_ln.%s"
#define TN_LN_POST "%s.post_ln.%s"
#define TN_TEXT_PROJ "text_projection.weight"
#define TN_VIS_PROJ "visual_projection.weight"

//
// utilities to get data from a gguf file
//

int get_key_idx(const gguf_context * ctx, const char * key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        fprintf(stderr, "key %s not found in file\n", key);
        throw std::runtime_error(format("Missing required key: %s", key));
    }

    return i;
}

const uint32_t get_u32(const gguf_context * ctx, std::string key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_u32(ctx, i);
}

const float get_f32(const gguf_context * ctx, std::string key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

// Optional metadata used by non-OpenAI CLIP variants (e.g. Chinese-CLIP). Files
// produced by the original converter do not carry these keys.
const char * get_str_optional(const gguf_context * ctx, const char * key, const char * fallback) {
    const int i = gguf_find_key(ctx, key);
    if (i == -1) {
        return fallback;
    }
    return gguf_get_val_str(ctx, i);
}

bool get_bool_optional(const gguf_context * ctx, const char * key, bool fallback) {
    const int i = gguf_find_key(ctx, key);
    if (i == -1) {
        return fallback;
    }
    return gguf_get_val_bool(ctx, i);
}

uint32_t get_u32_optional(const gguf_context * ctx, const char * key, uint32_t fallback) {
    const int i = gguf_find_key(ctx, key);
    if (i == -1) {
        return fallback;
    }
    return gguf_get_val_u32(ctx, i);
}

struct ggml_tensor * get_tensor_optional(struct ggml_context * ctx, std::string name) {
    return ggml_get_tensor(ctx, name.c_str());
}

struct ggml_tensor * get_tensor(struct ggml_context * ctx, std::string name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        printf("unable to find tensor %s\n", name.c_str());
        throw std::runtime_error(format("unable to find tensor %s\n", name.c_str()));
    }

    return cur;
}

std::string get_ftype(int ftype) {
    switch (ftype) {
    case 0:
        return "f32";
        break;
    case 1:
        return "f16";
        break;
    case 2:
        return "q4_0";
        break;
    case 3:
        return "q4_1";
        break;
    case 6:
        return "q5_0";
        break;
    case 7:
        return "q5_1";
        break;
    case 8:
        return "q8_0";
        break;
    default:
        throw std::runtime_error(format("Unrecognized file type: %d\n", ftype));
    }
}

//
// Vocab utils
//

struct clip_vocab {
    using id = clip_vocab_id;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    //    void add_special_token(const std::string & token);
};

//
// clip layers
//

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_text_model {
    struct clip_text_hparams hparams;

    // embeddings
    struct ggml_tensor * token_embeddings;
    struct ggml_tensor * position_embeddings;
    // BERT-style text towers add a token-type embedding and an embedding
    // layer norm. They are optional for the original CLIP text tower.
    struct ggml_tensor * token_type_embeddings = nullptr;

    std::vector<clip_layer> layers;

    // Optional embedding layer norm (BERT / Chinese-CLIP).
    struct ggml_tensor * pre_ln_w = nullptr;
    struct ggml_tensor * pre_ln_b = nullptr;

    // Optional final layer norm. The original CLIP text tower always has it,
    // BERT-style towers do not.
    struct ggml_tensor * post_ln_w = nullptr;
    struct ggml_tensor * post_ln_b = nullptr;

    struct ggml_tensor * projection;
};

struct clip_vision_model {
    struct clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct clip_buffer {
    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~clip_buffer() { delete[] data; }
};

struct clip_ctx {
    bool has_text_encoder = false;
    bool has_vision_encoder = false;
    struct clip_text_model text_model;
    struct clip_vision_model vision_model;
    struct clip_vocab vocab;
    float image_mean[3];
    float image_std[3];
    // 预处理是否走“短边等比缩放 + 中心裁剪”。Chinese-CLIP 的 ONNX/PIL 预处理
    // 是直接 resize 到 image_size×image_size（do_center_crop=false）。
    bool vision_do_center_crop = true;
    bool use_gelu = false;
    bool text_use_gelu = false;
    bool use_wordpiece = false;
    bool text_bert = false;
    bool text_pool_cls = false;
    int32_t text_pad_length = 0;
    int32_t ftype = 1;
    struct ggml_context * ctx = nullptr;      /* GGUF 张量元信息（no_alloc） */
    struct gguf_context * ctx_gguf = nullptr;
    /* 计算后端：Vulkan（可选）+ CPU 回退，统一由 sched 调度。 */
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name = "CPU";
};

//
// memory allocation and management
//

// 后端选择与图执行（新版 ggml：backend + scheduler）
// ---------------------------------------------------------------------------

/* 后端偏好：auto（默认，优先 GPU）/ cpu / vulkan / cuda / hip（ROCm）。 */
enum class clip_backend_preference : int {
    auto_select = 0,
    cpu         = 1,
    vulkan      = 2,
    cuda        = 3,
    hip         = 4,
};

static std::atomic<int> g_clip_backend_preference{static_cast<int>(clip_backend_preference::auto_select)};

/* 单次 encode 的图节点上限；CLIP 计算图远小于该值。 */
static constexpr size_t kClipSchedGraphSize = 4096;

/* 图构建上下文只放元数据，张量由 sched 分配。 */
static constexpr size_t kClipGraphCtxSize = 32u * 1024u * 1024u;

/* 挑选可用的 GPU 后端：按设备类型查找，因为 ggml 的 GPU 设备名是显卡型号
 * （如 Adreno 840），不是后端名（Vulkan）。移动端 GPU 通常标记为 IGPU。
 * reg_filter 非空时只接受后端注册名包含该字符串的设备（"CUDA" / "Vulkan" /
 * "HIP"），用于同机多后端共存时精确选择。 */
static ggml_backend_t clip_init_gpu_backend(const char * reg_filter = nullptr) {
    /* 注意：ggml 同时定义了同名函数 ggml_backend_dev_type()，因此类型必须写全
     * `enum ggml_backend_dev_type`，否则 MSVC 会把裸名解析成函数。 */
    const enum ggml_backend_dev_type types[] = {
        GGML_BACKEND_DEVICE_TYPE_GPU,
        GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
    for (enum ggml_backend_dev_type type : types) {
        const int n_devices = (int) ggml_backend_dev_count();
        for (int i = 0; i < n_devices; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get((size_t) i);
            if (dev == nullptr || ggml_backend_dev_type(dev) != type) {
                continue;
            }
            if (reg_filter != nullptr) {
                ggml_backend_reg_t reg  = ggml_backend_dev_backend_reg(dev);
                const char *       name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
                if (name == nullptr || std::strstr(name, reg_filter) == nullptr) {
                    continue;
                }
            }
            if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                return backend;
            }
        }
    }
    return nullptr;
}

/* 选取主后端：Vulkan（构建时启用且设备可用）优先，否则 CPU。 */
static bool clip_init_backend(struct clip_ctx * ctx, std::string * error) {
    if (ggml_backend_reg_count() == 0) {
        ggml_backend_load_all();
    }

    const auto preference =
        static_cast<clip_backend_preference>(g_clip_backend_preference.load(std::memory_order_relaxed));

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        if (error) *error = "无法创建 ggml CPU 后端";
        return false;
    }

    ggml_backend_t primary = cpu;
    const char *   name    = "CPU";

    /* 明确指定 GPU 后端时按注册名过滤（同机可能同时存在 CUDA / Vulkan）；
     * auto 则取注册顺序中的第一个可用 GPU。 */
    const char * gpu_filter = nullptr;
    const char * gpu_name   = "GPU";
    switch (preference) {
        case clip_backend_preference::vulkan: gpu_filter = "Vulkan"; gpu_name = "Vulkan"; break;
        case clip_backend_preference::cuda:   gpu_filter = "CUDA";   gpu_name = "CUDA";   break;
        case clip_backend_preference::hip:    gpu_filter = "HIP";    gpu_name = "HIP";    break;
        default: break;
    }

    if (preference != clip_backend_preference::cpu) {
        if (ggml_backend_t gpu = clip_init_gpu_backend(gpu_filter)) {
            primary = gpu;
            name    = gpu_name;
        }
    }

    if (primary == cpu && preference != clip_backend_preference::auto_select &&
        preference != clip_backend_preference::cpu) {
        if (error) {
            *error = std::string("请求了 ") + gpu_name + " 后端，但当前构建或设备没有可用的 " + gpu_name +
                     " 设备（CUDA / ROCm 需分别以 -DGGML_CUDA=ON / -DGGML_HIP=ON 重新编译）";
        }
        ggml_backend_free(cpu);
        return false;
    }

    ctx->backend     = primary;
    ctx->cpu_backend = primary == cpu ? nullptr : cpu;
    ctx->backend_name = primary == cpu ? std::string("CPU") : std::string(name);
    return true;
}

/* 创建 sched：GPU 时组合 {GPU, CPU}，不改的算子由 sched 回退。 */
static bool clip_init_sched(struct clip_ctx * ctx) {
    ggml_backend_t backends[2];
    int n = 0;
    backends[n++] = ctx->backend;
    if (ctx->cpu_backend != nullptr) {
        backends[n++] = ctx->cpu_backend;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n, kClipSchedGraphSize, false, false);
    return ctx->sched != nullptr;
}

/* 图执行：统一走 sched；CPU 线程数在调用前同步。 */
static bool clip_graph_compute(struct clip_ctx * ctx, struct ggml_cgraph * graph, int n_threads) {
#if defined(GGML_USE_CPU)
    ggml_backend_t cpu = ctx->cpu_backend != nullptr ? ctx->cpu_backend : ctx->backend;
    ggml_backend_cpu_set_n_threads(cpu, n_threads > 0 ? n_threads : 1);
#else
    (void)n_threads;
#endif
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, graph)) {
        fprintf(stderr, "%s: 图内存分配失败\n", __func__);
        return false;
    }
    const enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: 图计算失败 (%d)\n", __func__, (int)status);
        return false;
    }
    return true;
}

/* 主机侧 L2 归一化（逐行）；与原图内 sqrt/sum 归一化数值等价。 */
static void clip_normalize_rows(float * data, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        float * row = data + (size_t)r * dim;
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += (double)row[i] * (double)row[i];
        }
        const float inv = sum > 0.0 ? (float)(1.0 / std::sqrt(sum)) : 0.0f;
        for (int i = 0; i < dim; i++) {
            row[i] *= inv;
        }
    }
}

/* 后端偏好设置（供 C ABI 调用）。 */
static int clip_set_backend_preference(const char * name) {
    std::string value = name != nullptr ? name : "";
    for (char & c : value) {
        c = (char)std::tolower((unsigned char)c);
    }
    clip_backend_preference preference = clip_backend_preference::auto_select;
    if (value.empty() || value == "auto") {
        preference = clip_backend_preference::auto_select;
    } else if (value == "cpu") {
        preference = clip_backend_preference::cpu;
    } else if (value == "vulkan" || value == "gpu") {
        preference = clip_backend_preference::vulkan;
    } else if (value == "cuda") {
        preference = clip_backend_preference::cuda;
    } else if (value == "hip" || value == "rocm") {
        preference = clip_backend_preference::hip;
    } else {
        return 1;
    }
    g_clip_backend_preference.store(static_cast<int>(preference), std::memory_order_relaxed);
    return 0;
}

// read and create ggml_context containing the tensors and their data
struct clip_ctx * clip_model_load(const char * fname, const int verbosity = 1) {

    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname, params);

    if (verbosity >= 1) {
        const int n_tensors = gguf_get_n_tensors(ctx);
        const int n_kv = gguf_get_n_kv(ctx);
        const int ftype = get_u32(ctx, KEY_FTYPE);
        const std::string ftype_str = get_ftype(ftype);
        const int idx_desc = get_key_idx(ctx, KEY_DESCRIPTION);
        const std::string description = gguf_get_val_str(ctx, idx_desc);
        const int idx_name = gguf_find_key(ctx, KEY_NAME);
        if (idx_name != -1) { // make name optional temporarily as some of the uploaded models missing it due to a bug
            const std::string name = gguf_get_val_str(ctx, idx_name);
            printf("%s: model name:   %s\n", __func__, name.c_str());
        }
        printf("%s: description:  %s\n", __func__, description.c_str());
        printf("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx));
        printf("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx));
        printf("%s: n_tensors:    %d\n", __func__, n_tensors);
        printf("%s: n_kv:         %d\n", __func__, n_kv);
        printf("%s: ftype:        %s\n", __func__, ftype_str.c_str());
        printf("\n");
    }

    // kv
    if (verbosity >= 3) {
        const int n_kv = gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            printf("%s: kv[%d]: key = %s\n", __func__, i, key);
        }
        printf("\n");
    }

    // data
    size_t ctx_size = 0;
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(meta, name);
            // 只计算元数据开销；张量数据由后端缓冲分配。
            ctx_size += ggml_tensor_overhead();
            if (verbosity >= 3) {
                printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu\n", __func__, i, ggml_n_dims(cur), cur->name,
                       ggml_nbytes(cur));
            }
        }
    }

    clip_ctx * new_clip = new clip_ctx;

    // model size and capabilities
    {
        int idx = get_key_idx(ctx, KEY_HAS_TEXT_ENC);
        new_clip->has_text_encoder = gguf_get_val_bool(ctx, idx);

        idx = get_key_idx(ctx, KEY_HAS_VIS_ENC);
        new_clip->has_vision_encoder = gguf_get_val_bool(ctx, idx);

        idx = get_key_idx(ctx, KEY_USE_GELU);
        new_clip->use_gelu = gguf_get_val_bool(ctx, idx);
        // Newer files carry per-tower activation / tokenizer / architecture
        // metadata. Fall back to the original CLIP semantics when absent.
        new_clip->use_gelu = get_bool_optional(ctx, KEY_VISION_USE_GELU, new_clip->use_gelu);
        new_clip->text_use_gelu = get_bool_optional(ctx, KEY_TEXT_USE_GELU, new_clip->use_gelu);
        new_clip->use_wordpiece = std::string(get_str_optional(ctx, KEY_TOKENIZER, "bpe")) == "wordpiece";
        new_clip->text_bert = std::string(get_str_optional(ctx, KEY_TEXT_ARCH, "clip")) == "bert";
        new_clip->text_pool_cls =
            std::string(get_str_optional(ctx, KEY_TEXT_POOLING, new_clip->text_bert ? "cls" : "eot")) == "cls";
        new_clip->text_pad_length = (int32_t)get_u32_optional(ctx, KEY_TEXT_PAD_LENGTH, 0);

        if (verbosity >= 1) {
            printf("%s: text_encoder:   %d\n", __func__, new_clip->has_text_encoder);
            printf("%s: vision_encoder: %d\n", __func__, new_clip->has_vision_encoder);
            if (new_clip->has_text_encoder) {
                printf("%s: text tokenizer: %s\n", __func__, new_clip->use_wordpiece ? "wordpiece" : "bpe");
                printf("%s: text arch:      %s\n", __func__, new_clip->text_bert ? "bert" : "clip");
                printf("%s: text pooling:   %s\n", __func__, new_clip->text_pool_cls ? "cls" : "eot");
            }
            printf("%s: model size:     %.2f MB\n", __func__, (ctx_size / 1024.0 / 1024.0));
            printf("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
        }
    }

    // load tensors
    {
        struct ggml_init_params params = {};
        params.mem_size = ctx_size + 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc = true;

        new_clip->ctx = ggml_init(params);
        if (!new_clip->ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            return nullptr;
        }

        std::string backend_error;
        if (!clip_init_backend(new_clip, &backend_error)) {
            fprintf(stderr, "%s: %s\n", __func__, backend_error.c_str());
            clip_free(new_clip);
            return nullptr;
        }

        const int n_tensors = gguf_get_n_tensors(ctx);

        // 先在 no_alloc 上下文里建好全部张量元数据，再一次性分配后端缓冲。
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_clip->ctx, t);
            ggml_set_name(cur, name);
        }

        new_clip->weights_buffer = ggml_backend_alloc_ctx_tensors(new_clip->ctx, new_clip->backend);
        if (new_clip->weights_buffer == nullptr) {
            fprintf(stderr, "%s: 无法为权重分配后端缓冲\n", __func__);
            clip_free(new_clip);
            return nullptr;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            printf("cannot open model file for loading tensors\n");
            clip_free(new_clip);
            return nullptr;
        }

        std::vector<uint8_t> staging;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_get_tensor(new_clip->ctx, name);

            const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                printf("%s: failed to seek for tensor %s\n", __func__, name);
                clip_free(new_clip);
                return nullptr;
            }

            const size_t nbytes = ggml_nbytes(t);
            staging.resize(nbytes);
            fin.read(reinterpret_cast<char *>(staging.data()), (std::streamsize)nbytes);
            if (!fin) {
                printf("%s: failed to read tensor %s\n", __func__, name);
                clip_free(new_clip);
                return nullptr;
            }
            ggml_backend_tensor_set(cur, staging.data(), 0, nbytes);
        }

        fin.close();

        if (!clip_init_sched(new_clip)) {
            fprintf(stderr, "%s: 无法创建 ggml 调度器\n", __func__);
            clip_free(new_clip);
            return nullptr;
        }
    }

    // text model
    if (new_clip->has_text_encoder) {
        // load text model
        auto & text_model = new_clip->text_model;
        auto & hparams = text_model.hparams;
        hparams.hidden_size = get_u32(ctx, format(KEY_N_EMBD, "text"));
        hparams.n_head = get_u32(ctx, format(KEY_N_HEAD, "text"));
        hparams.n_intermediate = get_u32(ctx, format(KEY_N_FF, "text"));
        hparams.n_layer = get_u32(ctx, format(KEY_N_BLOCK, "text"));
        hparams.num_positions = get_u32(ctx, KEY_N_POSITIONS);
        hparams.projection_dim = get_u32(ctx, format(KEY_PROJ_DIM, "text"));
        hparams.eps = get_f32(ctx, format(KEY_LAYER_NORM_EPS, "text"));

        const int idx_tokens = get_key_idx(ctx, KEY_TOKENS);
        hparams.n_vocab = gguf_get_arr_n(ctx, idx_tokens);
        auto & vocab = new_clip->vocab;
        for (int id = 0; id < hparams.n_vocab; ++id) {
            const std::string token = gguf_get_arr_str(ctx, idx_tokens, id);
            vocab.id_to_token[id] = token;
            vocab.token_to_id[token] = id;
        }

        if (verbosity >= 2) {
            printf("\n%s: text model hparams\n", __func__);
            printf("n_vocab            %d\n", hparams.n_vocab);
            printf("num_positions      %d\n", hparams.num_positions);
            printf("t_hidden_size      %d\n", hparams.hidden_size);
            printf("t_n_intermediate   %d\n", hparams.n_intermediate);
            printf("t_projection_dim   %d\n", hparams.projection_dim);
            printf("t_n_head           %d\n", hparams.n_head);
            printf("t_n_layer          %d\n", hparams.n_layer);
        }

        text_model.token_embeddings = get_tensor(new_clip->ctx, format(TN_TOKEN_EMBD, "t"));
        text_model.position_embeddings = get_tensor(new_clip->ctx, format(TN_POS_EMBD, "t"));
        text_model.post_ln_w = get_tensor_optional(new_clip->ctx, format(TN_LN_POST, "t", "weight"));
        text_model.post_ln_b = get_tensor_optional(new_clip->ctx, format(TN_LN_POST, "t", "bias"));
        // BERT-style text towers (Chinese-CLIP) use these instead of an EOT
        // pooling + final layer norm.
        text_model.pre_ln_w = get_tensor_optional(new_clip->ctx, format(TN_LN_PRE, "t", "weight"));
        text_model.pre_ln_b = get_tensor_optional(new_clip->ctx, format(TN_LN_PRE, "t", "bias"));
        text_model.token_type_embeddings = get_tensor_optional(new_clip->ctx, TN_TOKEN_TYPE_EMBD);
        text_model.projection = get_tensor(new_clip->ctx, TN_TEXT_PROJ);
        if (text_model.projection->ne[1] != hparams.projection_dim) {
            throw std::runtime_error(format("clip.text.projection_dim (%d) does not match text_projection.weight (%d)",
                                            hparams.projection_dim, (int)text_model.projection->ne[1]));
        }
        text_model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = text_model.layers[il];
            layer.k_w = get_tensor(new_clip->ctx, format(TN_ATTN_K, "t", il, "weight"));
            layer.q_w = get_tensor(new_clip->ctx, format(TN_ATTN_Q, "t", il, "weight"));
            layer.v_w = get_tensor(new_clip->ctx, format(TN_ATTN_V, "t", il, "weight"));
            layer.o_w = get_tensor(new_clip->ctx, format(TN_ATTN_OUTPUT, "t", il, "weight"));
            layer.ln_1_w = get_tensor(new_clip->ctx, format(TN_LN_1, "t", il, "weight"));
            layer.ln_2_w = get_tensor(new_clip->ctx, format(TN_LN_2, "t", il, "weight"));
            layer.ff_i_w = get_tensor(new_clip->ctx, format(TN_FFN_DOWN, "t", il, "weight"));
            layer.ff_o_w = get_tensor(new_clip->ctx, format(TN_FFN_UP, "t", il, "weight"));
            layer.k_b = get_tensor(new_clip->ctx, format(TN_ATTN_K, "t", il, "bias"));
            layer.q_b = get_tensor(new_clip->ctx, format(TN_ATTN_Q, "t", il, "bias"));
            layer.v_b = get_tensor(new_clip->ctx, format(TN_ATTN_V, "t", il, "bias"));
            layer.o_b = get_tensor(new_clip->ctx, format(TN_ATTN_OUTPUT, "t", il, "bias"));
            layer.ln_1_b = get_tensor(new_clip->ctx, format(TN_LN_1, "t", il, "bias"));
            layer.ln_2_b = get_tensor(new_clip->ctx, format(TN_LN_2, "t", il, "bias"));
            layer.ff_i_b = get_tensor(new_clip->ctx, format(TN_FFN_DOWN, "t", il, "bias"));
            layer.ff_o_b = get_tensor(new_clip->ctx, format(TN_FFN_UP, "t", il, "bias"));
        }
    }

    // vision model
    if (new_clip->has_vision_encoder) {
        // load vision model
        auto & vision_model = new_clip->vision_model;
        auto & hparams = vision_model.hparams;
        hparams.hidden_size = get_u32(ctx, format(KEY_N_EMBD, "vision"));
        hparams.n_head = get_u32(ctx, format(KEY_N_HEAD, "vision"));
        hparams.n_intermediate = get_u32(ctx, format(KEY_N_FF, "vision"));
        hparams.n_layer = get_u32(ctx, format(KEY_N_BLOCK, "vision"));
        hparams.image_size = get_u32(ctx, KEY_IMAGE_SIZE);
        hparams.patch_size = get_u32(ctx, KEY_PATCH_SIZE);
        hparams.projection_dim = get_u32(ctx, format(KEY_PROJ_DIM, "vision"));
        hparams.eps = get_f32(ctx, format(KEY_LAYER_NORM_EPS, "vision"));

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std = get_key_idx(ctx, KEY_IMAGE_STD);
        for (int i = 0; i < 3; ++i) {
            new_clip->image_mean[i] = ((float *)gguf_get_arr_data(ctx, idx_mean))[i];
            new_clip->image_std[i] = ((float *)gguf_get_arr_data(ctx, idx_std))[i];
        }

        // 旧版转换器写出的 GGUF 没有该键，缺省保持上游的裁剪行为。
        new_clip->vision_do_center_crop = get_bool_optional(ctx, KEY_DO_CENTER_CROP, true);

        if (verbosity >= 2) {
            printf("\n%s: vision model hparams\n", __func__);
            printf("image_size         %d\n", hparams.image_size);
            printf("do_center_crop     %d\n", (int)new_clip->vision_do_center_crop);
            printf("patch_size         %d\n", hparams.patch_size);
            printf("v_hidden_size      %d\n", hparams.hidden_size);
            printf("v_n_intermediate   %d\n", hparams.n_intermediate);
            printf("v_projection_dim   %d\n", hparams.projection_dim);
            printf("v_n_head           %d\n", hparams.n_head);
            printf("v_n_layer          %d\n", hparams.n_layer);
        }

        vision_model.patch_embeddings = get_tensor(new_clip->ctx, TN_PATCH_EMBD);
        vision_model.class_embedding = get_tensor(new_clip->ctx, TN_CLASS_EMBD);
        vision_model.position_embeddings = get_tensor(new_clip->ctx, format(TN_POS_EMBD, "v"));
        vision_model.pre_ln_w = get_tensor(new_clip->ctx, format(TN_LN_PRE, "v", "weight"));
        vision_model.pre_ln_b = get_tensor(new_clip->ctx, format(TN_LN_PRE, "v", "bias"));
        vision_model.post_ln_w = get_tensor(new_clip->ctx, format(TN_LN_POST, "v", "weight"));
        vision_model.post_ln_b = get_tensor(new_clip->ctx, format(TN_LN_POST, "v", "bias"));
        vision_model.projection = get_tensor(new_clip->ctx, TN_VIS_PROJ);
        if (vision_model.projection->ne[1] != hparams.projection_dim) {
            throw std::runtime_error(format("clip.vision.projection_dim (%d) does not match visual_projection.weight (%d)",
                                            hparams.projection_dim, (int)vision_model.projection->ne[1]));
        }
        vision_model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.k_w = get_tensor(new_clip->ctx, format(TN_ATTN_K, "v", il, "weight"));
            layer.q_w = get_tensor(new_clip->ctx, format(TN_ATTN_Q, "v", il, "weight"));
            layer.v_w = get_tensor(new_clip->ctx, format(TN_ATTN_V, "v", il, "weight"));
            layer.o_w = get_tensor(new_clip->ctx, format(TN_ATTN_OUTPUT, "v", il, "weight"));
            layer.ln_1_w = get_tensor(new_clip->ctx, format(TN_LN_1, "v", il, "weight"));
            layer.ln_2_w = get_tensor(new_clip->ctx, format(TN_LN_2, "v", il, "weight"));
            layer.ff_i_w = get_tensor(new_clip->ctx, format(TN_FFN_DOWN, "v", il, "weight"));
            layer.ff_o_w = get_tensor(new_clip->ctx, format(TN_FFN_UP, "v", il, "weight"));
            layer.k_b = get_tensor(new_clip->ctx, format(TN_ATTN_K, "v", il, "bias"));
            layer.q_b = get_tensor(new_clip->ctx, format(TN_ATTN_Q, "v", il, "bias"));
            layer.v_b = get_tensor(new_clip->ctx, format(TN_ATTN_V, "v", il, "bias"));
            layer.o_b = get_tensor(new_clip->ctx, format(TN_ATTN_OUTPUT, "v", il, "bias"));
            layer.ln_1_b = get_tensor(new_clip->ctx, format(TN_LN_1, "v", il, "bias"));
            layer.ln_2_b = get_tensor(new_clip->ctx, format(TN_LN_2, "v", il, "bias"));
            layer.ff_i_b = get_tensor(new_clip->ctx, format(TN_FFN_DOWN, "v", il, "bias"));
            layer.ff_o_b = get_tensor(new_clip->ctx, format(TN_FFN_UP, "v", il, "bias"));
        }
    }

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;

    if (verbosity >= 1) {
        printf("\n%s: backend = %s\n", __func__, new_clip->backend_name.c_str());
    }

    return new_clip;
}

//
// WordPiece tokenizer (BERT-style text towers, e.g. Chinese-CLIP)
//

static void utf8_decode(const std::string & str, std::vector<uint32_t> & codepoints) {
    const unsigned char * s = (const unsigned char *)str.c_str();
    size_t i = 0;
    const size_t n = str.size();
    while (i < n) {
        uint32_t cp = 0;
        unsigned char c = s[i];
        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c >> 5) == 0x6 && i + 1 < n) {
            cp = ((uint32_t)(c & 0x1F) << 6) | (s[i + 1] & 0x3F);
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < n) {
            cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < n) {
            cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12) |
                 ((uint32_t)(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
            i += 4;
        } else {
            cp = 0xFFFD;
            i += 1;
        }
        codepoints.push_back(cp);
    }
}

static std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
    return out;
}

static std::string utf8_lower_ascii(const std::string & str) {
    std::string out = str;
    for (size_t i = 0; i < out.size(); i++) {
        unsigned char c = (unsigned char)out[i];
        if (c >= 'A' && c <= 'Z') {
            out[i] = (char)(c - 'A' + 'a');
        }
    }
    return out;
}

static bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

static bool is_whitespace_codepoint(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000;
}

static bool is_punctuation_codepoint(uint32_t cp) {
    if (cp < 128) {
        return std::isalnum((int)cp) == 0;
    }
    // CJK symbols/punctuation, half-width/full-width forms and general punctuation.
    return (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFF65) || (cp >= 0x2000 && cp <= 0x206F);
}

static bool vocab_lookup(const clip_vocab & vocab, const std::string & token, clip_vocab_id & id) {
    auto it = vocab.token_to_id.find(token);
    if (it == vocab.token_to_id.end()) {
        return false;
    }
    id = it->second;
    return true;
}

// Splits text the way HF BertTokenizer's BasicTokenizer does (whitespace, CJK
// characters and punctuation become individual tokens) and applies greedy
// longest-match-first WordPiece. The result is padded to a fixed length with
// [PAD] so that the attention mask has a stable shape.
static std::vector<clip_vocab_id> tokenize_wordpiece(const clip_ctx * ctx, const char * text) {
    const clip_vocab & vocab = ctx->vocab;

    clip_vocab_id unk_id = 100;
    clip_vocab_id cls_id = 101;
    clip_vocab_id sep_id = 102;
    clip_vocab_id pad_id = 0;
    vocab_lookup(vocab, "[UNK]", unk_id);
    vocab_lookup(vocab, "[CLS]", cls_id);
    vocab_lookup(vocab, "[SEP]", sep_id);
    vocab_lookup(vocab, "[PAD]", pad_id);

    std::vector<uint32_t> codepoints;
    utf8_decode(utf8_lower_ascii(text), codepoints);

    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < codepoints.size(); i++) {
        const uint32_t cp = codepoints[i];
        if (is_whitespace_codepoint(cp)) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        } else if (is_cjk_codepoint(cp) || is_punctuation_codepoint(cp)) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            words.push_back(utf8_encode(cp));
        } else {
            current += utf8_encode(cp);
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }

    std::vector<clip_vocab_id> ids;
    ids.push_back(cls_id);

    for (size_t w = 0; w < words.size(); w++) {
        std::vector<uint32_t> word_cps;
        utf8_decode(words[w], word_cps);

        if (word_cps.size() > 100) { // max_input_chars_per_word
            ids.push_back(unk_id);
            continue;
        }

        std::vector<clip_vocab_id> sub_ids;
        bool is_bad = false;
        size_t start = 0;
        while (start < word_cps.size()) {
            size_t end = word_cps.size();
            bool found = false;
            while (start < end) {
                std::string sub;
                if (start > 0) {
                    sub += "##";
                }
                for (size_t k = start; k < end; k++) {
                    sub += utf8_encode(word_cps[k]);
                }
                clip_vocab_id id;
                if (vocab_lookup(vocab, sub, id)) {
                    sub_ids.push_back(id);
                    start = end;
                    found = true;
                    break;
                }
                end--;
            }
            if (!found) {
                is_bad = true;
                break;
            }
        }

        if (is_bad) {
            ids.push_back(unk_id);
        } else {
            ids.insert(ids.end(), sub_ids.begin(), sub_ids.end());
        }
    }

    ids.push_back(sep_id);

    int pad_length = ctx->text_pad_length > 0 ? ctx->text_pad_length : 64;
    if (pad_length < 2) {
        pad_length = 2;
    }
    if (ctx->text_model.hparams.num_positions > 0 && pad_length > ctx->text_model.hparams.num_positions) {
        pad_length = ctx->text_model.hparams.num_positions;
    }
    if ((int)ids.size() > pad_length) {
        ids.resize(pad_length - 1);
        ids.push_back(sep_id);
    }
    while ((int)ids.size() < pad_length) {
        ids.push_back(pad_id);
    }

    return ids;
}

bool clip_tokenize(const clip_ctx * ctx, const char * text, struct clip_tokens * tokens) {
    if (!ctx->has_text_encoder) {
        printf("This GGUF file seems to have no text encoder\n");
        return false;
    }

    if (ctx->use_wordpiece) {
        const std::vector<clip_vocab_id> v_tokens = tokenize_wordpiece(ctx, text);
        tokens->size = v_tokens.size();
        tokens->data = new clip_vocab_id[v_tokens.size()];
        std::copy(v_tokens.begin(), v_tokens.end(), tokens->data);
        return true;
    }

    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        // Generate the subpattern from the special_tokens vector if it's not empty
        if (!ctx->vocab.special_tokens.empty()) {
            std::string special_tokens_subpattern;
            for (const auto & token : ctx->vocab.special_tokens) {
                if (!special_tokens_subpattern.empty()) {
                    special_tokens_subpattern += "|";
                }
                special_tokens_subpattern += token;
            }

            // Modify the regex pattern with the generated special tokens subpattern
            pat = special_tokens_subpattern + "|" + pat;
        }

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    std::vector<clip_vocab::id> v_tokens;
    v_tokens.push_back(49406); // startoftext

    for (const auto & word : words) {
        // feel lucky? let's try if it's a full word
        std::string full_word = "";
        if (word.find(" ") == 0) // starts_with for C++11
        {
            full_word += word.substr(1);
        } else {
            full_word += word;
        }
        full_word += "</w>";
        auto wit = ctx->vocab.token_to_id.find(full_word);
        if (wit != ctx->vocab.token_to_id.end()) {
            v_tokens.push_back(wit->second);
            continue;
        }

        for (int i = 0; i < word.size();) {
            for (int j = word.size() - 1; j >= i; j--) {
                auto cand = word.substr(i, j - i + 1);
                auto it = ctx->vocab.token_to_id.find(cand);
                if (it != ctx->vocab.token_to_id.end()) { // word.substr(i, j-i+1) in vocab
                    v_tokens.push_back(it->second);
                    i = j + 1;
                    break;
                } else if (j == i) { // word.substr(i, 1) has no matching
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                    i++;
                }
            }
        }
    }

    v_tokens.push_back(49407); // endoftext

    tokens->size = v_tokens.size();

    tokens->data = new int[v_tokens.size()];
    std::copy(v_tokens.begin(), v_tokens.end(), tokens->data);

    return true;
}

clip_image_u8 * clip_image_u8_make() { return new clip_image_u8(); }

clip_image_f32 * clip_image_f32_make() { return new clip_image_f32(); }

void clip_image_u8_clean(clip_image_u8* img) {
    if (img->data){
	delete[] img->data;
	img->data = NULL;
    }
}

void clip_image_f32_clean(clip_image_f32* res) {
    if (res->data){
	delete[] res->data;
	res->data = NULL;
    }
}

void clip_image_u8_free(clip_image_u8* img) {
    clip_image_u8_clean(img);
    delete img;
}

void clip_image_f32_free(clip_image_f32* res) {
    clip_image_f32_clean(res);
    delete res;
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname);
        return false;
    }

    img->nx = nx;
    img->ny = ny;
    img->size = nx * ny * 3;
    img->data = new uint8_t[img->size]();
    memcpy(img->data, data, img->size);

    stbi_image_free(data);

    return true;
}

static inline double bicubic_filter(double x) {
#define a -0.5
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
        return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
#undef a
}

static bool precompute_coeffs(int inSize, float in0, float in1, int outSize, double ** kkp, int ** boundsp, int * ksize) {
    double support = 2.0; // Bicubic filter support from Resample.c
    double filterscale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }
    support *= filterscale;
    int ksize_local = (int)ceil(support) * 2 + 1;

    double * kk = (double *)malloc(outSize * ksize_local * sizeof(double));
    int * bounds = (int *)malloc(outSize * 2 * sizeof(int));
    if (!kk || !bounds) {
        free(kk);
        free(bounds);
        return false;
    }

    for (int xx = 0; xx < outSize; xx++) {
        double center = in0 + (xx + 0.5) * (in1 - in0) / outSize;
        double ww = 0.0;
        double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0)
            xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > inSize)
            xmax = inSize;
        xmax -= xmin;

        double * k = &kk[xx * ksize_local];
        for (int x = 0; x < xmax; x++) {
            double w = bicubic_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; x++) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }
        for (int x = xmax; x < ksize_local; x++) {
            k[x] = 0.0;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }

    *kkp = kk;
    *boundsp = bounds;
    *ksize = ksize_local;
    return true;
}

// normalize: x = (x - mean) / std
bool clip_image_preprocess(const clip_ctx * ctx, const clip_image_u8 * img, clip_image_f32 * res) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    const int nx = img->nx;
    const int ny = img->ny;
    const int nx2 = ctx->vision_model.hparams.image_size;
    const int ny2 = ctx->vision_model.hparams.image_size;

    // Setting the output image size and allocating memory
    res->nx = nx2;
    res->ny = ny2;
    res->size = 3 * nx2 * ny2;
    res->data = new float[res->size]();
    if (!res->data) {
        printf("clip_image_f32 Memory allocation failed\n");
        return false;
    }

    // 目标缩放尺寸：
    //   do_center_crop=true （上游默认）：短边等比缩放到 image_size，再中心裁剪；
    //   do_center_crop=false（Chinese-CLIP）：整幅图直接 resize 到 image_size×image_size，
    //   与 ONNX/PIL 预处理保持一致。
    // 两种路径共用下方的双三次插值（precompute_coeffs/bicubic_filter 对应 PIL
    // Resample.c 的 BICUBIC，即 preprocessor_config.json 中的 resample=3）。
    int nx3, ny3;
    if (ctx->vision_do_center_crop) {
        const float scale = std::min((float)nx, (float)ny) / (float)ctx->vision_model.hparams.image_size;
        nx3 = (int)(nx / scale + 0.5f);
        ny3 = (int)(ny / scale + 0.5f);
    } else {
        nx3 = nx2;
        ny3 = ny2;
    }

    const auto & m3 = ctx->image_mean;
    const auto & s3 = ctx->image_std;

    // Calculating horizontal and vertical coeffs
    double *kk_horiz, *kk_vert;
    int *bounds_horiz, *bounds_vert;
    int ksize_horiz, ksize_vert;

    if (!precompute_coeffs(nx, 0.0f, (float)nx, nx3, &kk_horiz, &bounds_horiz, &ksize_horiz) ||
        !precompute_coeffs(ny, 0.0f, (float)ny, ny3, &kk_vert, &bounds_vert, &ksize_vert)) {
        delete[] res->data;
        free(kk_horiz);
        free(bounds_horiz);
        free(kk_vert);
        free(bounds_vert);
        printf("Failed to calculate coeffs\n");
        return false;
    }

    // Intermediate image buffer (stores horizontal resampling results)
    float * temp = new float[3 * nx3 * ny]();
    if (!temp) {
        delete[] res->data;
        free(kk_horiz);
        free(bounds_horiz);
        free(kk_vert);
        free(bounds_vert);
        printf("Failed to allocate intermediate buffer memory\n");
        return false;
    }

    // Horizontal bicubic resampling
    for (int y = 0; y < ny; y++) {
        for (int xx = 0; xx < nx3; xx++) {
            int xmin = bounds_horiz[xx * 2 + 0];
            int xmax = bounds_horiz[xx * 2 + 1];
            double * k = &kk_horiz[xx * ksize_horiz];
            for (int c = 0; c < 3; c++) {
                double ss = 0.0;
                for (int x = 0; x < xmax; x++) {
                    int src_idx = 3 * (y * nx + (x + xmin)) + c;
                    ss += (double)img->data[src_idx] * k[x];
                }
                int dst_idx = 3 * (y * nx3 + xx) + c;
                temp[dst_idx] = std::min(std::max((float)ss, 0.0f), 255.0f);
            }
        }
    }

    // Vertical bicubic resampling
    float * resampled = new float[3 * nx3 * ny3]();
    if (!resampled) {
        delete[] temp;
        delete[] res->data;
        free(kk_horiz);
        free(bounds_horiz);
        free(kk_vert);
        free(bounds_vert);
        printf("Failed to allocate resampling buffer memory\n");
        return false;
    }

    for (int yy = 0; yy < ny3; yy++) {
        int ymin = bounds_vert[yy * 2 + 0];
        int ymax = bounds_vert[yy * 2 + 1];
        double * k = &kk_vert[yy * ksize_vert];
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                double ss = 0.0;
                for (int y = 0; y < ymax; y++) {
                    int src_idx = 3 * ((y + ymin) * nx3 + x) + c;
                    ss += (double)temp[src_idx] * k[y];
                }
                int dst_idx = 3 * (yy * nx3 + x) + c;
                resampled[dst_idx] = std::min(std::max((float)ss, 0.0f), 255.0f);
            }
        }
    }

    // Center crop and normalize
    int x_offset = (nx3 - nx2) / 2;
    int y_offset = (ny3 - ny2) / 2;

    for (int yy = 0; yy < ny2; yy++) {
        for (int x = 0; x < nx2; x++) {
            int src_y = yy + y_offset;
            int src_x = x + x_offset;
            int src_idx = 3 * (src_y * nx3 + src_x);
            int dst_idx = 3 * (yy * nx2 + x);
            for (int c = 0; c < 3; c++) {
                float v = resampled[src_idx + c];
                res->data[dst_idx + c] = ((v / 255.0f) - m3[c]) / s3[c];
            }
        }
    }

    delete[] resampled;
    delete[] temp;
    free(kk_horiz);
    free(bounds_horiz);
    free(kk_vert);
    free(bounds_vert);

    return true;
}

// Structure to hold the image data as an input to function to be executed for thread
typedef struct {
    const clip_image_u8 * input;
    clip_image_f32 * resized;
    const clip_ctx * ctx;
} ImageData;

// Function to batch-preprocess multiple images
void clip_image_batch_preprocess(const clip_ctx * ctx, const int n_threads, const clip_image_u8_batch * img_inputs,
                                 clip_image_f32_batch * imgs_resized) {
    imgs_resized->size = img_inputs->size;

    if (img_inputs->size == 0) {
        return;
    }

    int num_threads = std::min(std::max(n_threads, 1), static_cast<int>(img_inputs->size));

    std::vector<ImageData> imageData(img_inputs->size);
    for (size_t i = 0; i < img_inputs->size; i++) {
        imageData[i].input = &img_inputs->data[i];
        imageData[i].resized = &imgs_resized->data[i];
        imageData[i].ctx = ctx;
    }

    if (num_threads == 1) {
        // Single-threaded case
        for (size_t i = 0; i < img_inputs->size; i++) {
            clip_image_preprocess(ctx, &img_inputs->data[i], &imgs_resized->data[i]);
        }
        return;
    }

    // Divide the images among the threads
    const int images_per_thread = static_cast<int>(img_inputs->size) / num_threads;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; t++) {
        const int start_index = t * images_per_thread;
        const int end_index = (t == num_threads - 1) ? static_cast<int>(img_inputs->size) : start_index + images_per_thread;

        threads.emplace_back([ctx, &imageData, start_index, end_index]() {
            for (int i = start_index; i < end_index; i++) {
                clip_image_preprocess(ctx, imageData[i].input, imageData[i].resized);
            }
        });
    }

    // Wait for all threads to finish
    for (auto & thread : threads) {
        thread.join();
    }
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->sched != nullptr) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    if (ctx->weights_buffer != nullptr) {
        ggml_backend_buffer_free(ctx->weights_buffer);
        ctx->weights_buffer = nullptr;
    }
    ggml_free(ctx->ctx);
    ctx->ctx = nullptr;
    if (ctx->backend != nullptr) {
        ggml_backend_free(ctx->backend);
        ctx->backend = nullptr;
    }
    if (ctx->cpu_backend != nullptr) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }
    gguf_free(ctx->ctx_gguf);
    delete ctx;
}

bool clip_text_encode(const clip_ctx * ctx, const int n_threads, const clip_tokens * tokens, float * vec,
                      const bool normalize) {
    if (!ctx->has_text_encoder) {
        printf("This GGUF file seems to have no text encoder\n");
        return false;
    }

    const auto & model = ctx->text_model;
    const auto & hparams = model.hparams;
    const size_t N = tokens->size;

    const int n_vocab = hparams.n_vocab;
    const int num_positions = hparams.num_positions;
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;
    const int n_layer = hparams.n_layer;
    const int n_intermediate = hparams.n_intermediate;
    const int projection_dim = hparams.projection_dim;
    const float eps = hparams.eps;

    /* 输入与常量张量：先建在 no_alloc 上下文，再由 CPU 后端缓冲分配并写入。 */
    struct ggml_init_params in_params = {};
    in_params.mem_size = 8u * 1024u * 1024u;
    in_params.no_alloc = true;
    struct ggml_context * in_ctx = ggml_init(in_params);
    if (in_ctx == nullptr) {
        fprintf(stderr, "%s: 无法创建输入上下文\n", __func__);
        return false;
    }

    struct ggml_tensor * input_ids  = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, N);
    struct ggml_tensor * positions  = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, N);
    struct ggml_tensor * zero_types = nullptr;
    struct ggml_tensor * attn_mask  = nullptr;
    struct ggml_tensor * pool_token = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, 1);
    if (ctx->text_bert && model.token_type_embeddings) {
        zero_types = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, N);
    }

    /* BERT 文本塔：padding 位置不作为注意力 key，保证定长 padding 不影响 CLS。 */
    bool need_attn_mask = false;
    clip_vocab_id pad_id = 0;
    if (ctx->text_bert) {
        vocab_lookup(ctx->vocab, "[PAD]", pad_id);
        for (size_t i = 0; i < N; i++) {
            if (tokens->data[i] == pad_id) {
                need_attn_mask = true;
                break;
            }
        }
        if (need_attn_mask) {
            attn_mask = ggml_new_tensor_2d(in_ctx, GGML_TYPE_F32, N, N);
        }
    }

    ggml_backend_t in_backend = ctx->cpu_backend != nullptr ? ctx->cpu_backend : ctx->backend;
    ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(in_ctx, in_backend);
    if (in_buf == nullptr) {
        fprintf(stderr, "%s: 无法分配输入缓冲\n", __func__);
        ggml_free(in_ctx);
        return false;
    }

    {
        std::vector<int32_t> ids(N);
        std::vector<int32_t> pos(N);
        for (size_t i = 0; i < N; i++) {
            ids[i] = tokens->data[i];
            pos[i] = (int32_t)i;
        }
        ggml_backend_tensor_set(input_ids, ids.data(), 0, N * sizeof(int32_t));
        ggml_backend_tensor_set(positions, pos.data(), 0, N * sizeof(int32_t));
        if (zero_types != nullptr) {
            std::vector<int32_t> zeros(N, 0);
            ggml_backend_tensor_set(zero_types, zeros.data(), 0, N * sizeof(int32_t));
        }

        const int32_t pool_idx = ctx->text_pool_cls ? 0 : (int32_t)(N - 1);
        ggml_backend_tensor_set(pool_token, &pool_idx, 0, sizeof(int32_t));

        if (attn_mask != nullptr) {
            std::vector<float> mask(N * N);
            for (size_t q = 0; q < N; q++) {
                const bool query_is_pad = tokens->data[q] == pad_id;
                for (size_t k = 0; k < N; k++) {
                    const bool masked = !query_is_pad && tokens->data[k] == pad_id;
                    mask[k + q * N] = masked ? -INFINITY : 0.0f;
                }
            }
            ggml_backend_tensor_set(attn_mask, mask.data(), 0, N * N * sizeof(float));
        }
    }

    /* 图构建上下文（只含元数据）。 */
    struct ggml_init_params params = {};
    params.mem_size = kClipGraphCtxSize;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context * ctx0 = ggml_init(params);
    if (ctx0 == nullptr) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(in_ctx);
        fprintf(stderr, "%s: 无法创建图构建上下文\n", __func__);
        return false;
    }

    struct ggml_tensor * embeddings = ggml_get_rows(ctx0, model.token_embeddings, input_ids);

    embeddings = ggml_add(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), embeddings);

    // BERT-style text towers use token-type embeddings (all zeros for the
    // single-segment CLIP input) and a layer norm on the embeddings.
    if (zero_types != nullptr) {
        embeddings = ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.token_type_embeddings, zero_types));
    }

    auto layer_norm = [&](struct ggml_tensor * x, struct ggml_tensor * w, struct ggml_tensor * b) {
        x = ggml_norm(ctx0, x, eps);
        return ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, w, x), x), ggml_repeat(ctx0, b, x));
    };

    if (ctx->text_bert && model.pre_ln_w && model.pre_ln_b) {
        embeddings = layer_norm(embeddings, model.pre_ln_w, model.pre_ln_b);
    }

    /* 因果掩码：严格上三角（key > query）为 -1e9，其余为 0。
     * 各层文本注意力复用同一张量，替代已从上游 ggml 移除的 ggml_diag_mask_inf。 */
    struct ggml_tensor * causal_mask = nullptr;
    if (!ctx->text_bert) {
        struct ggml_tensor * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, N), 1.0f);
        causal_mask = ggml_scale(ctx0, ggml_tri(ctx0, ones, GGML_TRI_TYPE_UPPER), -1e9f);
    }

    auto attention = [&](struct ggml_tensor * cur, const clip_layer & layer, const bool causal) {
        struct ggml_tensor * Q =
            ggml_add(ctx0, ggml_repeat(ctx0, layer.q_b, cur), ggml_mul_mat(ctx0, layer.q_w, cur));

        Q = ggml_scale_inplace(ctx0, Q, 1.0f / std::sqrt((float)d_head));
        Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, N, 1);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(ctx0, Q, d_head, N, n_head);

        struct ggml_tensor * K =
            ggml_add(ctx0, ggml_repeat(ctx0, layer.k_b, cur), ggml_mul_mat(ctx0, layer.k_w, cur));

        K = ggml_reshape_4d(ctx0, K, d_head, n_head, N, 1);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(ctx0, K, d_head, N, n_head);

        struct ggml_tensor * V =
            ggml_add(ctx0, ggml_repeat(ctx0, layer.v_b, cur), ggml_mul_mat(ctx0, layer.v_w, cur));
        V = ggml_reshape_4d(ctx0, V, d_head, n_head, N, 1);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
        V = ggml_reshape_3d(ctx0, V, N, d_head, n_head);

        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        if (causal) {
            // 上游 ggml 已移除 ggml_diag_mask_inf，改用带掩码的 soft_max：
            // 严格上三角（key > query）取 -1e9，等价于原来的 -inf 屏蔽。
            KQ = ggml_soft_max_ext(ctx0, KQ, causal_mask, 1.0f, 0.0f);
        } else {
            if (attn_mask) {
                KQ = ggml_add(ctx0, KQ, ggml_repeat(ctx0, attn_mask, KQ));
            }
            KQ = ggml_soft_max_inplace(ctx0, KQ);
        }

        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_reshape_4d(ctx0, KQV, d_head, N, n_head, 1);
        KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

        struct ggml_tensor * out = ggml_cpy(ctx0, KQV, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, N));
        return ggml_add(ctx0, ggml_repeat(ctx0, layer.o_b, out), ggml_mul_mat(ctx0, layer.o_w, out));
    };

    auto ffn = [&](struct ggml_tensor * cur, const clip_layer & layer) {
        cur = ggml_mul_mat(ctx0, layer.ff_i_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, layer.ff_i_b, cur), cur);

        if (ctx->text_use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, layer.ff_o_w, cur);
        return ggml_add(ctx0, ggml_repeat(ctx0, layer.ff_o_b, cur), cur);
    };

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const clip_layer & layer = model.layers[il];

        if (ctx->text_bert) {
            // BERT post-LN: x = LN(x + attn(x)); x = LN(x + ffn(x)).
            struct ggml_tensor * attn = attention(embeddings, layer, false);
            struct ggml_tensor * cur = layer_norm(ggml_add(ctx0, attn, embeddings), layer.ln_1_w, layer.ln_1_b);
            embeddings = layer_norm(ggml_add(ctx0, ffn(cur, layer), cur), layer.ln_2_w, layer.ln_2_b);
        } else {
            // Original CLIP text tower: pre-LN + causal attention.
            struct ggml_tensor * cur = layer_norm(embeddings, layer.ln_1_w, layer.ln_1_b);
            embeddings = ggml_add(ctx0, attention(cur, layer, true), embeddings);
            cur = layer_norm(embeddings, layer.ln_2_w, layer.ln_2_b);
            embeddings = ggml_add(ctx0, ffn(cur, layer), embeddings);
        }
    }

    // final -layer_norm. Present in CLIP text towers, absent in BERT towers.
    if (model.post_ln_w && model.post_ln_b) {
        embeddings = layer_norm(embeddings, model.post_ln_w, model.post_ln_b);
    }

    // CLS pooling for BERT-style towers, EOT (last token) pooling for CLIP.
    embeddings = ggml_get_rows(ctx0, embeddings, pool_token);

    // text projection
    embeddings = ggml_mul_mat(ctx0, model.projection, embeddings);

    ggml_set_name(embeddings, "check");

    // run the computation
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, kClipSchedGraphSize, false);
    ggml_build_forward_expand(gf, embeddings);
    const bool ok = clip_graph_compute(const_cast<clip_ctx *>(ctx), gf, n_threads);
    if (ok) {
        ggml_backend_tensor_get(embeddings, vec, 0, sizeof(float) * (size_t)projection_dim);
        if (normalize) {
            clip_normalize_rows(vec, 1, projection_dim);
        }
    }

    ggml_backend_buffer_free(in_buf);
    ggml_free(in_ctx);
    ggml_free(ctx0);
    return ok;
}

bool clip_text_encode_raw(const clip_ctx * ctx, const int n_threads, const clip_tokens * tokens, float * vec) {
    return clip_text_encode(ctx, n_threads, tokens, vec, false);
}

bool clip_image_encode(const clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec, const bool normalize) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    clip_image_f32_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec, normalize);
}

bool clip_image_batch_encode(const clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs, float * vec,
                             const bool normalize) {

    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size / patch_size) * (image_size / patch_size));
    const int num_positions = num_patches + 1;
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;
    const int n_layer = hparams.n_layer;
    const int n_intermediate = hparams.n_intermediate;
    const int projection_dim = hparams.projection_dim;
    const float eps = hparams.eps;
    int batch_size = imgs->size;

    /* 输入与常量张量：先建在 no_alloc 上下文，再由 CPU 后端缓冲分配并写入。 */
    struct ggml_init_params in_params = {};
    in_params.mem_size = 8u * 1024u * 1024u;
    in_params.no_alloc = true;
    struct ggml_context * in_ctx = ggml_init(in_params);
    if (in_ctx == nullptr) {
        fprintf(stderr, "%s: 无法创建输入上下文\n", __func__);
        return false;
    }

    struct ggml_tensor * inp_raw   = ggml_new_tensor_4d(in_ctx, GGML_TYPE_F32, image_size, image_size, 3, batch_size);
    struct ggml_tensor * positions = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, num_positions);
    struct ggml_tensor * cls       = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, batch_size);

    ggml_backend_t in_backend = ctx->cpu_backend != nullptr ? ctx->cpu_backend : ctx->backend;
    ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(in_ctx, in_backend);
    if (in_buf == nullptr) {
        fprintf(stderr, "%s: 无法分配输入缓冲\n", __func__);
        ggml_free(in_ctx);
        return false;
    }

    {
        const size_t n = (size_t)image_size * (size_t)image_size;
        std::vector<float> raw((size_t)batch_size * 3 * n);
        for (int b = 0; b < batch_size; b++) {
            const int nx = imgs->data[b].nx;
            const int ny = imgs->data[b].ny;
            GGML_ASSERT(nx == image_size && ny == image_size);
            for (int k = 0; k < 3; k++) {
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        raw[(size_t)b * 3 * n + (size_t)k * n + (size_t)y * nx + x] =
                            imgs->data[b].data[3 * (y * nx + x) + k];
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, raw.data(), 0, raw.size() * sizeof(float));

        std::vector<int32_t> pos(num_positions);
        for (int i = 0; i < num_positions; i++) {
            pos[i] = i;
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));

        std::vector<int32_t> cls_idx(batch_size);
        for (int b = 0; b < batch_size; b++) {
            cls_idx[b] = b * num_positions;
        }
        ggml_backend_tensor_set(cls, cls_idx.data(), 0, cls_idx.size() * sizeof(int32_t));
    }

    /* 图构建上下文（只含元数据）。 */
    struct ggml_init_params params = {};
    params.mem_size = kClipGraphCtxSize;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context * ctx0 = ggml_init(params);
    if (ctx0 == nullptr) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(in_ctx);
        fprintf(stderr, "%s: 无法创建图构建上下文\n", __func__);
        return false;
    }

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // concat class_embeddings and patch_embeddings
    struct ggml_tensor * cls_tokens = ggml_repeat(
        ctx0, model.class_embedding, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, 1, batch_size));
    struct ggml_tensor * embeddings = ggml_concat(ctx0, cls_tokens, inp, 1);

    embeddings =
        ggml_add(ctx0, embeddings, ggml_repeat(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), embeddings));

    // pre-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings, eps);

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.pre_ln_w, embeddings), embeddings),
                              ggml_repeat(ctx0, model.pre_ln_b, embeddings));
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_1_w, cur), cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur), ggml_mul_mat(ctx0, model.layers[il].q_w, cur));

            Q = ggml_scale_inplace(ctx0, Q, 1.0f / std::sqrt((float)d_head));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur), ggml_mul_mat(ctx0, model.layers[il].k_w, cur));

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur), ggml_mul_mat(ctx0, model.layers[il].v_w, cur));

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

            cur = ggml_cpy(ctx0, KQV, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size));
        }

        // attention output
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].o_b, cur), ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_2_w, cur), cur),
                           ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_i_b, cur), cur);

        if (ctx->use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_o_b, cur), cur);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // get the output of cls token, e.g., 0th index
    embeddings = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, embeddings, hidden_size, num_positions * batch_size), cls);

    // post-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings, eps);

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.post_ln_w, embeddings), embeddings),
                              ggml_repeat(ctx0, model.post_ln_b, embeddings));
    }

    // final visual projection
    embeddings = ggml_mul_mat(ctx0, model.projection, embeddings);

    ggml_set_name(embeddings, "check");

    // run the computation
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, kClipSchedGraphSize, false);
    ggml_build_forward_expand(gf, embeddings);
    const bool ok = clip_graph_compute(const_cast<clip_ctx *>(ctx), gf, n_threads);
    if (ok) {
        ggml_backend_tensor_get(embeddings, vec, 0, sizeof(float) * (size_t)projection_dim * (size_t)batch_size);
        if (normalize) {
            clip_normalize_rows(vec, batch_size, projection_dim);
        }
    }

    ggml_backend_buffer_free(in_buf);
    ggml_free(in_ctx);
    ggml_free(ctx0);
    return ok;
}

bool clip_image_encode_raw(const clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    return clip_image_encode(ctx, n_threads, img, vec, false);
}

int clip_set_backend(const char * name) {
    return clip_set_backend_preference(name);
}

const char * clip_model_backend_name(const struct clip_ctx * ctx) {
    if (ctx == nullptr) {
        return "";
    }
    return ctx->backend_name.c_str();
}

float clip_similarity_score(const float * vec1, const float * vec2, const int vec_dim) {
    float dot_product = 0.0;
    for (int i = 0; i < vec_dim; i++) {
        dot_product += vec1[i] * vec2[i];
    }

    return dot_product;
}

bool clip_compare_text_and_image(const clip_ctx * ctx, const int n_threads, const char * text, const clip_image_u8 * image,
                                 float * score) {
    if (!(ctx->has_text_encoder && ctx->has_vision_encoder)) {
        printf("clip_compare_text_and_image function can only be used with two-tower models\n");
        return false;
    }

    // prepare image and text vectors
    const int projection_dim = ctx->vision_model.hparams.projection_dim;
    std::vector<float> img_vec(projection_dim);
    std::vector<float> txt_vec(projection_dim);

    // tokenize and encode text
    clip_tokens tokens;
    if (!clip_tokenize(ctx, text, &tokens)) {
        return false;
    }

    if (!clip_text_encode(ctx, n_threads, &tokens, txt_vec.data(), true)) {
        return false;
    }

    // preprocess and encode image
    clip_image_f32 img_res;

    if (!clip_image_preprocess(ctx, image, &img_res)) {
        return false;
    }

    if (!clip_image_encode(ctx, n_threads, &img_res, img_vec.data(), true)) {
        return false;
    }

    // compute similarity
    *score = clip_similarity_score(img_vec.data(), txt_vec.data(), projection_dim);

    return true;
}

typedef struct {
    float score;
    int index;
} ScoreIndexPair;

int compare_scores(const void * a, const void * b) {
    const ScoreIndexPair * pair1 = (const ScoreIndexPair *)a;
    const ScoreIndexPair * pair2 = (const ScoreIndexPair *)b;

    if (pair1->score < pair2->score) {
        return 1;
    } else if (pair1->score > pair2->score) {
        return -1;
    } else {
        return 0;
    }
}

bool softmax_with_sorting(float * arr, const int length, float * sorted_scores, int * indices) {
    ScoreIndexPair * score_index_pairs = (ScoreIndexPair *)malloc(length * sizeof(ScoreIndexPair));
    if (!score_index_pairs) {
        return false;
    }

    // Calculate softmax probabilities

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        arr[i] = exp(arr[i]) + 1e-9;
        sum += arr[i];
    }

    for (int i = 0; i < length; i++) {
        arr[i] /= sum;
        score_index_pairs[i].score = arr[i];
        score_index_pairs[i].index = i;
    }

    // Sort scores in descending order
    qsort(score_index_pairs, length, sizeof(ScoreIndexPair), compare_scores);

    // Copy sorted scores and indices to the respective arrays
    for (int i = 0; i < length; i++) {
        sorted_scores[i] = score_index_pairs[i].score;
        indices[i] = score_index_pairs[i].index;
    }

    free(score_index_pairs);
    return true;
}

bool clip_zero_shot_label_image(struct clip_ctx * ctx, const int n_threads, const struct clip_image_u8 * input_img,
                                const char ** labels, const size_t n_labels, float * scores, int * indices) {
    if (!(ctx->has_text_encoder && ctx->has_vision_encoder)) {
        printf("clip_zero_shot_label_image function can only be used with two-tower models\n");
        return false;
    }

    // load the image
    clip_image_f32 img_res;

    const int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;

    clip_image_preprocess(ctx, input_img, &img_res);

    std::vector<float> img_vec(vec_dim);
    if (!clip_image_encode(ctx, n_threads, &img_res, img_vec.data(), false)) {
        return false;
    }

    // encode texts and compute similarities
    std::vector<float> txt_vec(vec_dim);
    std::vector<float> similarities(n_labels);

    for (int i = 0; i < (int)n_labels; i++) {
        const auto & text = labels[i];
        clip_tokens tokens;
        clip_tokenize(ctx, text, &tokens);
        clip_text_encode(ctx, n_threads, &tokens, txt_vec.data(), false);
        similarities[i] = clip_similarity_score(img_vec.data(), txt_vec.data(), vec_dim);
    }

    // apply softmax and sort scores
    softmax_with_sorting(similarities.data(), (int)n_labels, scores, indices);

    return true;
}

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {

    // 量化需要读取宿主内存中的权重，强制 CPU 后端。
    clip_set_backend_preference("cpu");

    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
    case 2:
        type = GGML_TYPE_Q4_0;
        break;
    case 3:
        type = GGML_TYPE_Q4_1;
        break;
    case 6:
        type = GGML_TYPE_Q5_0;
        break;
    case 7:
        type = GGML_TYPE_Q5_1;
        break;
    case 8:
        type = GGML_TYPE_Q8_0;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
        return false;
    };

    auto ctx_clip = clip_model_load(fname_inp, 2);
    const auto & ctx_src = ctx_clip->ctx_gguf;
    const auto & ctx_data = ctx_clip->ctx;

    auto ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_src);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_src);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        gguf_add_tensor(ctx_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    for (size_t i = 0; i < meta_size; ++i) {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };

    std::vector<uint8_t> read_data(512);
    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    std::vector<int64_t> hist_all(1 << 4, 0);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto & s : k_names) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (ggml_n_dims(cur) == 2);

        if (quantize) {
            new_type = type;
            const size_t n_elms = ggml_nelements(cur);
            float * f32_data;

            switch (cur->type) {
            case GGML_TYPE_F32:
                f32_data = (float *)cur->data;
                break;
            case GGML_TYPE_F16:
                if (conv_buf.size() < n_elms) {
                    conv_buf.resize(n_elms);
                }
                for (int j = 0; j < n_elms; ++j) {
                    conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                }
                f32_data = (float *)conv_buf.data();
                break;
            default:
                printf("Please use an input file in f32 or f16\n");
                return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            std::vector<int64_t> hist_cur(1 << 4, 0);

            new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, (int64_t)(n_elms / cur->ne[0]), cur->ne[0],
                                           nullptr);

            for (int j = 0; j < hist_cur.size(); ++j) {
                hist_all[j] += hist_cur[j];
            }
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data);
        fout.write((const char *)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
        for (int j = 0; j < pad; ++j) {
            fout.put(0);
        }

        printf("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
               orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *)meta.data(), meta_size);

    fout.close();

    clip_free(ctx_clip);
    gguf_free(ctx_out);

    {
        printf("%s: original size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quantized size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); ++i) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (size_t i = 0; i < hist_all.size(); ++i) {
            printf("%5.3f ", hist_all[i] / (float)sum_all);
        }
        printf("\n");
    }

    return true;
}

struct clip_text_hparams * clip_get_text_hparams(struct clip_ctx * ctx) { return &ctx->text_model.hparams; }
struct clip_vision_hparams * clip_get_vision_hparams(struct clip_ctx * ctx) { return &ctx->vision_model.hparams; }
