// diskformer × llama.cpp: load a GGUF with its MoE expert tensors paged (WebGPU_Paged, NakliTechie/llama.cpp branch
// webgpu-diskformer) and chat greedily. Built twice: natively against Dawn (the check against llama-completion) and with
// Emscripten for a browser worker, where the GGUF is a WORKERFS mount of a File and experts are read from it on demand.

#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-ext.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __EMSCRIPTEN__
#    include <emscripten.h>
EM_JS(void, df_emit, (const char * piece), { if (Module.onPiece) Module.onPiece(UTF8ToString(Number(piece))); });
// n reads of the GGUF; files[i] picks the part of a split GGUF (Module.ggufParts, one per part). With an OPFS sync
// access handle (part.handle) each read lands straight in the heap, synchronously; otherwise the picked File
// (part.file) is sliced, all reads in flight at once, and the worker suspends (JSPI) until they land. Pointers arrive
// as BigInt (memory64); heap views are taken after any await.
EM_ASYNC_JS(void, df_read_file_batch, (int n, const double * files, const double * offsets, const double * sizes, char * const * dsts), {
    const fl = Number(files) / 8, off = Number(offsets) / 8, sz = Number(sizes) / 8, dp = Number(dsts) / 8;
    const parts = Module.ggufParts;
    if (parts.every((p) => p.handle)) {
        for (let i = 0; i < n; i++) {
            const dst = Number(HEAPU64[dp + i]), s = HEAPF64[sz + i];
            const got = parts[HEAPF64[fl + i]].handle.read(HEAPU8.subarray(dst, dst + s), { at: HEAPF64[off + i] });
            if (got !== s) throw new Error(`short OPFS read: ${got} of ${s}`);
        }
        return;
    }
    const reqs = [];
    for (let i = 0; i < n; i++) {
        const o = HEAPF64[off + i], s = HEAPF64[sz + i];
        reqs.push(parts[HEAPF64[fl + i]].file.slice(o, o + s).arrayBuffer());
    }
    const bufs = await Promise.all(reqs);
    for (let i = 0; i < n; i++) HEAPU8.set(new Uint8Array(bufs[i]), Number(HEAPU64[dp + i]));
});
#else
static void df_emit(const char * piece) {
    fputs(piece, stdout);
    fflush(stdout);
}
#    define EMSCRIPTEN_KEEPALIVE
#endif

// The page source: expert bytes come from the GGUF itself (tensor name -> which file, absolute offset of its data).
// A split GGUF (name-0000k-of-0000n.gguf, as published) is read part by part, never merged.
struct df_tensor_at {
    int    file;
    size_t off;
};
struct df_gguf_source {
    std::vector<std::string>                      paths;
    std::unordered_map<std::string, df_tensor_at> offsets;
};
static df_gguf_source g_src;
// Expert bytes read from the GGUF and read calls, since load: the page's "read from disk" figures.
static double g_read_bytes = 0, g_read_calls = 0;

static bool df_src_has(void * ud, const char * name, size_t) {
    return ((df_gguf_source *) ud)->offsets.count(name) != 0;
}

static void df_src_write(void *, const char * name, size_t, const void *, size_t) {
    fprintf(stderr, "df: tensor %s is not in the GGUF\n", name);
    abort();
}

static void df_src_read_batch(void * ud, size_t n, const char * const * names, const size_t * offs, void * const * dsts,
                              const size_t * sizes) {
    auto * src = (df_gguf_source *) ud;
    for (size_t i = 0; i < n; i++) {
        g_read_bytes += (double) sizes[i];
    }
    g_read_calls += (double) n;
#ifdef __EMSCRIPTEN__
    std::vector<double> f(n), o(n), sz(n);
    for (size_t i = 0; i < n; i++) {
        const df_tensor_at & at = src->offsets.at(names[i]);
        f[i]  = (double) at.file;
        o[i]  = (double) (at.off + offs[i]);
        sz[i] = (double) sizes[i];
    }
    df_read_file_batch((int) n, f.data(), o.data(), sz.data(), (char * const *) dsts);
#else
    // a few threads, each with its own FILE per part
    const size_t             n_threads = std::min<size_t>(n, 8);
    std::vector<std::thread> pool;
    for (size_t t = 0; t < n_threads; t++) {
        pool.emplace_back([&, t]() {
            std::vector<FILE *> f(src->paths.size(), nullptr);
            for (size_t i = t; i < n; i += n_threads) {
                const df_tensor_at & at = src->offsets.at(names[i]);
                if (f[at.file] == nullptr) {
                    f[at.file] = fopen(src->paths[at.file].c_str(), "rb");
                }
                fseeko(f[at.file], (off_t) (at.off + offs[i]), SEEK_SET);
                if (fread(dsts[i], 1, sizes[i], f[at.file]) != sizes[i]) {
                    abort();
                }
            }
            for (FILE * fp : f) {
                if (fp != nullptr) {
                    fclose(fp);
                }
            }
        });
    }
    for (auto & th : pool) {
        th.join();
    }
#endif
}

#ifdef __EMSCRIPTEN__
// df_gpu.js: n ranges of the GGUF into GPU buffers through mapped staging buffers
extern "C" void df_upload_batch_js(void * device, int n, const double * files, const double * offsets, const double * sizes,
                                   void * const * buffers, const double * dst_offsets);

static void df_src_upload_batch(void * ud, void * device, size_t n, const char * const * names, const size_t * offs,
                                const size_t * sizes, void * const * buffers, const size_t * dst_offs) {
    auto *              src = (df_gguf_source *) ud;
    std::vector<double> f(n), o(n), sz(n), d(n);
    g_read_calls += (double) n;
    for (size_t i = 0; i < n; i++) {
        g_read_bytes += (double) sizes[i];
        const df_tensor_at & at = src->offsets.at(names[i]);
        f[i]  = (double) at.file;
        o[i]  = (double) (at.off + offs[i]);
        sz[i] = (double) sizes[i];
        d[i]  = (double) dst_offs[i];
    }
    df_upload_batch_js(device, (int) n, f.data(), o.data(), sz.data(), buffers, d.data());
}
#endif

static void df_src_read(void * ud, const char * name, size_t off, void * dst, size_t size) {
    void * d = dst;
    df_src_read_batch(ud, 1, &name, &off, &d, &size);
}

static bool df_src_open_part(const std::string & path, int file) {
    gguf_init_params params = { true, nullptr };
    gguf_context *   gguf   = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        return false;
    }
    const size_t base = gguf_get_data_offset(gguf);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        g_src.offsets[gguf_get_tensor_name(gguf, i)] = { file, base + gguf_get_tensor_offset(gguf, i) };
    }
    gguf_free(gguf);
    g_src.paths.push_back(path);
    return true;
}

// path: the GGUF, or the first part of a split one (its siblings are found by llama.cpp's split naming)
static bool df_src_open(const char * path) {
    g_src = {};
    gguf_init_params params = { true, nullptr };
    gguf_context *   gguf   = gguf_init_from_file(path, params);
    if (gguf == nullptr) {
        return false;
    }
    const int64_t key   = gguf_find_key(gguf, "split.count");
    const int     count = key >= 0 ? (int) gguf_get_val_u16(gguf, key) : 1;
    gguf_free(gguf);
    if (count <= 1) {
        return df_src_open_part(path, 0);
    }
    char prefix[4096];
    if (llama_split_prefix(prefix, sizeof(prefix), path, 0, count) == 0) {
        fprintf(stderr, "df: %s is not named as part 1 of %d\n", path, count);
        return false;
    }
    for (int k = 0; k < count; k++) {
        char part[4096];
        llama_split_path(part, sizeof(part), prefix, k, count);
        if (!df_src_open_part(part, k)) {
            fprintf(stderr, "df: cannot read part %s\n", part);
            return false;
        }
    }
    return true;
}

static llama_model *             g_model = nullptr;
static llama_context *           g_ctx   = nullptr;
static common_chat_templates_ptr g_tmpls;
static std::string               g_out;
static std::string               g_stats;
static volatile bool             g_abort = false;  // set by df_abort() while a reply streams (JSPI yields between reads)

// Speculative decoding with a drafter GGUF (DFlash, DSpark, MTP, Eagle3 or a plain draft model: the type comes from the
// drafter's metadata, as llama.cpp's --spec-type auto does). df_set_draft names it before df_load; df_load then sizes
// the target context for verification and loads the drafter beside it; every reply goes through llama.cpp's own
// speculative loop (common/speculative.cpp), which stays greedy-identical to plain decoding.
static std::string                        g_draft_path;
static common_params                      g_spec_params;
static common_speculative_init_result_ptr g_spec_init;
static common_speculative *               g_spec    = nullptr;
// 3, llama.cpp's own default: replies stay identical to plain greedy (Nemotron 3.5 Bolt E4B + DFlash on WebGPU, 9/9).
// Longer drafts verify in bigger batches, whose WebGPU kernels round differently: llama-server itself drifts at 7 (7/9
// vs plain), and this loop more (5/9 vs llama-server at 7, 16% of drafts accepted against 65% at 3); DF_DRAFT_N_MAX
// overrides it for that open investigation.
static const int32_t                      DF_DRAFT_N_MAX = 3;

extern "C" EMSCRIPTEN_KEEPALIVE void df_set_draft(const char * path) { g_draft_path = path ? path : ""; }

// A LoRA adapter GGUF (llama.cpp's convert_lora_to_gguf.py output). df_set_lora names it and its scale before df_load;
// df_load loads it beside the model (llama_adapter_lora_init) and sets it on the context, as llama-server --lora does.
// Its tensors sit next to the base tensors they change, so an adapter on the dense layers leaves expert paging alone.
static std::string          g_lora_path;
static float                g_lora_scale = 1.0f;
static llama_adapter_lora * g_lora       = nullptr;

extern "C" EMSCRIPTEN_KEEPALIVE void df_set_lora(const char * path, double scale) {
    g_lora_path  = path ? path : "";
    g_lora_scale = (float) scale;
}

typedef ggml_backend_buffer_type_t (*df_paged_buft_fn)(ggml_backend_dev_t, uint32_t, const ggml_webgpu_page_source *);

// A WebGPU_Paged buffer type with n_slots experts per tensor on the GPU, reading from the GGUF source.
static ggml_backend_buffer_type_t df_paged_buft(int n_slots) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("WebGPU");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    auto make = (df_paged_buft_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_webgpu_paged_buffer_type");
    if (make == nullptr) {
        return nullptr;
    }
#ifdef __EMSCRIPTEN__
    static ggml_webgpu_page_source src = { &g_src, df_src_has, df_src_write, df_src_read, df_src_read_batch, df_src_upload_batch };
#else
    static ggml_webgpu_page_source src = { &g_src, df_src_has, df_src_write, df_src_read, df_src_read_batch, nullptr };
#endif
    return make(ggml_backend_reg_dev_get(reg, 0), (uint32_t) n_slots, &src);
}

static const char * const DF_PAGED = "exps\\.weight";  // tensors paged: the routed experts
static std::string        g_plan;

// Tokens per physical batch (llama.cpp's n_ubatch). The compute buffers scale with it (Gemma 4 26B-A4B: 542 MiB at
// 512, 135 at 128), and a smaller one halves prompt speed at 128, because each pass pages in its own experts. df_plan
// picks the largest that leaves room for one token's experts and df_load uses the same; DF_UBATCH fixes it instead.
static uint32_t g_ubatch = 0;
static void df_set_ubatch(llama_context_params & cp) {
    if (const char * u = getenv("DF_UBATCH")) {
        cp.n_ubatch = (uint32_t) atoi(u);
    } else if (g_ubatch) {
        cp.n_ubatch = g_ubatch;
    }
}

// How many expert slots per tensor fit a GPU budget, as JSON: {"budget","dense","kv","compute","reserve","per_slot","n_expert",
// "n_used","slots"} in bytes, or {"error": ...}. The GPU bill without experts (weights, KV cache, compute buffers) is
// measured the way llama.cpp's --fit does it: a load that only simulates allocations, experts kept off the GPU. A
// slot holds one expert of every paged tensor.
extern "C" EMSCRIPTEN_KEEPALIVE const char * df_plan(const char * path, double budget, int n_ctx) {
    std::setlocale(LC_NUMERIC, "C");
    llama_backend_init();
    ggml_backend_load_all();
    gguf_init_params gp   = { true, nullptr };
    gguf_context *   gguf = gguf_init_from_file(path, gp);
    if (gguf == nullptr) {
        return (g_plan = "{\"error\":\"cannot read the GGUF\"}").c_str();
    }
    const int64_t arch_key = gguf_find_key(gguf, "general.architecture");
    const std::string arch = arch_key >= 0 ? gguf_get_val_str(gguf, arch_key) : "";
    const int64_t ne_key   = gguf_find_key(gguf, (arch + ".expert_count").c_str());
    const int64_t nu_key   = gguf_find_key(gguf, (arch + ".expert_used_count").c_str());
    const uint32_t n_expert = ne_key >= 0 ? gguf_get_val_u32(gguf, ne_key) : 0;
    const uint32_t n_used   = nu_key >= 0 ? gguf_get_val_u32(gguf, nu_key) : 0;
    size_t paged_bytes = 0;
    const std::regex paged_re(DF_PAGED);
    const auto add_paged = [&](gguf_context * g) {
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            if (std::regex_search(gguf_get_tensor_name(g, i), paged_re)) {
                paged_bytes += gguf_get_tensor_size(g, i);
            }
        }
    };
    add_paged(gguf);
    // a split GGUF: the experts in the other parts count too
    const int64_t split_key = gguf_find_key(gguf, "split.count");
    const int     n_split   = split_key >= 0 ? (int) gguf_get_val_u16(gguf, split_key) : 1;
    gguf_free(gguf);
    char prefix[4096];
    if (n_split > 1 && llama_split_prefix(prefix, sizeof(prefix), path, 0, n_split) > 0) {
        for (int k = 1; k < n_split; k++) {
            char part[4096];
            llama_split_path(part, sizeof(part), prefix, k, n_split);
            gguf_context * g = gguf_init_from_file(part, gp);
            if (g == nullptr) {
                return (g_plan = std::string("{\"error\":\"cannot read ") + part + "\"}").c_str();
            }
            add_paged(g);
            gguf_free(g);
        }
    }
    // a dense model has nothing to page: the whole model must fit the budget, which this measures the same way
    const bool dense_only = n_expert == 0 || paged_bytes == 0;

    static llama_model_tensor_buft_override cpu[] = { { DF_PAGED, nullptr }, { nullptr, nullptr } };
    cpu[0].buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    llama_model_params mp    = llama_model_default_params();
    mp.n_gpu_layers          = 999;
    mp.tensor_buft_overrides = cpu;
    mp.no_alloc              = true;
    mp.use_extra_bufts       = false;
    mp.load_mode             = LLAMA_LOAD_MODE_NONE;
    llama_model * model      = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        return (g_plan = "{\"error\":\"cannot load the model\"}").c_str();
    }
    const size_t per_slot = dense_only ? 0 : paged_bytes / n_expert;
    // reserve: the real load's compute buffer measured 5.5 MiB above this simulation (Gemma 4 26B-A4B, experts paged
    // vs on the CPU), plus the backend's slot maps and parameter buffers
    const size_t reserve  = 32u << 20;
    // physical batch: 512 when the budget allows, else halved until one token's experts fit (DF_UBATCH: that one only)
    const char *          fixed = getenv("DF_UBATCH");
    std::vector<uint32_t> sizes = fixed ? std::vector<uint32_t>{ (uint32_t) atoi(fixed) } : std::vector<uint32_t>{ 512, 256, 128, 64 };
    size_t   dense = 0, kv = 0, compute = 0;
    long     slots = 0;
    uint32_t ubatch = 0;
    for (const uint32_t u : sizes) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                = n_ctx;
        cp.n_batch              = 512;
        cp.n_ubatch             = u;
        llama_context * ctx     = llama_init_from_model(model, cp);
        if (ctx == nullptr) {
            llama_model_free(model);
            return (g_plan = "{\"error\":\"cannot create a context\"}").c_str();
        }
        dense = kv = compute = 0;
        for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            const auto         type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dense += mb.model;
                kv += mb.context;
                compute += mb.compute;
            }
        }
        llama_free(ctx);
        ubatch            = u;
        const double room = budget - (double) (dense + kv + compute + reserve);
        if (dense_only) {
            slots = room >= 0 ? 0 : -1;
            if (slots == 0) {
                break;
            }
            continue;
        }
        slots             = room > 0 ? std::min<long>((long) (room / per_slot), n_expert) : 0;
        if (slots >= (long) n_used) {
            break;
        }
    }
    llama_model_free(model);
    g_ubatch = ubatch;
    // a drafter (df_set_draft) and a LoRA adapter (df_set_lora) sit on the GPU beside the target: count their weights
    // (a drafter's small KV and compute are in the reserve's margin)
    for (const std::string & side : { g_draft_path, g_lora_path }) {
        if (side.empty()) {
            continue;
        }
        if (gguf_context * g = gguf_init_from_file(side.c_str(), gp)) {
            size_t side_bytes = 0;
            for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
                side_bytes += gguf_get_tensor_size(g, i);
            }
            gguf_free(g);
            dense += side_bytes;
            const double room = budget - (double) (dense + kv + compute + reserve);
            slots = dense_only ? (room >= 0 ? 0 : -1) : (room > 0 ? std::min<long>((long) (room / per_slot), n_expert) : 0);
        }
    }
    if (dense_only) {
        char buf[384];
        snprintf(buf, sizeof(buf),
                 "{\"budget\":%.0f,\"dense\":%zu,\"kv\":%zu,\"compute\":%zu,\"reserve\":%zu,\"n_expert\":0,\"ubatch\":%u,"
                 "\"slots\":0%s}",
                 budget, dense, kv, compute, reserve, ubatch,
                 slots < 0 ? ",\"error\":\"the budget does not hold this dense model\"" : "");
        return (g_plan = buf).c_str();
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"budget\":%.0f,\"dense\":%zu,\"kv\":%zu,\"compute\":%zu,\"reserve\":%zu,\"per_slot\":%zu,\"n_expert\":%u,"
             "\"n_used\":%u,\"ubatch\":%u,\"slots\":%ld%s}",
             budget, dense, kv, compute, reserve, per_slot, n_expert, n_used, ubatch, slots,
             slots < (long) n_used ? ",\"error\":\"the budget does not hold the dense part and one token's experts\"" : "");
    return (g_plan = buf).c_str();
}

// Loads `path`; experts keep `n_slots` per tensor on the GPU and are read back from the same file. 0 on success.
extern "C" EMSCRIPTEN_KEEPALIVE int df_load(const char * path, int n_slots, int n_ctx) {
    std::setlocale(LC_NUMERIC, "C");
    setenv("GGML_WEBGPU_PAGED_STATS", "1", 0);
    llama_backend_init();
    ggml_backend_load_all();
    if (!df_src_open(path)) {
        fprintf(stderr, "df_load: cannot read GGUF %s\n", path);
        return 1;
    }
    ggml_backend_buffer_type_t paged = df_paged_buft(std::max(n_slots, 1));  // a dense model pages nothing
    if (paged == nullptr) {
        fprintf(stderr, "df_load: no WebGPU_Paged buffer type (is this the webgpu-diskformer build?)\n");
        return 1;
    }
    static llama_model_tensor_buft_override overrides[] = { { DF_PAGED, nullptr }, { nullptr, nullptr } };
    overrides[0].buft = paged;

    llama_model_params mp     = llama_model_default_params();
    mp.n_gpu_layers           = 999;
    mp.tensor_buft_overrides  = overrides;
#ifdef __EMSCRIPTEN__
    mp.load_mode = LLAMA_LOAD_MODE_NONE;  // WORKERFS has no mmap: plain reads
#endif
    g_model = llama_model_load_from_file(path, mp);
    if (g_model == nullptr) {
        return 2;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = n_ctx;
    cp.n_batch              = 512;
    df_set_ubatch(cp);
    cp.no_perf              = false;
    // DF_FLASH_ATTN=0 / 1 forces flash attention off / on (default: llama.cpp decides), to compare with a reference
    // recorded either way
    if (const char * fa = getenv("DF_FLASH_ATTN")) {
        cp.flash_attn_type = atoi(fa) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    common_params & sp = g_spec_params;
    if (!g_draft_path.empty()) {
        sp                                 = common_params();
        sp.speculative.types               = common_speculative_types_from_gguf(g_draft_path);
        if (sp.speculative.types.empty()) {
            fprintf(stderr, "df_load: %s is not a drafter llama.cpp knows (no dflash, dspark or MTP metadata)\n", g_draft_path.c_str());
            return 4;
        }
        sp.speculative.draft.mparams.path  = g_draft_path;
        sp.speculative.draft.n_max         = getenv("DF_DRAFT_N_MAX") ? atoi(getenv("DF_DRAFT_N_MAX")) : DF_DRAFT_N_MAX;
        sp.n_ctx                           = n_ctx;
        sp.n_batch                         = (int32_t) cp.n_batch;
        sp.n_ubatch                        = (int32_t) cp.n_ubatch;
        sp.n_gpu_layers                    = 999;
        sp.sampling.temp                   = 0.0f;
#ifdef __EMSCRIPTEN__
        sp.load_mode = LLAMA_LOAD_MODE_NONE;
#endif
        // the target verifies id_last plus a whole draft in one batch, every position an output; the drafter's context
        // is built from sp and needs the same limits (a block drafter such as DFlash outputs every block position:
        // with the default of 1 per sequence its drafts were wrong, and the replies drifted from plain greedy)
        const auto lim           = common_speculative_get_output_limits(cp.n_batch, 1, common_speculative_n_max(&sp.speculative));
        sp.n_outputs_max         = lim.total;
        sp.n_outputs_max_per_seq = lim.per_seq;
        cp.n_outputs_max         = (uint32_t) lim.total;
        cp.n_outputs_max_per_seq = (uint32_t) lim.per_seq;
        cp.n_rs_seq              = sp.speculative.need_n_rs_seq();
        // as llama.cpp's speculative tools build it (common_context_params_to_llama): a window-sized SWA cache, so a
        // partly accepted draft rolls back through a checkpoint
        cp.swa_full              = sp.swa_full;
    }
    g_ctx                   = llama_init_from_model(g_model, cp);
    if (g_ctx == nullptr) {
        return 3;
    }
    if (!g_lora_path.empty()) {
        g_lora = llama_adapter_lora_init(g_model, g_lora_path.c_str());
        if (g_lora == nullptr || llama_set_adapters_lora(g_ctx, &g_lora, 1, &g_lora_scale) != 0) {
            fprintf(stderr, "df_load: cannot apply the LoRA adapter %s\n", g_lora_path.c_str());
            return 7;
        }
    }
    if (!g_draft_path.empty()) {
        common_params dp = common_base_params_to_speculative(sp);
        g_spec_init      = common_speculative_init_from_params(dp, g_model, g_ctx);
        if (!g_spec_init || !g_spec_init->context()) {
            fprintf(stderr, "df_load: cannot load the drafter %s\n", g_draft_path.c_str());
            return 5;
        }
        sp.speculative.draft.ctx_tgt = g_ctx;
        sp.speculative.draft.ctx_dft = g_spec_init->context();
        g_spec                       = common_speculative_init(sp.speculative, 1);
        if (g_spec == nullptr) {
            fprintf(stderr, "df_load: speculative decoding failed to start\n");
            return 6;
        }
    }
    // a template llama.cpp's jinja cannot parse must not take the model down: raw prompts (df_complete) still work,
    // and chat calls report the failure
    try {
        g_tmpls = common_chat_templates_init(g_model, "");
    } catch (const std::exception & e) {
        fprintf(stderr, "df_load: chat template failed: %s\n", e.what());
        g_tmpls.reset();
    }
    return 0;
}

// Greedy continuation of an already formatted prompt, at most n_predict tokens. Pieces stream through df_emit;
// returns the whole continuation.
static const char * df_complete_tokens(std::vector<llama_token> tokens, int n_predict);

extern "C" EMSCRIPTEN_KEEPALIVE const char * df_complete(const char * prompt, int n_predict) {
    return df_complete_tokens(common_tokenize(g_ctx, prompt, true, true), n_predict);
}

// A rendered chat prompt, part by part: template text with its special tokens, user text as plain text (a user who
// types "<|im_start|>" gets those characters, not the control token).
static std::vector<llama_token> df_tokenize_prompt(const common_chat_prompt & prompt) {
    std::vector<llama_token> tokens;
    bool                     first = true;
    for (const auto & part : prompt.parts) {
        const auto t = common_tokenize(g_ctx, part.text, first, !part.is_input);
        tokens.insert(tokens.end(), t.begin(), t.end());
        first = false;
    }
    return tokens;
}

// Tokens the next reply may not produce (logit bias -inf), set per reply: with reasoning off, the thought-opening token,
// so a model whose template always reasons (LFM2.5) cannot open a new block after the empty one.
static std::vector<llama_logit_bias> g_banned;

// The same reply through llama.cpp's speculative loop (examples/speculative-simple): the drafter proposes up to its
// block, the target verifies id_last plus the draft in one batch, the greedy sampler keeps the agreeing prefix; a
// context that cannot drop part of a sequence restores a checkpoint instead. Stats add drafted and accepted counts.
static const char * df_complete_tokens_spec(std::vector<llama_token> tokens, int n_predict) {
    const llama_vocab * vocab   = llama_model_get_vocab(g_model);
    llama_context *     ctx_dft = g_spec_params.speculative.draft.ctx_dft;
    const llama_seq_id  seq_id  = 0;
    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_memory_clear(llama_get_memory(ctx_dft), true);

    g_out.clear();
    g_abort            = false;
    const int  n_ctx   = (int) llama_n_ctx(g_ctx);
    bool       ctx_full = tokens.size() + 1 >= (size_t) n_ctx || tokens.empty();
    const bool ckpt_tgt = common_context_can_seq_rm(g_ctx) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    common_params_sampling sparams = g_spec_params.sampling;
    sparams.logit_bias.insert(sparams.logit_bias.end(), g_banned.begin(), g_banned.end());
    common_sampler_ptr smpl(common_sampler_init(g_model, sparams));
    common_batch       batch(g_ctx);
    const int64_t      t0 = ggml_time_us();
    int64_t            t_first = t0;
    int                n_decode = 0, n_drafted = 0, n_accept = 0, n_steps = 0;
    int64_t            us_draft = 0, us_verify = 0, us_feed = 0;   // per phase, for the stats
    llama_tokens       prompt_tgt, draft;
    llama_token        id_last = 0;
    int                n_past  = 0;
    bool               failed  = false;

    if (!ctx_full) {
        // the prompt but its last token, in n_batch chunks; each chunk also feeds the drafter
        for (size_t i = 0; i + 1 < tokens.size() && !failed; i += 512) {
            const size_t n = std::min<size_t>(512, tokens.size() - 1 - i);
            batch.clear();
            for (size_t j = 0; j < n; j++) {
                batch.add(tokens[i + j], (llama_pos) (i + j), seq_id, false);
            }
            failed = llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) || !common_speculative_process(g_spec, batch);
        }
        id_last = tokens.back();
        prompt_tgt.assign(tokens.begin(), tokens.end() - 1);
        n_past = (int) prompt_tgt.size();
        common_speculative_begin(g_spec, seq_id, prompt_tgt);
        t_first = ggml_time_us();
    }

    common_prompt_checkpoint ckpt;
    bool                     done = ctx_full || failed;
    while (!done) {
        if (draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(), llama_memory_seq_pos_min(llama_get_memory(g_ctx), seq_id),
                            llama_memory_seq_pos_max(llama_get_memory(g_ctx), seq_id));
            if (ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            const int n_draft_max = std::max(0, std::min(n_ctx - n_past - 2, n_predict - n_decode - 1));
            common_speculative_get_draft_params(g_spec, seq_id) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
            const int64_t td = ggml_time_us();
            common_speculative_draft(g_spec);
            us_draft += ggml_time_us() - td;
            if (!draft.empty() && ckpt_tgt) {
                ckpt.update_tgt(g_ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (ckpt_dft) {
                GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            }
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
        }

        batch.clear();
        batch.add(id_last, n_past++, seq_id, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch.add(draft[i], n_past + (llama_pos) i, seq_id, true);
        }
        const int64_t tv = ggml_time_us();
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            failed = true;
            break;
        }
        const int64_t tf = ggml_time_us();
        us_verify += tf - tv;
        if (!common_speculative_process(g_spec, batch)) {
            failed = true;
            break;
        }
        us_feed += ggml_time_us() - tf;
        n_steps++;
        common_sampler_ptr smpl_save;
        if (ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        const size_t n_draft = draft.size();
        auto         ids     = common_sampler_sample_and_accept_n(smpl.get(), g_ctx, draft);
        GGML_ASSERT(!ids.empty());

        if (ckpt_tgt && ids.size() - 1 < n_draft) {
            // partial acceptance without partial removal: back to the checkpoint, the accepted tokens as the draft
            draft = std::move(ids);
            GGML_ASSERT(ckpt.load_tgt(g_ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            llama_memory_seq_rm(llama_get_memory(g_ctx), seq_id, ckpt.pos_max + 1, -1);
            GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            prompt_tgt.resize(ckpt.n_tokens);
            smpl   = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(g_spec, seq_id, (uint16_t) (ids.size() - 1));
        n_past    += (int) ids.size() - 1;
        n_drafted += (int) n_draft;
        n_accept  += (int) ids.size() - 1;
        for (size_t i = 0; i < ids.size() && !done; ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last) || n_decode >= n_predict) {
                done = true;
                break;
            }
            const std::string piece = common_token_to_piece(g_ctx, id_last);
            g_out += piece;
            df_emit(piece.c_str());
            n_decode++;
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(g_ctx), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        if (g_abort) {
            break;
        }
        if (n_past + 2 >= n_ctx) {
            ctx_full = true;
            break;
        }
        done = done || n_decode >= n_predict;
    }
    if (failed) {
        g_out = "[decode failed]";
    }
    const int64_t t1 = ggml_time_us();
    g_stats = "{\"prompt_tokens\":" + std::to_string(tokens.size()) +
              ",\"prompt_ms\":" + std::to_string((t_first - t0) / 1000) +
              ",\"decode_tokens\":" + std::to_string(n_decode) +
              ",\"decode_ms\":" + std::to_string((t1 - t_first) / 1000) +
              ",\"draft_steps\":" + std::to_string(n_steps) +
              ",\"drafted\":" + std::to_string(n_drafted) + ",\"accepted\":" + std::to_string(n_accept) +
              ",\"draft_ms\":" + std::to_string(us_draft / 1000) + ",\"verify_ms\":" + std::to_string(us_verify / 1000) +
              ",\"feed_ms\":" + std::to_string(us_feed / 1000) +
              ",\"n_ctx\":" + std::to_string(n_ctx) + ",\"n_pos\":" + std::to_string(n_past) +
              ",\"aborted\":" + (g_abort ? "true" : "false") + ",\"context_full\":" + (ctx_full ? "true" : "false") + "}";
    return g_out.c_str();
}

static const char * df_complete_tokens(std::vector<llama_token> tokens, int n_predict) {
    if (g_spec) {
        return df_complete_tokens_spec(std::move(tokens), n_predict);
    }
    const llama_vocab *      vocab  = llama_model_get_vocab(g_model);
    llama_memory_clear(llama_get_memory(g_ctx), true);

    llama_sampler * smpl  = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!g_banned.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), (int32_t) g_banned.size(), g_banned.data()));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(g_ctx);

    g_out.clear();
    g_abort                = false;
    const int     n_ctx    = (int) llama_n_ctx(g_ctx);
    const double  read0    = g_read_bytes;
    double        read_decode0 = 0;
    bool          ctx_full = tokens.size() + 1 >= (size_t) n_ctx;
    if (ctx_full) {
        tokens.clear();
    }
    const int64_t t0       = ggml_time_us();
    int64_t       t_first  = 0;
    int           n_pos    = 0;
    int           n_decode = 0;
    // prompt in n_batch chunks, then one token at a time
    for (size_t i = 0; i < tokens.size(); i += 512) {
        const size_t n = std::min<size_t>(512, tokens.size() - i);
        llama_batch_ext_clear(batch);
        for (size_t j = 0; j < n; j++) {
            const int32_t   idx = llama_batch_ext_add_token(batch, 0, tokens[i + j]);
            const llama_pos pos = n_pos + (llama_pos) j;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n - 1, true);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            g_out = "[decode failed]";
            break;
        }
        n_pos += n;
    }
    for (int k = 0; k < n_predict && !ctx_full && g_out != "[decode failed]"; k++) {
        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        if (k == 0) {
            t_first      = ggml_time_us();
            read_decode0 = g_read_bytes;
        }
        if (g_abort || n_pos + 1 >= n_ctx) {
            ctx_full = !g_abort;
            break;
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char buf[256];
        int  n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        std::string piece(buf, n > 0 ? n : 0);
        g_out += piece;
        df_emit(piece.c_str());
        llama_batch_ext_clear(batch);
        const int32_t   idx = llama_batch_ext_add_token(batch, 0, id);
        const llama_pos pos = n_pos;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, 0, true);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            break;
        }
        n_pos++;
        n_decode++;
    }
    const int64_t t1 = ggml_time_us();
    g_stats          = "{\"prompt_tokens\":" + std::to_string(tokens.size()) +
              ",\"prompt_ms\":" + std::to_string((t_first - t0) / 1000) +
              ",\"decode_tokens\":" + std::to_string(n_decode) +
              ",\"decode_ms\":" + std::to_string((t1 - t_first) / 1000) +
              ",\"prompt_read_bytes\":" + std::to_string((long long) (read_decode0 - read0)) +
              ",\"decode_read_bytes\":" + std::to_string((long long) (g_read_bytes - read_decode0)) +
              ",\"read_bytes_total\":" + std::to_string((long long) g_read_bytes) +
              ",\"n_ctx\":" + std::to_string(n_ctx) + ",\"n_pos\":" + std::to_string(n_pos) +
              ",\"aborted\":" + (g_abort ? "true" : "false") + ",\"context_full\":" + (ctx_full ? "true" : "false") + "}";
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    return g_out.c_str();
}

// One user turn through the model's chat template (jinja), greedy.
// The chat template applied to `inputs`, then a greedy reply; a template that fails reports why instead of throwing.
// With reasoning off, a template that opens a thought block whatever enable_thinking says (LFM2 / LFM2.5 end the
// generation prompt with "<think>") gets it closed empty, as Qwen3's and Gemma 4's templates render reasoning off.
static const char * df_complete_chat(const common_chat_templates_inputs & inputs, int n_predict) {
    std::vector<llama_token> tokens;
    try {
        if (!g_tmpls) {
            throw std::runtime_error("no chat template (it failed to load)");
        }
        common_chat_prompt prompt = common_chat_templates_apply(g_tmpls.get(), inputs).prompt;
        if (!inputs.enable_thinking && !prompt.parts.empty() && !prompt.parts.back().is_input) {
            std::string & last = prompt.parts.back().text;
            const size_t  end  = last.find_last_not_of(" \t\n");
            if (end != std::string::npos && end + 1 >= 7 && last.compare(end + 1 - 7, 7, "<think>") == 0) {
                last.erase(end + 1);
                last += "\n</think>\n\n";
                // the empty block alone is not enough: LFM2.5 still opened a new one (1 of 9 gate conversations)
                const auto open = common_tokenize(g_ctx, "<think>", false, true);
                if (open.size() == 1) {
                    g_banned.push_back({ open[0], -INFINITY });
                }
            }
        }
        tokens = df_tokenize_prompt(prompt);
    } catch (const std::exception & e) {
        fprintf(stderr, "df_chat: chat template failed: %s\n", e.what());
        g_out = std::string("[chat template failed: ") + e.what() + "]";
        g_banned.clear();
        return g_out.c_str();
    }
    const char * out = df_complete_tokens(std::move(tokens), n_predict);
    g_banned.clear();
    return out;
}

extern "C" EMSCRIPTEN_KEEPALIVE const char * df_chat(const char * user, int n_predict) {
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = user;
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.messages              = { msg };
    inputs.add_generation_prompt = true;
    return df_complete_chat(inputs, n_predict);
}

// A whole conversation through the model's chat template (jinja), greedy: messages as JSON [{"role","content"}, ...].
// The KV cache is rebuilt from the full prompt each turn. thinking 0 renders the template with reasoning off (an empty
// thought block for Gemma 4, no <think> for Qwen3); 1 keeps the template's default.
extern "C" EMSCRIPTEN_KEEPALIVE const char * df_chat_messages(const char * messages_json, int n_predict, int thinking) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja       = true;
    inputs.enable_thinking = thinking != 0;
    for (const auto & m : nlohmann::ordered_json::parse(messages_json)) {
        common_chat_msg msg;
        msg.role    = m.at("role").get<std::string>();
        msg.content = m.at("content").get<std::string>();
        inputs.messages.push_back(msg);
    }
    inputs.add_generation_prompt = true;
    return df_complete_chat(inputs, n_predict);
}

// Stops the reply in progress at the next token.
extern "C" EMSCRIPTEN_KEEPALIVE void df_abort() { g_abort = true; }

extern "C" EMSCRIPTEN_KEEPALIVE const char * df_stats() {
    return g_stats.c_str();
}

#ifndef __EMSCRIPTEN__
#    include <fstream>

// The reference reply minus the generation prefix it repeats (the longest suffix of the prompt it starts with).
static std::string df_expected(const std::string & prompt, const std::string & content) {
    for (size_t k = std::min(prompt.size(), content.size()); k > 0; k--) {
        if (prompt.compare(prompt.size() - k, k, content, 0, k) == 0) {
            return content.substr(k);
        }
    }
    return content;
}

// df-chat <model.gguf> <n_slots> <prompt> [n_predict]
// df-chat <model.gguf> <n_slots> --gate <refs.json>   replay llama.cpp's replies; exit 1 on any difference
int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf (n_slots | <GB>G) (prompt [n_predict [--no-think]] | --gate refs.json | --record refs.json out.json)\n", argv[0]);
        return 1;
    }
    // DF_DRAFT=<drafter.gguf>: speculative decoding with that drafter (DFlash, DSpark, MTP, ...)
    if (const char * d = getenv("DF_DRAFT")) {
        df_set_draft(d);
    }
    // DF_LORA=<adapter.gguf> [DF_LORA_SCALE=1]: a LoRA adapter on the model, as llama-server --lora (--lora-scaled)
    if (const char * l = getenv("DF_LORA")) {
        df_set_lora(l, getenv("DF_LORA_SCALE") ? atof(getenv("DF_LORA_SCALE")) : 1.0);
    }
    // n_slots, or a GPU budget in GB ("2.5G") that df_plan turns into slots
    int               n_slots = atoi(argv[2]);
    const std::string slots_arg = argv[2];
    if (!slots_arg.empty() && slots_arg.back() == 'G') {
        nlohmann::json plan = nlohmann::json::parse(df_plan(argv[1], atof(argv[2]) * 1e9, 1024));
        fprintf(stderr, "df_plan: %s\n", plan.dump().c_str());
        if (plan.contains("error")) {
            return 1;
        }
        n_slots = plan["slots"];
    }
    if (int rc = df_load(argv[1], n_slots, 1024)) {
        return rc;
    }
    // --record: the conversations of another model's refs, rendered with this model's template and answered here
    // (greedy, 64 tokens each), written as refs for gate.html; for a model llama-server cannot page
    // (a split GGUF), so its native paged replies are the reference the browser is checked against
    if (std::string(argv[3]) == "--record" && argc > 5) {
        std::ifstream  in(argv[4]);
        nlohmann::json refs = nlohmann::json::parse(in), out = { { "conversations", nlohmann::json::array() } };
        out["about"] = std::string("df-chat --record, native WebGPU with experts paged (") + std::to_string(n_slots) + " slots), greedy, on " + argv[1];
        for (const auto & c : refs["conversations"]) {
            common_chat_templates_inputs inputs;
            inputs.use_jinja = true;
            for (const auto & m : c["messages"]) {
                common_chat_msg msg;
                msg.role    = m.at("role").get<std::string>();
                msg.content = m.at("content").get<std::string>();
                inputs.messages.push_back(msg);
            }
            inputs.add_generation_prompt = true;
            std::string prompt;
            for (const auto & part : common_chat_templates_apply(g_tmpls.get(), inputs).prompt.parts) {
                prompt += part.text;
            }
            const int n = 64;  // every reply the same length (the source refs stopped where their own model did)
            out["conversations"].push_back({ { "name", c["name"] }, { "messages", c["messages"] }, { "prompt", prompt },
                                             { "content", std::string(df_complete(prompt.c_str(), n)) }, { "tokens", n } });
            fprintf(stderr, "%s done\n", c["name"].get<std::string>().c_str());
        }
        std::ofstream(argv[5]) << out.dump(1);
        return 0;
    }
    if (std::string(argv[3]) != "--gate") {
        const int n_predict = argc > 4 ? atoi(argv[4]) : 64;
        if (argc > 5 && std::string(argv[5]) == "--no-think") {
            // the browser's chat path with thinking: false
            df_chat_messages(nlohmann::json::array({ { { "role", "user" }, { "content", argv[3] } } }).dump().c_str(), n_predict, 0);
        } else {
            df_chat(argv[3], n_predict);
        }
        fprintf(stderr, "\n%s\n", df_stats());
        llama_free(g_ctx);
        llama_model_free(g_model);
        return 0;
    }
    std::ifstream  in(argv[4]);
    nlohmann::json refs = nlohmann::json::parse(in);
    int            same = 0, total = 0;
    for (const auto & c : refs["conversations"]) {
        const std::string prompt   = c["prompt"];
        const std::string expected = df_expected(prompt, c["content"]);
        const std::string got      = df_complete(prompt.c_str(), c["tokens"].get<int>());
        size_t            at       = 0;
        while (at < got.size() && at < expected.size() && got[at] == expected[at]) {
            at++;
        }
        const bool ok = got == expected;
        same += ok;
        total++;
        printf("%s %-22s %s  %s\n", ok ? "OK  " : "DIFF", c["name"].get<std::string>().c_str(), df_stats(),
               ok ? "" : ("first difference at byte " + std::to_string(at) + " of " + std::to_string(expected.size()))
                             .c_str());
        fflush(stdout);
    }
    printf("%d/%d identical to the reference\n", same, total);
    return same == total ? 0 : 1;
}
#endif
