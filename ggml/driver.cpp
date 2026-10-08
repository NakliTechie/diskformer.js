// diskformer × llama.cpp: load a GGUF with its MoE expert tensors paged (WebGPU_Paged, NakliTechie/llama.cpp branch
// webgpu-disk-tier) and chat greedily. Built twice: natively against Dawn (the check against llama-completion) and with
// Emscripten for a browser worker, where the GGUF is a WORKERFS mount of a File and experts are read from it on demand.

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __EMSCRIPTEN__
#    include <emscripten.h>
EM_JS(void, df_emit, (const char * piece), { if (Module.onPiece) Module.onPiece(UTF8ToString(Number(piece))); });
// n reads of the GGUF. With an OPFS sync access handle (Module.ggufHandle) each read lands straight in the heap,
// synchronously; otherwise the picked File (Module.ggufFile) is sliced, all reads in flight at once, and the worker
// suspends (JSPI) until they land. Pointers arrive as BigInt (memory64); heap views are taken after any await.
EM_ASYNC_JS(void, df_read_file_batch, (int n, const double * offsets, const double * sizes, char * const * dsts), {
    const off = Number(offsets) / 8, sz = Number(sizes) / 8, dp = Number(dsts) / 8;
    const handle = Module.ggufHandle;
    if (handle) {
        for (let i = 0; i < n; i++) {
            const dst = Number(HEAPU64[dp + i]), s = HEAPF64[sz + i];
            const got = handle.read(HEAPU8.subarray(dst, dst + s), { at: HEAPF64[off + i] });
            if (got !== s) throw new Error(`short OPFS read: ${got} of ${s}`);
        }
        return;
    }
    const file = Module.ggufFile;
    const reqs = [];
    for (let i = 0; i < n; i++) {
        const o = HEAPF64[off + i], s = HEAPF64[sz + i];
        reqs.push(file.slice(o, o + s).arrayBuffer());
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

// The page source: expert bytes come from the GGUF itself (tensor name -> absolute offset of its data).
struct df_gguf_source {
    std::string                             path;
    std::unordered_map<std::string, size_t> offsets;
};
static df_gguf_source g_src;

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
#ifdef __EMSCRIPTEN__
    std::vector<double> o(n), sz(n);
    for (size_t i = 0; i < n; i++) {
        o[i]  = (double) (src->offsets.at(names[i]) + offs[i]);
        sz[i] = (double) sizes[i];
    }
    df_read_file_batch((int) n, o.data(), sz.data(), (char * const *) dsts);
#else
    // a few threads, each with its own FILE
    const size_t             n_threads = std::min<size_t>(n, 8);
    std::vector<std::thread> pool;
    for (size_t t = 0; t < n_threads; t++) {
        pool.emplace_back([&, t]() {
            FILE * f = fopen(src->path.c_str(), "rb");
            for (size_t i = t; i < n; i += n_threads) {
                fseeko(f, (off_t) (src->offsets.at(names[i]) + offs[i]), SEEK_SET);
                if (fread(dsts[i], 1, sizes[i], f) != sizes[i]) {
                    abort();
                }
            }
            fclose(f);
        });
    }
    for (auto & th : pool) {
        th.join();
    }
#endif
}

#ifdef __EMSCRIPTEN__
// df_gpu.js: n ranges of the GGUF into GPU buffers through mapped staging buffers
extern "C" void df_upload_batch_js(void * device, int n, const double * offsets, const double * sizes,
                                   void * const * buffers, const double * dst_offsets);

static void df_src_upload_batch(void * ud, void * device, size_t n, const char * const * names, const size_t * offs,
                                const size_t * sizes, void * const * buffers, const size_t * dst_offs) {
    auto *              src = (df_gguf_source *) ud;
    std::vector<double> o(n), sz(n), d(n);
    for (size_t i = 0; i < n; i++) {
        o[i]  = (double) (src->offsets.at(names[i]) + offs[i]);
        sz[i] = (double) sizes[i];
        d[i]  = (double) dst_offs[i];
    }
    df_upload_batch_js(device, (int) n, o.data(), sz.data(), buffers, d.data());
}
#endif

static void df_src_read(void * ud, const char * name, size_t off, void * dst, size_t size) {
    void * d = dst;
    df_src_read_batch(ud, 1, &name, &off, &d, &size);
}

static bool df_src_open(const char * path) {
    gguf_init_params params = { true, nullptr };
    gguf_context *   gguf   = gguf_init_from_file(path, params);
    if (gguf == nullptr) {
        return false;
    }
    const size_t base = gguf_get_data_offset(gguf);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        g_src.offsets[gguf_get_tensor_name(gguf, i)] = base + gguf_get_tensor_offset(gguf, i);
    }
    gguf_free(gguf);
    g_src.path = path;
    return true;
}

static llama_model *             g_model = nullptr;
static llama_context *           g_ctx   = nullptr;
static common_chat_templates_ptr g_tmpls;
static std::string               g_out;
static std::string               g_stats;

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
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        if (std::regex_search(gguf_get_tensor_name(gguf, i), paged_re)) {
            paged_bytes += gguf_get_tensor_size(gguf, i);
        }
    }
    gguf_free(gguf);
    if (n_expert == 0 || paged_bytes == 0) {
        return (g_plan = "{\"error\":\"not a mixture-of-experts GGUF\"}").c_str();
    }

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
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = n_ctx;
    cp.n_batch              = 512;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        llama_model_free(model);
        return (g_plan = "{\"error\":\"cannot create a context\"}").c_str();
    }
    size_t dense = 0, kv = 0, compute = 0;
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
    llama_model_free(model);

    const size_t per_slot = paged_bytes / n_expert;
    // reserve: the real load's compute buffer measured 5.5 MiB above this simulation (Gemma 4 26B-A4B, experts paged
    // vs on the CPU), plus the backend's slot maps and parameter buffers
    const size_t reserve  = 32u << 20;
    const double room     = budget - (double) (dense + kv + compute + reserve);
    const long   slots    = room > 0 ? std::min<long>((long) (room / per_slot), n_expert) : 0;
    char         buf[512];
    snprintf(buf, sizeof(buf),
             "{\"budget\":%.0f,\"dense\":%zu,\"kv\":%zu,\"compute\":%zu,\"reserve\":%zu,\"per_slot\":%zu,\"n_expert\":%u,"
             "\"n_used\":%u,\"slots\":%ld%s}",
             budget, dense, kv, compute, reserve, per_slot, n_expert, n_used, slots,
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
    ggml_backend_buffer_type_t paged = df_paged_buft(n_slots);
    if (paged == nullptr) {
        fprintf(stderr, "df_load: no WebGPU_Paged buffer type (is this the webgpu-disk-tier build?)\n");
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
    cp.no_perf              = false;
    g_ctx                   = llama_init_from_model(g_model, cp);
    if (g_ctx == nullptr) {
        return 3;
    }
    g_tmpls = common_chat_templates_init(g_model, "");
    return 0;
}

// Greedy continuation of an already formatted prompt, at most n_predict tokens. Pieces stream through df_emit;
// returns the whole continuation.
extern "C" EMSCRIPTEN_KEEPALIVE const char * df_complete(const char * prompt, int n_predict) {
    std::vector<llama_token> tokens = common_tokenize(g_ctx, prompt, true, true);
    const llama_vocab *      vocab  = llama_model_get_vocab(g_model);
    llama_memory_clear(llama_get_memory(g_ctx), true);

    llama_sampler * smpl  = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(g_ctx);

    g_out.clear();
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
    for (int k = 0; k < n_predict && g_out != "[decode failed]"; k++) {
        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        if (k == 0) {
            t_first = ggml_time_us();
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
              ",\"decode_ms\":" + std::to_string((t1 - t_first) / 1000) + "}";
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    return g_out.c_str();
}

// One user turn through the model's chat template (jinja), greedy.
extern "C" EMSCRIPTEN_KEEPALIVE const char * df_chat(const char * user, int n_predict) {
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = user;
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.messages              = { msg };
    inputs.add_generation_prompt = true;
    const std::string prompt     = common_chat_templates_apply(g_tmpls.get(), inputs).prompt;
    return df_complete(prompt.c_str(), n_predict);
}

extern "C" EMSCRIPTEN_KEEPALIVE const char * df_stats() {
    return g_stats.c_str();
}

#ifndef __EMSCRIPTEN__
#    include <fstream>
#    include <nlohmann/json.hpp>

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
        fprintf(stderr, "usage: %s model.gguf (n_slots | <GB>G) (prompt [n_predict] | --gate refs.json)\n", argv[0]);
        return 1;
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
    if (std::string(argv[3]) != "--gate") {
        df_chat(argv[3], argc > 4 ? atoi(argv[4]) : 64);
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
