// diskformer × llama.cpp: load a GGUF with its MoE expert tensors paged (WebGPU_Paged, NakliTechie/llama.cpp branch
// webgpu-disk-tier) and chat greedily. Built twice: natively against Dawn (the check against llama-completion) and with
// Emscripten for a browser worker, where the GGUF is a WORKERFS mount of a File and experts are read from it on demand.

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#    include <emscripten.h>
EM_JS(void, df_emit, (const char * piece), { if (Module.onPiece) Module.onPiece(UTF8ToString(Number(piece))); });
#else
static void df_emit(const char * piece) {
    fputs(piece, stdout);
    fflush(stdout);
}
#    define EMSCRIPTEN_KEEPALIVE
#endif

static llama_model *             g_model = nullptr;
static llama_context *           g_ctx   = nullptr;
static common_chat_templates_ptr g_tmpls;
static std::string               g_out;
static std::string               g_stats;

// The paged buffer type the WebGPU backend lists once GGML_WEBGPU_PAGED_SLOTS is set.
static ggml_backend_buffer_type_t df_paged_buft() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("WebGPU");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t * extra = get_extra ? get_extra(ggml_backend_reg_dev_get(reg, 0)) : nullptr;
    return extra ? extra[0] : nullptr;
}

// Loads `path`; experts keep `n_slots` per tensor on the GPU and are read back from the same file. 0 on success.
extern "C" EMSCRIPTEN_KEEPALIVE int df_load(const char * path, int n_slots, int n_ctx) {
    std::setlocale(LC_NUMERIC, "C");
    setenv("GGML_WEBGPU_PAGED_SLOTS", std::to_string(n_slots).c_str(), 1);
    setenv("GGML_WEBGPU_PAGED_FILE", path, 1);
    llama_backend_init();
    ggml_backend_load_all();

    ggml_backend_buffer_type_t paged = df_paged_buft();
    if (paged == nullptr) {
        fprintf(stderr, "df_load: no WebGPU_Paged buffer type (is this the webgpu-disk-tier build?)\n");
        return 1;
    }
    static llama_model_tensor_buft_override overrides[] = { { "exps\\.weight", nullptr }, { nullptr, nullptr } };
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

// One user turn, greedy, at most n_predict tokens. Pieces stream through df_emit; returns the whole reply.
extern "C" EMSCRIPTEN_KEEPALIVE const char * df_chat(const char * user, int n_predict) {
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = user;
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.messages              = { msg };
    inputs.add_generation_prompt = true;
    const std::string prompt     = common_chat_templates_apply(g_tmpls.get(), inputs).prompt;

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

extern "C" EMSCRIPTEN_KEEPALIVE const char * df_stats() {
    return g_stats.c_str();
}

#ifndef __EMSCRIPTEN__
// df-chat <model.gguf> <n_slots> <prompt> [n_predict]
int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf n_slots prompt [n_predict]\n", argv[0]);
        return 1;
    }
    if (int rc = df_load(argv[1], atoi(argv[2]), 1024)) {
        return rc;
    }
    df_chat(argv[3], argc > 4 ? atoi(argv[4]) : 64);
    fprintf(stderr, "\n%s\n", df_stats());
    return 0;
}
#endif
