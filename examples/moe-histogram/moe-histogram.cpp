// moe-histogram — count which experts an MoE model actually routes to.
//
// Captures the `ffn_moe_topk` tensor for every layer via the scheduler's eval
// callback and accumulates a per-(layer, expert) selection count over a corpus.
//
// The question it answers: is expert usage skewed enough that pinning the hot
// ones in VRAM and pushing the cold tail to CPU would work? A perfectly uniform
// router means the top 25% of experts carry 25% of the routing and static
// placement buys nothing. See FreeTokenLayers.md.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

struct histo {
    int n_layer  = 0;
    int n_expert = 0;
    // optional trace sink: counts alone cannot evaluate a dynamic policy, which
    // needs the ORDER of selections. Records are <u16 layer><u16 k><k x u16 expert>
    // per (token, layer), preceded by a small header.
    FILE * trace = nullptr;
    int64_t trace_tok = 0;
    // counts[il][expert]
    std::vector<std::vector<int64_t>> counts;
    int64_t n_selections = 0;

    void ensure(int il, int n_exp) {
        if (n_exp > n_expert) { n_expert = n_exp; }
        if (il >= (int) counts.size()) { counts.resize(il + 1); }
        if ((int) counts[il].size() < n_expert) { counts[il].resize(n_expert, 0); }
        n_layer = std::max(n_layer, il + 1);
    }
};

static bool moe_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * h = (histo *) user_data;

    if (ask) {
        // only the routing tensor, named "ffn_moe_topk-<il>"
        return strncmp(t->name, "ffn_moe_topk", 12) == 0;
    }

    if (t->type != GGML_TYPE_I32) {
        return true;
    }

    int il = -1;
    const char * dash = strrchr(t->name, '-');
    if (dash) { il = atoi(dash + 1); }
    if (il < 0) { return true; }

    // NOTE: ggml_argsort_top_k returns a VIEW of the full argsort output - k
    //       elements per row, but rows strided by the parent's nb[1] (n_expert
    //       wide). Reading ggml_nelements() contiguous values gets whole
    //       permutations instead of the top-k, in which every expert appears
    //       exactly once, which looks like a perfectly uniform router. Stride.
    const int64_t k    = t->ne[0];
    const int64_t rows = ggml_nrows(t);
    const size_t  nb1  = t->nb[1];
    if (k <= 0 || rows <= 0) { return true; }

    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(raw.data(), (const char *) t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    }

    int max_id = 0;
    for (int64_t j = 0; j < rows; ++j) {
        const int32_t * row = (const int32_t *) (raw.data() + (size_t) j * nb1);
        for (int64_t i = 0; i < k; ++i) { max_id = std::max(max_id, row[i]); }
    }
    h->ensure(il, max_id + 1);

    if (h->trace) {
        std::vector<uint16_t> rec;
        rec.reserve(2 + k);
        for (int64_t j = 0; j < rows; ++j) {
            const int32_t * row = (const int32_t *) (raw.data() + (size_t) j * nb1);
            rec.clear();
            rec.push_back((uint16_t) il);
            rec.push_back((uint16_t) k);
            for (int64_t i = 0; i < k; ++i) { rec.push_back((uint16_t) row[i]); }
            fwrite(rec.data(), sizeof(uint16_t), rec.size(), h->trace);
        }
    }

    for (int64_t j = 0; j < rows; ++j) {
        const int32_t * row = (const int32_t *) (raw.data() + (size_t) j * nb1);
        for (int64_t i = 0; i < k; ++i) {
            const int e = row[i];
            if (e >= 0 && e < (int) h->counts[il].size()) {
                h->counts[il][e]++;
                h->n_selections++;
            }
        }
    }
    return true;
}

// normalised entropy: 1.0 == perfectly uniform, 0.0 == one expert takes everything
static double norm_entropy(const std::vector<int64_t> & c) {
    int64_t tot = 0;
    for (auto v : c) { tot += v; }
    if (tot == 0) { return 0.0; }
    double H = 0.0;
    int used = 0;
    for (auto v : c) {
        if (v > 0) {
            const double p = (double) v / (double) tot;
            H -= p * std::log(p);
            used++;
        }
    }
    const double Hmax = std::log((double) c.size());
    (void) used;
    return Hmax > 0.0 ? H / Hmax : 0.0;
}

// what fraction of routing decisions the top `frac` of experts absorb
static double coverage(std::vector<int64_t> c, double frac) {
    int64_t tot = 0;
    for (auto v : c) { tot += v; }
    if (tot == 0) { return 0.0; }
    std::sort(c.begin(), c.end(), std::greater<int64_t>());
    const size_t k = std::max<size_t>(1, (size_t) (c.size() * frac));
    int64_t top = 0;
    for (size_t i = 0; i < k && i < c.size(); ++i) { top += c[i]; }
    return (double) top / (double) tot;
}

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    histo h;
    if (const char * tp = getenv("MOE_TRACE")) {
        h.trace = fopen(tp, "wb");
        if (h.trace) {
            // header: magic, n_layer and n_expert are patched in at the end
            const uint32_t magic = 0x454f4d31; // "1MOE"
            fwrite(&magic, sizeof(magic), 1, h.trace);
        }
    }
    params.cb_eval           = moe_cb;
    params.cb_eval_user_data = &h;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks =
        common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);

    if (toks.empty()) {
        LOG_ERR("%s: no input tokens - pass -f <file> or -p <prompt>\n", __func__);
        return 1;
    }

    const int n_batch = std::max(1, (int) params.n_batch);
    const int n_ctx   = llama_n_ctx(ctx);
    const int limit   = std::min<int>(toks.size(), params.n_predict > 0 ? params.n_predict : toks.size());

    LOG_INF("%s: %d tokens, n_batch=%d n_ctx=%d\n", __func__, limit, n_batch, n_ctx);

    int done = 0;
    while (done < limit) {
        const int n = std::min(n_batch, limit - done);
        // keep every chunk inside one context window; routing statistics do not
        // need continuity across the whole corpus
        if (done % (n_ctx - n_batch > 0 ? n_ctx - n_batch : n_ctx) == 0) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + done, n))) {
            LOG_ERR("%s: decode failed at %d\n", __func__, done);
            break;
        }
        done += n;
        if (done % (n_batch * 8) == 0) {
            LOG_INF("  %d / %d tokens\n", done, limit);
        }
    }

    // ---- report ----
    printf("\n=== MoE expert routing histogram ===\n");
    printf("layers=%d  experts=%d  selections=%lld\n\n",
           h.n_layer, h.n_expert, (long long) h.n_selections);

    if (h.n_layer == 0 || h.n_expert == 0) {
        printf("no routing tensors captured - is this actually an MoE model?\n");
        return 1;
    }

    printf("%-6s %8s %8s %8s %8s  %s\n",
           "layer", "entropy", "top10%", "top25%", "top50%", "hottest/coldest");
    std::vector<int64_t> all(h.n_expert, 0);
    double e_sum = 0.0; int e_n = 0;

    for (int il = 0; il < h.n_layer; ++il) {
        if (il >= (int) h.counts.size() || h.counts[il].empty()) { continue; }
        auto & c = h.counts[il];
        for (size_t i = 0; i < c.size() && i < all.size(); ++i) { all[i] += c[i]; }

        const double H  = norm_entropy(c);
        const int64_t mx = *std::max_element(c.begin(), c.end());
        const int64_t mn = *std::min_element(c.begin(), c.end());
        e_sum += H; e_n++;

        printf("%-6d %8.4f %7.1f%% %7.1f%% %7.1f%%  %lld / %lld\n",
               il, H,
               100.0 * coverage(c, 0.10),
               100.0 * coverage(c, 0.25),
               100.0 * coverage(c, 0.50),
               (long long) mx, (long long) mn);
    }

    // Residency is per-(layer, expert) SLOT, so the meaningful aggregate pools
    // pairs as distinct entities. Summing by expert index across layers would
    // conflate expert 5 of layer 0 with expert 5 of layer 30 and understate the
    // skew badly - different layers have different hot experts.
    std::vector<int64_t> pairs;
    pairs.reserve((size_t) h.n_layer * h.n_expert);
    int64_t zero_pairs = 0;
    for (int il = 0; il < h.n_layer; ++il) {
        if (il >= (int) h.counts.size()) { continue; }
        for (size_t e = 0; e < h.counts[il].size(); ++e) {
            pairs.push_back(h.counts[il][e]);
            if (h.counts[il][e] == 0) { zero_pairs++; }
        }
    }

    printf("\n--- aggregate over (layer, expert) slots ---\n");
    printf("mean per-layer normalised entropy : %.4f  (1.0 = perfectly uniform)\n", e_n ? e_sum / e_n : 0.0);
    printf("total slots                       : %zu\n", pairs.size());
    printf("never routed to                   : %lld  (%.1f%%)\n",
           (long long) zero_pairs, 100.0 * (double) zero_pairs / (double) std::max<size_t>(1, pairs.size()));

    printf("\ncache capacity -> routing covered (the hit rate a perfect oracle gets):\n");
    const double caps[] = { 0.10, 0.25, 0.375, 0.50, 0.75 };
    for (double c : caps) {
        printf("  %5.1f%% of slots resident -> %5.1f%% covered   (uniform would be %.1f%%)\n",
               100.0 * c, 100.0 * coverage(pairs, c), 100.0 * c);
    }

    // dump raw counts so this can be re-analysed without re-running
    if (const char * path = getenv("MOE_HISTOGRAM_CSV")) {
        FILE * f = fopen(path, "w");
        if (f) {
            fprintf(f, "layer,expert,count\n");
            for (int il = 0; il < h.n_layer; ++il) {
                if (il >= (int) h.counts.size()) { continue; }
                for (size_t e = 0; e < h.counts[il].size(); ++e) {
                    fprintf(f, "%d,%zu,%lld\n", il, e, (long long) h.counts[il][e]);
                }
            }
            fclose(f);
            printf("\nraw counts written to %s\n", path);
        }
    }

    const double c37 = coverage(pairs, 0.375);
    printf("\nverdict: ");
    if (c37 > 0.80) {
        printf("STRONGLY SKEWED - static hot/cold placement is worth building (Phase 1).\n");
    } else if (c37 > 0.60) {
        printf("SKEWED - static placement is worth trying; dynamic residency buys more.\n");
    } else {
        printf("NEAR-UNIFORM - static pinning is not worth it; only dynamic residency helps.\n");
    }

    if (h.trace) {
        fclose(h.trace);
        printf("routing trace written to %s\n", getenv("MOE_TRACE"));
    }

    llama_backend_free();
    return 0;
}
