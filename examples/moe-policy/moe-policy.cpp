// moe-policy — replay a routing trace under different expert-placement policies.
//
// Answers, before any of the plumbing is built: given N slots of VRAM, which
// policy keeps the most experts resident, how much traffic does each miss cost,
// and what does that do to per-token latency on this machine's bandwidths.
//
//   llama-moe-histogram ... (with MOE_TRACE=trace.bin)   -> a trace
//   llama-moe-policy trace.bin [profile.csv]             -> the comparison
//
// The profile is optional and only used by the fixed and hybrid policies. Taking
// the profile from ONE corpus and the trace from ANOTHER is the honest test, and
// the one that matters for shifting workloads like agentic coding.

#include "expert_policy.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct trace {
    // one entry per (token, layer): the experts that layer routed to
    struct rec { uint32_t token; uint16_t layer; std::vector<uint16_t> experts; };
    std::vector<rec> recs;
    int n_layer  = 0;
    int n_expert = 0;
};

static bool load_trace(const char * path, trace & t) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open trace %s\n", path); return false; }

    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x454f4d32) {
        fprintf(stderr, "%s: bad magic (not a moe trace)\n", path);
        fclose(f);
        return false;
    }
    while (true) {
        uint16_t hdr[2];
        if (fread(hdr, sizeof(uint16_t), 2, f) != 2) { break; }
        const uint16_t layer = hdr[0];
        const uint16_t k     = hdr[1];
        if (k == 0 || k > 4096) { break; }
        uint32_t tok = 0;
        if (fread(&tok, sizeof(tok), 1, f) != 1) { break; }
        trace::rec r;
        r.token = tok;
        r.layer = layer;
        r.experts.resize(k);
        if (fread(r.experts.data(), sizeof(uint16_t), k, f) != k) { break; }
        for (uint16_t e : r.experts) { t.n_expert = std::max(t.n_expert, (int) e + 1); }
        t.n_layer = std::max(t.n_layer, (int) layer + 1);
        t.recs.push_back(std::move(r));
    }
    fclose(f);
    // The trace arrives layer-major because the eval callback fires per
    // (layer, batch). Decode visits every layer of one token before moving on,
    // so replaying in emission order would show a single layer's working set and
    // overstate the hit rate badly. Sort into true decode order.
    std::stable_sort(t.recs.begin(), t.recs.end(),
        [](const trace::rec & a, const trace::rec & b) {
            if (a.token != b.token) { return a.token < b.token; }
            return a.layer < b.layer;
        });
    return !t.recs.empty();
}

static std::unordered_map<moe::slot_t, uint64_t> load_profile(const char * path) {
    std::unordered_map<moe::slot_t, uint64_t> p;
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "warning: cannot open profile %s\n", path); return p; }
    char line[256];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return p; }   // header
    int layer, expert; long long count;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%d,%d,%lld", &layer, &expert, &count) == 3) {
            p[moe::make_slot(layer, expert)] = (uint64_t) count;
        }
    }
    fclose(f);
    return p;
}

// Replay the trace. One "step" is one (token, layer) routing event, which is the
// granularity a real implementation decides at.
static moe::stats run(const trace & t, const moe::config & cfg,
                      const std::unordered_map<moe::slot_t, uint64_t> & profile) {
    moe::placement p(cfg, profile);
    std::vector<moe::slot_t> needed;
    std::unordered_set<moe::slot_t> seen;

    for (const auto & r : t.recs) {
        needed.clear();
        seen.clear();
        for (uint16_t e : r.experts) {
            const moe::slot_t s = moe::make_slot(r.layer, e);
            if (seen.insert(s).second) { needed.push_back(s); }
        }
        p.step(needed);
    }
    return p.get_stats();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <trace.bin> [profile.csv]\n"
            "  env: MOE_GB_PER_SLOT, MOE_B_PCIE, MOE_B_HOST, MOE_B_VRAM,\n"
            "       MOE_CAPACITY_PCT (comma-separated, default 10,25,37.5,50)\n", argv[0]);
        return 1;
    }

    trace t;
    if (!load_trace(argv[1], t)) { return 1; }

    std::unordered_map<moe::slot_t, uint64_t> profile;
    if (argc > 2) { profile = load_profile(argv[2]); }

    const size_t n_slots = (size_t) t.n_layer * t.n_expert;

    moe::config base;
    if (const char * v = getenv("MOE_B_PCIE")) { base.b_pcie = atof(v); }
    if (const char * v = getenv("MOE_B_HOST")) { base.b_host = atof(v); }
    if (const char * v = getenv("MOE_B_VRAM")) { base.b_vram = atof(v); }
    base.gb_per_slot = getenv("MOE_GB_PER_SLOT") ? atof(getenv("MOE_GB_PER_SLOT")) : 0.0;
    base.cpu_budget  = getenv("MOE_CPU_BUDGET")  ? atof(getenv("MOE_CPU_BUDGET"))  : 1.0;
    base.ms_per_crossing = getenv("MOE_MS_CROSSING") ? atof(getenv("MOE_MS_CROSSING")) : 0.737;
    base.vram_gb       = getenv("MOE_VRAM_GB")       ? atof(getenv("MOE_VRAM_GB"))       : 0.0;
    base.vram_reserved = getenv("MOE_VRAM_RESERVED") ? atof(getenv("MOE_VRAM_RESERVED")) : 0.0;

    printf("trace   : %s\n", argv[1]);
    printf("events  : %zu (token,layer) routing decisions\n", t.recs.size());
    printf("model   : %d layers x %d experts = %zu slots\n", t.n_layer, t.n_expert, n_slots);
    printf("profile : %s (%zu slots)\n", argc > 2 ? argv[2] : "none", profile.size());
    printf("crossing : %.3f ms fixed per step that sends work to the CPU\n", base.ms_per_crossing);
    printf("bandwidth: PCIe %.1f  host %.1f (x%.2f budget = %.1f usable)  vram %.1f GB/s",
           base.b_pcie, base.b_host, base.cpu_budget, base.b_host * base.cpu_budget, base.b_vram);
    if (base.gb_per_slot > 0.0) { printf("   slot %.4f GB", base.gb_per_slot); }
    printf("\n");

    // If a VRAM budget is given, capacity is not a free parameter: the KV cache,
    // compute buffer and non-expert weights come out of the same 32 GiB first.
    if (base.vram_gb > 0.0 && base.gb_per_slot > 0.0) {
        const double avail = base.vram_gb - base.vram_reserved;
        const size_t fits  = avail > 0 ? (size_t) (avail / base.gb_per_slot) : 0;
        printf("vram     : %.1f GB total - %.1f reserved (KV + compute + non-expert) = %.1f for experts\n",
               base.vram_gb, base.vram_reserved, avail);
        printf("           -> %zu slots of %zu (%.1f%% capacity)\n",
               fits, n_slots, 100.0 * (double) fits / (double) n_slots);
    }

    std::vector<double> caps = { 10.0, 25.0, 37.5, 50.0 };
    if (const char * v = getenv("MOE_CAPACITY_PCT")) {
        caps.clear();
        for (char * tok = strtok(strdup(v), ","); tok; tok = strtok(nullptr, ",")) {
            caps.push_back(atof(tok));
        }
    }

    const struct { moe::policy_kind k; bool needs_profile; } policies[] = {
        { moe::policy_kind::cpu_only, false },
        { moe::policy_kind::fixed,    true  },
        { moe::policy_kind::lru,      false },
        { moe::policy_kind::hybrid,   true  },
    };

    for (double cap_pct : caps) {
        const size_t cap = (size_t) (n_slots * cap_pct / 100.0);
        printf("\n=== capacity %.1f%% (%zu of %zu slots) ===\n", cap_pct, cap, n_slots);
        printf("%-10s %9s %9s %9s %10s %10s %12s  %5s %5s\n",
               "policy", "hit rate", "fetched", "cpu", "PCIe GB", "CPU GB", "exposed ms/1k",
               "gpu%", "cpu%");

        for (const auto & pol : policies) {
            if (pol.needs_profile && profile.empty()) { continue; }

            moe::config cfg = base;
            cfg.policy   = pol.k;
            cfg.capacity = cap;
            cfg.balance  = moe::balance_kind::residual;

            const moe::stats s = run(t, cfg, profile);

            // exposed time per 1000 routing events, in ms - comparable across runs
            const double per_1k = s.steps ? 1000.0 * s.t_exposed / (double) s.steps * 1000.0 : 0.0;

            printf("%-10s %8.1f%% %9llu %9llu %10.1f %10.1f %12.2f  %5.0f%% %5.0f%%\n",
                   moe::policy_name(pol.k),
                   100.0 * s.hit_rate(),
                   (unsigned long long) s.fetched,
                   (unsigned long long) s.cpu_done,
                   s.gb_pcie, s.gb_cpu, per_1k,
                   100.0 * s.util_gpu(), 100.0 * s.util_cpu());
        }
    }

    printf("\n=== admission: does promoting CPU-computed experts help? (lru) ===\n");
    printf("%-22s %9s %10s %12s\n", "admission", "hit rate", "PCIe GB", "exposed ms/1k");
    for (int admit_cpu = 0; admit_cpu <= 1; ++admit_cpu) {
        moe::config cfg = base;
        cfg.policy   = moe::policy_kind::lru;
        cfg.capacity = (size_t) (n_slots * 0.375);
        cfg.balance  = moe::balance_kind::residual;
        cfg.admit_cpu_misses = admit_cpu != 0;
        const moe::stats s2 = run(t, cfg, profile);
        const double per_1k = s2.steps ? 1000.0 * s2.t_exposed / (double) s2.steps * 1000.0 : 0.0;
        printf("%-22s %8.1f%% %10.1f %12.2f\n",
               admit_cpu ? "fetch + cpu (ours)" : "fetch only (theirs)",
               100.0 * s2.hit_rate(), s2.gb_pcie, per_1k);
    }

    // how the miss split itself matters, at one capacity, on the best policy
    printf("\n=== miss handling (lru, capacity %.1f%%) ===\n", caps.empty() ? 37.5 : caps[caps.size()/2]);
    printf("%-10s %10s %10s %12s\n", "balance", "PCIe GB", "CPU GB", "exposed ms/1k");
    const struct { moe::balance_kind b; const char * n; } balances[] = {
        { moe::balance_kind::all_cpu,  "all_cpu"  },
        { moe::balance_kind::all_pcie, "all_pcie" },
        { moe::balance_kind::residual, "residual" },
        { moe::balance_kind::adaptive, "adaptive" },
    };
    const size_t cap = (size_t) (n_slots * (caps.empty() ? 37.5 : caps[caps.size()/2]) / 100.0);
    for (const auto & b : balances) {
        moe::config cfg = base;
        cfg.policy   = moe::policy_kind::lru;
        cfg.capacity = cap;
        cfg.balance  = b.b;
        const moe::stats s = run(t, cfg, profile);
        const double per_1k = s.steps ? 1000.0 * s.t_exposed / (double) s.steps * 1000.0 : 0.0;
        printf("%-10s %10.1f %10.1f %12.2f\n", b.n, s.gb_pcie, s.gb_cpu, per_1k);
    }

    return 0;
}
