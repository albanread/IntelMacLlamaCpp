// expert_policy.h — where MoE experts live, and who computes the misses.
//
// Standalone and dependency-free so it can be evaluated offline against routing
// traces first and wired into the MoE path later. The interface is deliberately
// the one llama.cpp would need: "here are the experts this step wants, tell me
// which are resident, which to fetch over PCIe, and which the CPU should do."
//
// See FreeTokenLayers.md. The bandwidth split follows arXiv:2608.16157.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace moe {

// a residency unit: one (layer, expert) pair, which is what a cache slot holds
using slot_t = uint32_t;

static inline slot_t make_slot(int layer, int expert) {
    return ((uint32_t) layer << 16) | (uint32_t) (expert & 0xffff);
}
static inline int slot_layer (slot_t s) { return (int) (s >> 16); }
static inline int slot_expert(slot_t s) { return (int) (s & 0xffff); }

enum class policy_kind {
    gpu_only,   // everything resident; the upper bound, needs VRAM we do not have
    cpu_only,   // nothing resident; what -ncmoe over every layer approximates
    fixed,      // residency chosen once from an offline profile, never changes
    lru,        // residency follows the router, evicting least-recently-used
    hybrid,     // a pinned core from the profile, the rest an LRU working set
};

inline const char * policy_name(policy_kind k) {
    switch (k) {
        case policy_kind::gpu_only: return "gpu_only";
        case policy_kind::cpu_only: return "cpu_only";
        case policy_kind::fixed:    return "fixed";
        case policy_kind::lru:      return "lru";
        case policy_kind::hybrid:   return "hybrid";
    }
    return "?";
}

// How to divide misses between the PCIe branch and the CPU branch.
enum class balance_kind {
    all_cpu,    // never fetch; the CPU does every miss
    all_pcie,   // never use the CPU; every miss is transferred
    residual,   // q = m * B_P / B_H  — arXiv:2608.16157
    adaptive,   // same split, but from bandwidths measured as we go
};

struct config {
    policy_kind  policy  = policy_kind::lru;
    balance_kind balance = balance_kind::residual;

    size_t capacity      = 0;     // resident slots; 0 means cpu_only in practice
    double pin_fraction  = 0.5;   // hybrid: share of capacity pinned from profile

    // GB/s. Defaults are this machine's estimates; measure before trusting them.
    double b_pcie = 12.0;         // host -> device transfer
    double b_host = 90.0;         // CPU expert processing out of system RAM
    double b_vram = 430.0;        // on-card expert processing

    double gb_per_slot = 0.0;     // one (layer, expert) pair's weights, in GB

    // adaptive balancing smoothing
    double ewma = 0.05;
};

struct decision {
    std::vector<slot_t> hit;    // resident — GPU computes, no transfer
    std::vector<slot_t> fetch;  // miss — transfer over PCIe, GPU computes
    std::vector<slot_t> cpu;    // miss — CPU computes from host RAM
};

struct stats {
    uint64_t steps      = 0;
    uint64_t requests   = 0;
    uint64_t hits       = 0;
    uint64_t misses     = 0;
    uint64_t fetched    = 0;
    uint64_t cpu_done   = 0;
    uint64_t evictions  = 0;

    double gb_pcie = 0.0;
    double gb_cpu  = 0.0;
    double gb_vram = 0.0;

    // seconds, modelled: the two miss branches run concurrently so only the
    // slower is exposed; resident work is on top of it
    double t_vram    = 0.0;
    double t_pcie    = 0.0;
    double t_cpu     = 0.0;
    double t_exposed = 0.0;

    double hit_rate() const { return requests ? (double) hits / (double) requests : 0.0; }
};

class placement {
public:
    placement(const config & cfg,
              const std::unordered_map<slot_t, uint64_t> & profile = {})
        : cfg_(cfg), b_pcie_(cfg.b_pcie), b_host_(cfg.b_host) {

        if (cfg_.policy == policy_kind::gpu_only) {
            unlimited_ = true;
            return;
        }
        if (cfg_.policy == policy_kind::cpu_only) {
            cfg_.capacity = 0;
            return;
        }

        // rank slots by profiled frequency; used by fixed and hybrid
        std::vector<std::pair<slot_t, uint64_t>> ranked(profile.begin(), profile.end());
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });

        size_t n_pin = 0;
        if (cfg_.policy == policy_kind::fixed) {
            n_pin = cfg_.capacity;
        } else if (cfg_.policy == policy_kind::hybrid) {
            n_pin = (size_t) (cfg_.capacity * cfg_.pin_fraction);
        }
        n_pin = std::min(n_pin, ranked.size());

        for (size_t i = 0; i < n_pin; ++i) {
            pinned_.insert(ranked[i].first);
        }
        // the LRU working set gets whatever capacity the pins do not take
        lru_capacity_ = cfg_.capacity > pinned_.size() ? cfg_.capacity - pinned_.size() : 0;
    }

    // Plan one step and commit the residency changes it implies.
    // `needed` is the set of experts this step routed to (deduplicated).
    decision step(const std::vector<slot_t> & needed) {
        decision d;
        st_.steps++;

        std::vector<slot_t> miss;
        for (slot_t s : needed) {
            st_.requests++;
            if (resident(s)) {
                st_.hits++;
                touch(s);
                d.hit.push_back(s);
            } else {
                st_.misses++;
                miss.push_back(s);
            }
        }

        const size_t m = miss.size();
        const size_t q = split(m);

        for (size_t i = 0; i < m; ++i) {
            if (i < q) {
                d.fetch.push_back(miss[i]);
                admit(miss[i]);          // fetched experts become resident
            } else {
                d.cpu.push_back(miss[i]);
            }
        }

        st_.fetched  += d.fetch.size();
        st_.cpu_done += d.cpu.size();

        account(d);
        return d;
    }

    const stats  & get_stats() const { return st_; }
    const config & get_config() const { return cfg_; }

    size_t resident_count() const { return pinned_.size() + lru_map_.size(); }

private:
    bool resident(slot_t s) const {
        if (unlimited_)             { return true; }
        if (pinned_.count(s))       { return true; }
        return lru_map_.find(s) != lru_map_.end();
    }

    void touch(slot_t s) {
        if (unlimited_ || pinned_.count(s)) { return; }
        auto it = lru_map_.find(s);
        if (it != lru_map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
        }
    }

    void admit(slot_t s) {
        if (unlimited_ || pinned_.count(s)) { return; }
        // `fixed` never changes its residency: a miss stays a miss forever
        if (cfg_.policy == policy_kind::fixed || lru_capacity_ == 0) { return; }

        while (lru_map_.size() >= lru_capacity_ && !lru_.empty()) {
            const slot_t victim = lru_.back();
            lru_.pop_back();
            lru_map_.erase(victim);
            st_.evictions++;
        }
        lru_.push_front(s);
        lru_map_[s] = lru_.begin();
    }

    // How many of `m` misses go over PCIe.
    //
    // Under load the DMA reads from host RAM, so it consumes host bandwidth the
    // CPU would otherwise use: the CPU branch gets the residual B_H - B_P.
    // Equalising the two branch times,
    //     q / B_P == (m - q) / (B_H - B_P)   =>   q = m * B_P / B_H
    // which is the paper's rule, and it falls out of the residual rather than
    // being an approximation of m * B_P / (B_P + B_H).
    size_t split(size_t m) const {
        if (m == 0) { return 0; }
        switch (cfg_.balance) {
            case balance_kind::all_cpu:  return 0;
            case balance_kind::all_pcie: return m;
            case balance_kind::residual:
            case balance_kind::adaptive: {
                const double bp = cfg_.balance == balance_kind::adaptive ? b_pcie_ : cfg_.b_pcie;
                const double bh = cfg_.balance == balance_kind::adaptive ? b_host_ : cfg_.b_host;
                if (bh <= 0.0) { return m; }
                const double q = (double) m * bp / bh;
                size_t qi = (size_t) (q + 0.5);
                return std::min(qi, m);
            }
        }
        return 0;
    }

    void account(const decision & d) {
        const double gb = cfg_.gb_per_slot;
        if (gb <= 0.0) { return; }

        const double gb_vram = gb * (double) (d.hit.size() + d.fetch.size());
        const double gb_pcie = gb * (double) d.fetch.size();
        const double gb_cpu  = gb * (double) d.cpu.size();

        st_.gb_vram += gb_vram;
        st_.gb_pcie += gb_pcie;
        st_.gb_cpu  += gb_cpu;

        const double t_vram = cfg_.b_vram  > 0 ? gb_vram / cfg_.b_vram : 0.0;
        const double t_pcie = cfg_.b_pcie  > 0 ? gb_pcie / cfg_.b_pcie : 0.0;
        // the CPU only gets what the transfer leaves it
        const double b_res  = std::max(1.0, cfg_.b_host - cfg_.b_pcie);
        const double t_cpu  = gb_cpu / b_res;

        st_.t_vram += t_vram;
        st_.t_pcie += t_pcie;
        st_.t_cpu  += t_cpu;
        // branches concurrent; resident work still has to happen
        st_.t_exposed += t_vram + std::max(t_pcie, t_cpu);

        if (cfg_.balance == balance_kind::adaptive) {
            // nudge the estimates toward whichever branch is running long, so a
            // wrong starting guess corrects itself instead of persisting
            if (t_pcie > 0.0 && t_cpu > 0.0) {
                const double r = t_pcie / t_cpu;
                b_pcie_ *= (1.0 - cfg_.ewma * (r - 1.0) / std::max(1.0, r));
                b_pcie_ = std::max(0.5, b_pcie_);
            }
        }
    }

    config cfg_;
    stats  st_;

    bool   unlimited_    = false;
    size_t lru_capacity_ = 0;

    std::unordered_map<slot_t, bool> pinned_map_;   // unused; kept for clarity
    std::unordered_map<slot_t, std::list<slot_t>::iterator> lru_map_;
    std::list<slot_t> lru_;

    struct pinned_set {
        std::unordered_map<slot_t, char> m;
        void   insert(slot_t s) { m[s] = 1; }
        size_t count (slot_t s) const { return m.count(s); }
        size_t size  () const { return m.size(); }
    } pinned_;

    // adaptive state
    double b_pcie_ = 0.0;
    double b_host_ = 0.0;
};

} // namespace moe
