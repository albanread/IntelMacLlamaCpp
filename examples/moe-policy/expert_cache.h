// expert_cache.h — a VRAM slot pool for MoE experts, with LRU residency.
//
// The model's expert weights live in host RAM. The card holds a pool of fixed
// size slots, each big enough for one (layer, expert) pair. This tracks which
// expert is in which slot, admits and evicts, and produces the slot table a
// kernel needs to find an expert it was asked for.
//
// Deliberately free of ggml/Metal types so it can be unit tested on its own; the
// copy itself is injected as a callback. See FreeTokenLayers.md §4 for why this
// is feasible at all: experts are contiguous slices along dim 2 of the fused
// {n_embd, n_ff, n_expert} tensor, so a slot is a plain memcpy of a sub-range.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

namespace moe {

static constexpr int32_t SLOT_NONE = -1;   // not resident: the CPU must do it

struct cache_stats {
    uint64_t lookups    = 0;
    uint64_t hits       = 0;
    uint64_t misses     = 0;
    uint64_t admissions = 0;
    uint64_t evictions  = 0;
    uint64_t bytes_in   = 0;   // host -> device traffic caused by admissions

    double hit_rate() const { return lookups ? (double) hits / (double) lookups : 0.0; }
};

// Copies one expert's weights from host memory into slot `slot` of the pool.
// Returns false if the copy failed, in which case the admission is rolled back.
using copy_fn = std::function<bool(int layer, int expert, size_t slot)>;

class expert_cache {
public:
    // n_slots: how many experts fit in the pool. slot_bytes is informational,
    // used only to account for admission traffic.
    expert_cache(size_t n_slots, size_t slot_bytes, copy_fn copy)
        : n_slots_(n_slots), slot_bytes_(slot_bytes), copy_(std::move(copy)) {
        free_.reserve(n_slots_);
        for (size_t i = 0; i < n_slots_; ++i) {
            free_.push_back(n_slots_ - 1 - i);   // hand out 0,1,2,... in order
        }
    }

    // Where does this expert live? SLOT_NONE means it does not, and the caller
    // must either admit() it or route the work to the CPU. Counts as a lookup.
    int32_t find(int layer, int expert) {
        st_.lookups++;
        const uint64_t k = key(layer, expert);
        auto it = map_.find(k);
        if (it == map_.end()) {
            st_.misses++;
            return SLOT_NONE;
        }
        st_.hits++;
        lru_.splice(lru_.begin(), lru_, it->second.pos);   // most recently used
        return it->second.slot;
    }

    // Peek without disturbing LRU order or statistics. For diagnostics.
    int32_t peek(int layer, int expert) const {
        auto it = map_.find(key(layer, expert));
        return it == map_.end() ? SLOT_NONE : it->second.slot;
    }

    // Bring an expert into the pool, evicting the least recently used if full.
    // Returns its slot, or SLOT_NONE if the copy failed.
    int32_t admit(int layer, int expert) {
        const uint64_t k = key(layer, expert);
        auto it = map_.find(k);
        if (it != map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.pos);
            return it->second.slot;
        }

        size_t slot;
        if (!free_.empty()) {
            slot = free_.back();
            free_.pop_back();
        } else {
            if (lru_.empty()) { return SLOT_NONE; }     // pool of size zero
            const uint64_t victim = lru_.back();
            auto vit = map_.find(victim);
            slot = vit->second.slot;
            lru_.pop_back();
            map_.erase(vit);
            st_.evictions++;
        }

        if (copy_ && !copy_(layer, expert, slot)) {
            free_.push_back(slot);                      // roll back cleanly
            return SLOT_NONE;
        }

        lru_.push_front(k);
        map_.emplace(k, entry{ slot, lru_.begin() });
        st_.admissions++;
        st_.bytes_in += slot_bytes_;
        return (int32_t) slot;
    }

    // Resolve a whole routing step. Fills `slots` with one entry per id: a slot
    // index for residents, SLOT_NONE for the rest. Nothing is admitted here -
    // the caller decides what to promote, because on this hardware a miss is
    // usually cheaper to compute on the CPU than to wait for a transfer.
    void resolve(int layer, const int32_t * experts, size_t n, int32_t * slots) {
        for (size_t i = 0; i < n; ++i) {
            slots[i] = find(layer, experts[i]);
        }
    }

    size_t              size()     const { return map_.size(); }
    size_t              capacity() const { return n_slots_; }
    const cache_stats & stats()    const { return st_; }
    void                reset_stats()    { st_ = cache_stats{}; }

    // Least-recently-used first, for tests and diagnostics.
    std::vector<std::pair<int,int>> lru_order() const {
        std::vector<std::pair<int,int>> v;
        for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
            v.emplace_back((int) (*it >> 32), (int) (*it & 0xffffffffu));
        }
        return v;
    }

private:
    static uint64_t key(int layer, int expert) {
        return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    }

    struct entry {
        size_t slot;
        std::list<uint64_t>::iterator pos;
    };

    size_t   n_slots_;
    size_t   slot_bytes_;
    copy_fn  copy_;

    std::unordered_map<uint64_t, entry> map_;
    std::list<uint64_t>                 lru_;   // front = most recent
    std::vector<size_t>                 free_;
    cache_stats                         st_;
};

} // namespace moe
