// test-expert-cache — unit tests for the MoE expert slot pool.
//
// The failure modes here are silent ones: a slot handed out twice would make two
// experts alias the same weights, which produces fluent, wrong output rather than
// a crash. So the invariant that matters most is that resident experts always map
// to distinct slots.

#include "expert_cache.h"

#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

// every resident expert occupies its own slot, and every slot is in range
static void check_no_aliasing(const moe::expert_cache & c, int n_layer, int n_expert) {
    std::set<int32_t> used;
    for (int l = 0; l < n_layer; ++l) {
        for (int e = 0; e < n_expert; ++e) {
            const int32_t s = c.peek(l, e);
            if (s == moe::SLOT_NONE) { continue; }
            CHECK(s >= 0 && (size_t) s < c.capacity(), "slot in range");
            CHECK(used.insert(s).second, "slot not handed out twice");
        }
    }
    CHECK(used.size() == c.size(), "resident count matches distinct slots");
}

int main() {
    auto nocopy = [](int, int, size_t) { return true; };

    printf("basic residency\n");
    {
        moe::expert_cache c(4, 1024, nocopy);
        CHECK(c.find(0, 7) == moe::SLOT_NONE, "cold lookup misses");
        const int32_t s = c.admit(0, 7);
        CHECK(s != moe::SLOT_NONE, "admit returns a slot");
        CHECK(c.find(0, 7) == s, "warm lookup hits the same slot");
        CHECK(c.size() == 1, "one resident");
        CHECK(c.admit(0, 7) == s, "re-admitting is idempotent");
        CHECK(c.size() == 1, "still one resident");
    }

    printf("layer and expert are distinct keys\n");
    {
        moe::expert_cache c(8, 1024, nocopy);
        const int32_t a = c.admit(1, 0);
        const int32_t b = c.admit(0, 1);
        CHECK(a != b, "(1,0) and (0,1) are different experts");
        check_no_aliasing(c, 4, 4);
    }

    printf("eviction is least-recently-used\n");
    {
        moe::expert_cache c(3, 1024, nocopy);
        c.admit(0, 0); c.admit(0, 1); c.admit(0, 2);
        c.find(0, 0);                    // touch 0 -> 1 is now the oldest
        c.admit(0, 3);                   // must evict expert 1
        CHECK(c.peek(0, 1) == moe::SLOT_NONE, "LRU victim evicted");
        CHECK(c.peek(0, 0) != moe::SLOT_NONE, "recently used survived");
        CHECK(c.peek(0, 2) != moe::SLOT_NONE, "untouched but newer survived");
        CHECK(c.peek(0, 3) != moe::SLOT_NONE, "new arrival resident");
        CHECK(c.size() == 3, "capacity respected");
        check_no_aliasing(c, 1, 8);
    }

    printf("evicted slots are reused, not leaked\n");
    {
        moe::expert_cache c(2, 1024, nocopy);
        for (int e = 0; e < 64; ++e) {
            c.admit(0, e);
            CHECK(c.size() <= 2, "never exceeds capacity");
            check_no_aliasing(c, 1, 64);
        }
        CHECK(c.stats().evictions == 62, "evicted exactly what it had to");
    }

    printf("a failed copy leaves no phantom resident\n");
    {
        bool allow = true;
        moe::expert_cache c(2, 1024, [&](int, int, size_t) { return allow; });
        c.admit(0, 0);
        allow = false;
        CHECK(c.admit(0, 1) == moe::SLOT_NONE, "failed admit reports failure");
        CHECK(c.peek(0, 1) == moe::SLOT_NONE, "failed admit is not resident");
        allow = true;
        CHECK(c.admit(0, 2) != moe::SLOT_NONE, "the rolled-back slot is reusable");
        check_no_aliasing(c, 1, 8);
    }

    printf("zero-capacity pool degrades to all-miss\n");
    {
        moe::expert_cache c(0, 1024, nocopy);
        CHECK(c.admit(0, 0) == moe::SLOT_NONE, "cannot admit into nothing");
        CHECK(c.find(0, 0) == moe::SLOT_NONE, "always misses");
        CHECK(c.size() == 0, "stays empty");
    }

    printf("resolve fills a whole routing step\n");
    {
        moe::expert_cache c(4, 1024, nocopy);
        c.admit(3, 10); c.admit(3, 20);
        const int32_t ids[4] = { 10, 99, 20, 98 };
        int32_t slots[4];
        c.resolve(3, ids, 4, slots);
        CHECK(slots[0] != moe::SLOT_NONE, "resident id resolves");
        CHECK(slots[1] == moe::SLOT_NONE, "absent id does not");
        CHECK(slots[2] != moe::SLOT_NONE, "second resident resolves");
        CHECK(slots[3] == moe::SLOT_NONE, "second absent does not");
        CHECK(slots[0] != slots[2], "distinct experts, distinct slots");
    }

    printf("statistics add up\n");
    {
        moe::expert_cache c(2, 4096, nocopy);
        c.find(0, 0);              // miss
        c.admit(0, 0);
        c.find(0, 0);              // hit
        c.find(0, 1);              // miss
        const auto & s = c.stats();
        CHECK(s.lookups == 3, "lookups counted");
        CHECK(s.hits == 1 && s.misses == 2, "hits and misses counted");
        CHECK(s.admissions == 1, "admissions counted");
        CHECK(s.bytes_in == 4096, "admission traffic accounted");
    }

    printf("\n%s\n", failures ? "FAILED" : "all expert_cache tests passed");
    return failures ? 1 : 0;
}
