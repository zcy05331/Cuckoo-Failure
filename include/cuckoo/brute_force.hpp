// Brute-force failure counts for small instances, used to validate the algorithms.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace cuckoo {

// One possible candidate choice of an item: a bitmask over the M slots, and the
// number of raw sampling outcomes that produce it.
struct Choice {
    uint64_t mask;
    unsigned long long weight;
};

// One table. Distinct model: the C(M, d) d-subsets of [M]. Replacement model:
// every nonempty subset S with |S| <= d, weighted by the number of d-tuples
// whose set of values is S (surjections from [d] onto S).
inline std::vector<Choice> one_table_choices(int M, int d, bool replacement) {
    if (M < 1 || M > 62) throw std::invalid_argument("brute force supports 1..62 slots");
    std::vector<Choice> out;
    for (uint64_t mask = 1; mask < (1ull << M); ++mask) {
        const int s = __builtin_popcountll(mask);
        if (!replacement) {
            if (s == d) out.push_back({mask, 1});
            continue;
        }
        if (s > d) continue;
        long long w = 0, c = 1;  // c = C(s, i)
        for (int i = 0; i <= s; ++i) {
            long long p = 1;
            for (int t = 0; t < d; ++t) p *= (s - i);
            w += (i % 2 ? -1 : 1) * c * p;
            c = c * (s - i) / (i + 1);
        }
        out.push_back({mask, static_cast<unsigned long long>(w)});
    }
    return out;
}

// Separate tables of lengths m: one slot from each table (slots of table j are
// numbered after those of tables < j), Q = prod m_j choices of weight 1.
inline std::vector<Choice> separate_table_choices(const std::vector<int>& m) {
    int M = 0;
    for (int x : m) M += x;
    if (m.empty() || M > 62) throw std::invalid_argument("brute force supports 1..62 slots");
    std::vector<Choice> out;
    std::vector<int> cur(m.size(), 0);
    while (true) {
        uint64_t mask = 0;
        int off = 0;
        for (size_t j = 0; j < m.size(); ++j) {
            mask |= 1ull << (off + cur[j]);
            off += m[j];
        }
        out.push_back({mask, 1});
        size_t j = 0;
        while (j < m.size() && ++cur[j] == m[j]) {
            cur[j] = 0;
            ++j;
        }
        if (j == m.size()) break;
    }
    return out;
}

// Does the candidate graph have a matching that covers every item? (Kuhn)
struct Matcher {
    int M = 0;
    std::vector<int> owner;
    std::vector<char> seen;
    const std::vector<uint64_t>* items = nullptr;

    bool augment(int i) {
        uint64_t m = (*items)[i];
        while (m) {
            const int s = __builtin_ctzll(m);
            m &= m - 1;
            if (seen[s]) continue;
            seen[s] = 1;
            if (owner[s] < 0 || augment(owner[s])) {
                owner[s] = i;
                return true;
            }
        }
        return false;
    }
    bool full_matching(const std::vector<uint64_t>& it) {
        items = &it;
        owner.assign(M, -1);
        seen.assign(M, 0);
        for (size_t i = 0; i < it.size(); ++i) {
            std::fill(seen.begin(), seen.end(), 0);
            if (!augment(static_cast<int>(i))) return false;
        }
        return true;
    }
};

struct BruteResult {
    unsigned long long nbad;   // failing raw outcomes
    unsigned long long total;  // Q^n raw outcomes (the caller keeps this below 2^64)
};

// Counts the failing outcomes by enumerating multisets of choices (items are
// exchangeable) with multinomial weights.
inline BruteResult brute_force_count(int n, int M, const std::vector<Choice>& choices) {
    unsigned long long Q = 0;
    for (const Choice& c : choices) Q += c.weight;
    unsigned long long total = 1;
    for (int i = 0; i < n; ++i) total *= Q;
    std::vector<unsigned long long> fact(n + 1, 1);
    for (int i = 1; i <= n; ++i) fact[i] = fact[i - 1] * i;

    std::vector<int> idx(n);
    std::vector<uint64_t> items(n);
    Matcher matcher;
    matcher.M = M;
    unsigned long long nbad = 0;
    const int C = static_cast<int>(choices.size());
    std::function<void(int, int)> rec = [&](int pos, int start) {
        if (pos == n) {
            if (matcher.full_matching(items)) return;
            unsigned long long w = fact[n];
            for (int i = 0; i < n;) {
                int j = i;
                while (j < n && idx[j] == idx[i]) ++j;
                w /= fact[j - i];
                for (int t = i; t < j; ++t) w *= choices[idx[i]].weight;
                i = j;
            }
            nbad += w;
            return;
        }
        for (int c = start; c < C; ++c) {
            idx[pos] = c;
            items[pos] = choices[c].mask;
            rec(pos + 1, c);
        }
    };
    rec(0, 0);
    return {nbad, total};
}

}  // namespace cuckoo
