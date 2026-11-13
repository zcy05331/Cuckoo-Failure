// One-table d-choice cuckoo hashing: exact failure counts (paper, Section 3).
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "modint.hpp"
#include "ntt.hpp"

namespace cuckoo {

// Sampling model of the candidate sets. g(v) is the number of candidate sets
// that fit inside v given slots: C(v, d) for a uniform d-subset (eq. (6) with
// v = M - b), v^d for d independent draws with replacement (Section 3.4).
struct OneTableModel {
    int d = 2;
    bool replacement = false;
    Mint g(int v, const Factorials& fac) const {
        if (v < 0) return Mint();
        return replacement ? Mint(v).pow(d) : fac.binom(v, d);
    }
};

// phi(a, b, r) of Beniamini's coefficient formula, paper eq. (4).
inline Mint phi(const Factorials& fac, long long a, long long b, long long r) {
    if (b <= 0) return fac.binom(a - 1, r);
    return -(fac.binom(a - b - 1, r - b) * fac.binom(r - 1, b - 1));
}

inline void check_one_table_args(int n, int M, const OneTableModel& model) {
    if (model.d < 1 || n < 1 || M <= n) throw std::invalid_argument("one table: need d >= 1 and 1 <= n < M");
}

// ---------------------------------------------------------------------------
// Algorithm 1 (Section 3.2): the forward recurrence over block boundaries.
// F[l][b] is the total weight of the prefixes ending at boundary (l, b): the
// first l left vertices are placed and the next block has a forbidden set of
// size b. Transitions use tau(l, b; r, c) of eq. (11); eq. (14) fills the
// table and eq. (15) adds the final block. Only (0, 0) and boundaries with
// z <= l < b can carry weight, so the loops run over O(n^2) states with
// O(n^2) predecessors each: O(n^4) operations, O(M^2) cells.
// ---------------------------------------------------------------------------
inline Mint one_table_forward(int n, int M, OneTableModel model) {
    check_one_table_args(n, M, model);
    const int z = M - n;  // number of dummy items
    Factorials fac(M);

    // hpow[b][u] = h(b)^u with h(b) = g(M - b), eq. (6)
    std::vector<std::vector<Mint>> hpow(M + 1, std::vector<Mint>(n + 1));
    for (int b = 0; b <= M; ++b) {
        const Mint h = model.g(M - b, fac);
        hpow[b][0] = Mint::one();
        for (int u = 1; u <= n; ++u) hpow[b][u] = hpow[b][u - 1] * h;
    }
    auto nu = [z](int l) { return std::max(0, l - z); };
    auto tau = [&](int l, int b, int r, int c) {
        const int u = r - std::max(l, z);
        return phi(fac, c - l, b - l, r - l) * fac.binom(M - b, c - b) * fac.binom(n - nu(l), u) * hpow[b][u];
    };

    std::vector<std::vector<Mint>> F(M, std::vector<Mint>(M + 1));
    F[0][0] = Mint::one();
    for (int r = z; r < M; ++r) {
        for (int c = r + 1; c <= M; ++c) {
            Mint s = F[0][0] * tau(0, 0, r, c);
            for (int l = z; l < r; ++l)
                for (int b = l + 1; b < c; ++b)
                    if (!F[l][b].is_zero()) s += F[l][b] * tau(l, b, r, c);
            F[r][c] = s;
        }
    }
    Mint total;
    for (int l = 0; l < M; ++l)
        for (int b = 0; b <= M; ++b)
            if (!F[l][b].is_zero()) total += F[l][b] * fac.binom(M - l - 1, M - b) * hpow[b][n - nu(l)];
    return total;
}

// ---------------------------------------------------------------------------
// Algorithms 2 and 3 (Section 3.3): the suffix table.
// V[p][v] is the total weight of all block sequences that start at a boundary
// with p remaining (real) items and v available slots, 0 <= v < p. In the
// coordinates p = M - l, v = M - b, the multiplier (11) for l >= z reads
//   tau = -C(v-w-1, v-q) C(p-q-1, p-v-1) C(v, w) C(p, q) g(v)^{p-q},
// and the terminal factor of (15) is C(p-1, v) g(v)^p, so V depends on neither
// M nor n. With the shared inner sums
//   S[v][q] = sum_{w<q} C(v-w-1, v-q) C(v, w) V[q][w],
//   V[p][v] = C(p-1, v) g(v)^p - sum_{q<=v} C(p-q-1, p-v-1) C(p, q) g(v)^{p-q} S[v][q],
// the table costs O(n^3), or O(n^2 log n) since both stages are convolutions
// with the kernel 1/t:
//   S[v][q] = v!/(v-q)! sum_w [V[q][w] / (w! (q-w-1)!)] / (v-w),
//   V[p][v] = g(v)^p [C(p-1, v) - p!/(p-v-1)! sum_q [S[v][q] g(v)^{-q} / (q! (v-q)!)] / (p-q)].
// nbad() adds the initial block of degree 0, which holds the dummy items:
//   N_bad = sum_{q,w} C(M-w-1, M-q) C(M, w) C(n, q) Q^{n-q} V[q][w].
// ---------------------------------------------------------------------------
struct OneTableSuffix {
    OneTableModel model;
    int nmax = 0;
    std::vector<std::vector<Mint>> V;  // V[p][v], 0 <= v < p <= nmax

    // Failing outcomes for n items (n <= nmax) in a table of M slots, O(n^2).
    Mint nbad(int n, int M) const {
        check_one_table_args(n, M, model);
        if (n > nmax) throw std::invalid_argument("n exceeds the suffix table");
        Factorials fac(M);
        const Mint Q = model.g(M, fac);
        std::vector<Mint> Qpow(n + 1);
        Qpow[0] = Mint::one();
        for (int i = 1; i <= n; ++i) Qpow[i] = Qpow[i - 1] * Q;
        Mint total;
        for (int q = 1; q <= n; ++q) {
            Mint inner;
            for (int w = 0; w < q; ++w)
                if (!V[q][w].is_zero()) inner += fac.binom(M - w - 1, M - q) * fac.binom(M, w) * V[q][w];
            total += fac.binom(n, q) * Qpow[n - q] * inner;
        }
        return total;
    }
};

namespace detail {
inline OneTableSuffix init_one_table_suffix(int nmax, OneTableModel model) {
    if (model.d < 1 || nmax < 1) throw std::invalid_argument("suffix table: need d >= 1 and nmax >= 1");
    OneTableSuffix T;
    T.model = model;
    T.nmax = nmax;
    T.V.assign(nmax + 1, {});
    for (int p = 0; p <= nmax; ++p) T.V[p].assign(p, Mint());
    return T;
}
}  // namespace detail

// O(n^3) construction of the suffix table (shared inner sums, no convolution).
inline OneTableSuffix build_one_table_suffix_cubic(int nmax, OneTableModel model) {
    OneTableSuffix T = detail::init_one_table_suffix(nmax, model);
    Factorials fac(nmax);
    std::vector<std::vector<Mint>> gpow(nmax + 1, std::vector<Mint>(nmax + 1));
    for (int v = 0; v <= nmax; ++v) {
        const Mint g = model.g(v, fac);
        gpow[v][0] = Mint::one();
        for (int e = 1; e <= nmax; ++e) gpow[v][e] = gpow[v][e - 1] * g;
    }
    std::vector<std::vector<Mint>> S(nmax);  // S[v][q], 1 <= q <= v
    for (int v = 0; v < nmax; ++v) S[v].assign(v + 1, Mint());

    for (int v = 0; v < nmax; ++v) {
        // Stage 1 with q = v: row V[v][.] is complete, so S[v'][v] for all v' >= v.
        if (v >= 1)
            for (int vp = v; vp < nmax; ++vp) {
                Mint s;
                for (int w = 0; w < v; ++w)
                    if (!T.V[v][w].is_zero()) s += fac.binom(vp - w - 1, vp - v) * fac.binom(vp, w) * T.V[v][w];
                S[vp][v] = s;
            }
        // Stage 2: column v of V from S[v][1..v].
        for (int p = v + 1; p <= nmax; ++p) {
            Mint val = fac.binom(p - 1, v) * gpow[v][p];
            for (int q = 1; q <= v; ++q)
                if (!S[v][q].is_zero())
                    val -= fac.binom(p - q - 1, p - v - 1) * fac.binom(p, q) * gpow[v][p - q] * S[v][q];
            T.V[p][v] = val;
        }
    }
    return T;
}

// Transform length for build_one_table_suffix_ntt. Both convolutions use the kernel 1/t only for
// 1 <= t <= nmax - 1, and only "middle" entries are read: stage 1 convolves a[0..v-1] and reads
// c[v..nmax-1], stage 2 convolves b[1..v] and reads c[v+1..nmax]. A product index s + t >= L wraps
// to s + t - L <= v + nmax - 1 - L, which is below v for L >= nmax, so no entry that is read picks
// up a wrapped term. The one entry at or past L is c[nmax] when L = nmax: it is read at index 0,
// where only s + t = L (the true c[nmax]) lands, since s + t >= 1 in stage 1 and >= 2 in stage 2.
// So L = next_pow2(nmax), half of next_pow2(nmax + 1) when nmax is a power of two.
inline size_t one_table_ntt_length(int nmax) { return std::max<size_t>(2, Ntt::next_pow2(static_cast<size_t>(nmax))); }

// O(n^2 log n) construction of the suffix table with NTT convolutions.
inline OneTableSuffix build_one_table_suffix_ntt(int nmax, OneTableModel model) {
    OneTableSuffix T = detail::init_one_table_suffix(nmax, model);
    Factorials fac(nmax);
    Ntt ntt;
    const size_t L = one_table_ntt_length(nmax);
    std::vector<Mint> ker(nmax);  // ker[t] = 1/t, t <= nmax - 1
    for (int t = 1; t < nmax; ++t) ker[t] = Mint(t).inv();
    const Ntt::Kernel K = ntt.prepare(ker, L);

    std::vector<Mint> gval(nmax + 1), ginv(nmax + 1);
    for (int v = 0; v <= nmax; ++v) {
        gval[v] = model.g(v, fac);
        ginv[v] = gval[v].is_zero() ? Mint() : gval[v].inv();
    }
    std::vector<std::vector<Mint>> S(nmax);  // S[v][q], 1 <= q <= v
    for (int v = 0; v < nmax; ++v) S[v].assign(v + 1, Mint());

    for (int v = 0; v < nmax; ++v) {
        if (v >= 1) {  // Stage 1 with q = v: one convolution gives S[v'][v] for all v' >= v.
            std::vector<Mint> a(v);
            bool any = false;
            for (int w = 0; w < v; ++w) {
                a[w] = T.V[v][w] * fac.inv_fact[w] * fac.inv_fact[v - w - 1];
                any = any || !a[w].is_zero();
            }
            if (any) {
                const std::vector<Mint> c = ntt.convolve(K, a);
                for (int vp = v; vp < nmax; ++vp) S[vp][v] = fac.fact[vp] * fac.inv_fact[vp - v] * c[vp];
            }
        }
        // Stage 2: one convolution gives column v of V.
        if (gval[v].is_zero()) continue;  // V[p][v] = 0 when no candidate set fits in v slots
        std::vector<Mint> b(v + 1);
        bool any = false;
        Mint gi = Mint::one();
        for (int q = 1; q <= v; ++q) {
            gi *= ginv[v];
            b[q] = S[v][q] * gi * fac.inv_fact[q] * fac.inv_fact[v - q];
            any = any || !b[q].is_zero();
        }
        std::vector<Mint> c;
        if (any) c = ntt.convolve(K, b);
        Mint gp = gval[v].pow(v + 1);  // g(v)^p
        for (int p = v + 1; p <= nmax; ++p) {
            Mint val = fac.binom(p - 1, v);
            if (any) val -= fac.fact[p] * fac.inv_fact[p - v - 1] * c[static_cast<size_t>(p) & (L - 1)];  // c[nmax] at 0 if L = nmax
            T.V[p][v] = val * gp;
            gp *= gval[v];
        }
    }
    return T;
}

}  // namespace cuckoo
