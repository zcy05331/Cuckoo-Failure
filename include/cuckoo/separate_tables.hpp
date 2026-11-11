// Separate-table d-choice cuckoo hashing: exact failure counts (paper, Section 4).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "modint.hpp"
#include "single_table.hpp"  // phi

namespace cuckoo {

inline void check_separate_args(int n, const std::vector<int>& m) {
    if (m.empty()) throw std::invalid_argument("separate tables: need d >= 1 tables");
    long long M = 0;
    for (int x : m) {
        if (x < 1) throw std::invalid_argument("separate tables: lengths must be positive");
        M += x;
    }
    if (n < 1 || M <= n) throw std::invalid_argument("separate tables: need 1 <= n < M");
}

// ---------------------------------------------------------------------------
// Algorithm 4 (Section 4.2): the forward recurrence over (l, b), where b is the
// vector of forbidden counts per table. Transitions use tau_d of eq. (18),
// eq. (19) fills the table and eq. (20) adds the final block. Only (0, 0) and
// the retained states (z <= l < M, |b| > l, b_j < m_j for all j) are stored:
// O(n^{d+1}) states, each scanning all earlier states: O(n^{2d+2}) operations.
// ---------------------------------------------------------------------------
inline Mint separate_forward(int n, const std::vector<int>& m) {
    check_separate_args(n, m);
    const int d = static_cast<int>(m.size());
    int M = 0;
    for (int x : m) M += x;
    const int z = M - n;
    Factorials fac(M);

    struct State {
        int l;
        std::vector<int> b;
        int bsum;
        Mint h;  // h_m(b) = prod_j (m_j - b_j), eq. (16)
    };
    std::vector<State> st;
    {
        State s0{0, std::vector<int>(d, 0), 0, Mint::one()};
        for (int x : m) s0.h *= Mint(x);
        st.push_back(s0);
    }
    // Retained states in order of increasing l: v = m - b >= 1 with |v| < p = M - l.
    std::vector<int> v(d);
    for (int l = z; l < M; ++l) {
        const int p = M - l;
        std::function<void(int, int)> gen = [&](int t, int remaining) {
            if (t == d) {
                State s{l, std::vector<int>(d), 0, Mint::one()};
                for (int j = 0; j < d; ++j) {
                    s.b[j] = m[j] - v[j];
                    s.bsum += s.b[j];
                    s.h *= Mint(v[j]);
                }
                st.push_back(s);
                return;
            }
            const int cap = std::min(m[t], remaining - (d - 1 - t));
            for (int x = 1; x <= cap; ++x) {
                v[t] = x;
                gen(t + 1, remaining - x);
            }
        };
        gen(0, p - 1);
    }

    const size_t N = st.size();
    std::vector<std::vector<Mint>> hp(N, std::vector<Mint>(n + 1));  // hp[i][u] = h_i^u
    for (size_t i = 0; i < N; ++i) {
        hp[i][0] = Mint::one();
        for (int u = 1; u <= n; ++u) hp[i][u] = hp[i][u - 1] * st[i].h;
    }
    auto nu = [z](int l) { return std::max(0, l - z); };

    std::vector<Mint> F(N);
    F[0] = Mint::one();
    for (size_t i = 1; i < N; ++i) {
        const State& T = st[i];
        const int r = T.l;
        Mint s;
        for (size_t j = 0; j < i && st[j].l < r; ++j) {
            if (F[j].is_zero()) continue;
            const State& P = st[j];
            if (P.bsum >= T.bsum) continue;
            bool le = true;
            for (int t = 0; t < d && le; ++t) le = P.b[t] <= T.b[t];
            if (!le) continue;
            Mint tau = phi(fac, T.bsum - P.l, P.bsum - P.l, r - P.l);
            if (tau.is_zero()) continue;
            for (int t = 0; t < d; ++t) tau *= fac.binom(m[t] - P.b[t], T.b[t] - P.b[t]);
            const int u = r - std::max(P.l, z);
            tau *= fac.binom(n - nu(P.l), u) * hp[j][u];
            s += F[j] * tau;
        }
        F[i] = s;
    }
    Mint total;
    for (size_t i = 0; i < N; ++i)
        if (!F[i].is_zero())
            total += F[i] * fac.binom(M - st[i].l - 1, M - st[i].bsum) * hp[i][n - nu(st[i].l)];
    return total;
}

// All integer vectors v >= 1 (coordinatewise) of dimension d with |v| <= smax,
// in order of increasing |v|, with the index of v - e_t for each coordinate t.
struct Simplex {
    int d = 0, smax = 0;
    std::vector<int> flat;  // d coordinates per vector
    std::vector<int> sum;   // |v|
    std::vector<int> cnt;   // cnt[s] = number of vectors with |v| <= s, 0 <= s <= smax
    std::vector<int> prev;  // prev[i*d + t] = index of v_i - e_t, or -1

    Simplex() = default;
    Simplex(int d_, int smax_) : d(d_), smax(std::max(smax_, 0)) {
        if (std::pow(static_cast<double>(smax + 2), d) > 1.8e19) throw std::invalid_argument("simplex too large");
        std::vector<int> cur(d);
        for (int s = d; s <= smax; ++s) gen(cur, 0, s, s);
        cnt.assign(smax + 1, 0);
        for (int s = 0, i = 0; s <= smax; ++s) {
            while (i < size() && sum[i] <= s) ++i;
            cnt[s] = i;
        }
        std::unordered_map<uint64_t, int> index;
        index.reserve(static_cast<size_t>(size()) * 2 + 1);
        for (int i = 0; i < size(); ++i) index[key(vec(i))] = i;
        prev.assign(static_cast<size_t>(size()) * d, -1);
        for (int i = 0; i < size(); ++i)
            for (int t = 0; t < d; ++t) {
                if (flat[static_cast<size_t>(i) * d + t] < 2) continue;
                cur.assign(flat.begin() + static_cast<long>(i) * d, flat.begin() + static_cast<long>(i + 1) * d);
                --cur[t];
                prev[static_cast<size_t>(i) * d + t] = index.at(key(cur.data()));
            }
    }
    int size() const { return static_cast<int>(sum.size()); }
    const int* vec(int i) const { return &flat[static_cast<size_t>(i) * d]; }

private:
    uint64_t key(const int* v) const {
        uint64_t k = 0;
        for (int t = 0; t < d; ++t) k = k * static_cast<uint64_t>(smax + 2) + static_cast<uint64_t>(v[t]);
        return k;
    }
    void gen(std::vector<int>& cur, int t, int remaining, int s) {
        if (t == d - 1) {
            cur[t] = remaining;
            flat.insert(flat.end(), cur.begin(), cur.end());
            sum.push_back(s);
            return;
        }
        for (int x = 1; x <= remaining - (d - 1 - t); ++x) {
            cur[t] = x;
            gen(cur, t + 1, remaining - x, s);
        }
    }
};

// ---------------------------------------------------------------------------
// Algorithm 5 (Section 4.3, "faster evaluation"): the suffix table for
// separate tables. In the coordinates p = M - l and v = m - b (available slots
// per table), with g(v) = prod_t v_t,
//   V[p][v] = C(p-1, |v|) g(v)^p - sum_{q<=|v|} C(p-q-1, p-|v|-1) C(p, q) g(v)^{p-q} S[v][q],
//   S[v][q] = sum_{w<=v, |w|<q} C(|v|-|w|-1, |v|-q) prod_t C(v_t, w_t) V[q][w],
// over the states v >= 1, |v| < p <= n (C(n, d+1) of them). The inner sums
// are a d-dimensional convolution,
//   S[v][q] = prod_t v_t! / (|v|-q)! * sum_{e+w=v} K[e] X_q[w],
//   K[e] = (|e|-1)! / prod_t e_t!,  X_q[w] = V[q][w] / (prod_t w_t! (q-|w|-1)!),
// whose kernel has the generating function -log(1 - sigma), sigma = x_1+...+x_d.
// K * X_q = sum_{s>=1} sigma^s X_q / s is evaluated by Horner's rule with O(n)
// multiplications by sigma, each O(d) per coefficient, which gives O(n^{d+2})
// in total for the table. The table depends only on n and d; nbad() then
// evaluates one table-length vector m in O(n^{d+1}):
//   N_bad = sum_{q,w} C(M-|w|-1, M-q) prod_t C(m_t, w_t) C(n, q) Q^{n-q} V[q][w].
// ---------------------------------------------------------------------------
struct SeparateSuffix {
    int d = 0, nmax = 0;
    Simplex sx;                        // vectors v >= 1 with |v| <= nmax - 1
    std::vector<std::vector<Mint>> V;  // V[p] has sx.cnt[p-1] entries, p = 1..nmax

    // Failing outcomes for n items (n <= nmax) and table lengths m, O(n^{d+1}).
    Mint nbad(int n, const std::vector<int>& m) const {
        check_separate_args(n, m);
        if (static_cast<int>(m.size()) != d) throw std::invalid_argument("wrong number of tables");
        if (n > nmax) throw std::invalid_argument("n exceeds the suffix table");
        int M = 0;
        for (int x : m) M += x;
        Factorials fac(M);
        Mint Q = Mint::one();
        for (int x : m) Q *= Mint(x);
        std::vector<Mint> Qpow(n + 1);
        Qpow[0] = Mint::one();
        for (int i = 1; i <= n; ++i) Qpow[i] = Qpow[i - 1] * Q;
        Mint total;
        for (int q = 1; q <= n; ++q) {
            Mint inner;
            for (int i = 0; i < sx.cnt[q - 1]; ++i) {
                if (V[q][i].is_zero()) continue;
                Mint w = fac.binom(M - sx.sum[i] - 1, M - q);
                const int* v = sx.vec(i);
                for (int t = 0; t < d && !w.is_zero(); ++t) w *= fac.binom(m[t], v[t]);
                inner += w * V[q][i];
            }
            total += fac.binom(n, q) * Qpow[n - q] * inner;
        }
        return total;
    }
};

inline SeparateSuffix build_separate_suffix(int nmax, int d) {
    if (d < 1 || nmax < 1) throw std::invalid_argument("suffix table: need d >= 1 and nmax >= 1");
    SeparateSuffix T;
    T.d = d;
    T.nmax = nmax;
    T.sx = Simplex(d, nmax - 1);
    const Simplex& sx = T.sx;
    const int N = sx.size();
    Factorials fac(nmax);
    std::vector<Mint> inv(nmax + 1);
    for (int s = 1; s <= nmax; ++s) inv[s] = Mint(s).inv();
    std::vector<Mint> g(N), vfact(N), vfact_inv(N);
    for (int i = 0; i < N; ++i) {
        const int* v = sx.vec(i);
        g[i] = Mint::one();
        vfact[i] = Mint::one();
        for (int t = 0; t < d; ++t) {
            g[i] *= Mint(v[t]);
            vfact[i] *= fac.fact[v[t]];
        }
        vfact_inv[i] = vfact[i].inv();
    }
    T.V.assign(nmax + 1, {});
    for (int p = 1; p <= nmax; ++p) {
        T.V[p].assign(sx.cnt[p - 1], Mint());
        for (int i = 0; i < sx.cnt[p - 1]; ++i) T.V[p][i] = fac.binom(p - 1, sx.sum[i]) * g[i].pow(p);
    }

    std::vector<Mint> X(N), R(N);
    for (int q = 1; q < nmax; ++q) {
        // Row q is final: every S[w][q'] with q' < q has been pushed into it.
        const int nq = sx.cnt[q - 1];  // support of X_q: |w| <= q - 1
        bool any = false;
        for (int i = 0; i < nq; ++i) {
            X[i] = T.V[q][i] * vfact_inv[i] * fac.inv_fact[q - 1 - sx.sum[i]];
            any = any || !X[i].is_zero();
        }
        if (!any) continue;
        // Horner: R_s = X_q / s + sigma R_{s+1}, needed for max(d, q-s) <= |v| <= nmax-1-s.
        // Updating in place in decreasing index order reads only old entries.
        std::fill(R.begin(), R.end(), Mint());
        for (int s = nmax - 1 - d; s >= 1; --s) {
            const int lo = std::max(d, q - s), hi = nmax - 1 - s;
            if (lo > hi) continue;
            const int ibeg = sx.cnt[lo - 1], iend = sx.cnt[hi];
            for (int i = iend - 1; i >= ibeg; --i) {
                Mint acc;
                const int* pv = &sx.prev[static_cast<size_t>(i) * d];
                for (int t = 0; t < d; ++t)
                    if (pv[t] >= 0) acc += R[pv[t]];
                if (i < nq) acc += X[i] * inv[s];
                R[i] = acc;
            }
        }
        // Y = sigma R_1 on |v| >= q; S[v][q] = prod v_t! / (|v|-q)! * Y[v], pushed into V[p][v].
        for (int i = nq; i < N; ++i) {
            Mint acc;
            const int* pv = &sx.prev[static_cast<size_t>(i) * d];
            for (int t = 0; t < d; ++t)
                if (pv[t] >= 0) acc += R[pv[t]];
            if (acc.is_zero()) continue;
            const int sv = sx.sum[i];
            const Mint S = acc * vfact[i] * fac.inv_fact[sv - q];
            Mint gp = g[i].pow(sv + 1 - q);  // g(v)^{p-q}
            for (int p = sv + 1; p <= nmax; ++p) {
                T.V[p][i] -= fac.binom(p - q - 1, p - sv - 1) * fac.binom(p, q) * gp * S;
                gp *= g[i];
            }
        }
    }
    return T;
}

}  // namespace cuckoo
