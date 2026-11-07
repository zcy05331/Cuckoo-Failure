// Modular arithmetic with a runtime prime modulus (< 2^32), factorial tables
// with the paper's binomial convention, and NTT-friendly prime generation.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuckoo {

struct Mint {
    static inline uint32_t mod = 998244353u;  // set before any arithmetic
    uint32_t v = 0;

    Mint() = default;
    Mint(long long x) {
        long long r = x % static_cast<long long>(mod);
        if (r < 0) r += mod;
        v = static_cast<uint32_t>(r);
    }
    static Mint raw(uint32_t x) {
        Mint m;
        m.v = x;
        return m;
    }

    Mint& operator+=(Mint o) {
        uint64_t s = static_cast<uint64_t>(v) + o.v;
        if (s >= mod) s -= mod;
        v = static_cast<uint32_t>(s);
        return *this;
    }
    Mint& operator-=(Mint o) {
        v = (v >= o.v) ? v - o.v : static_cast<uint32_t>(static_cast<uint64_t>(v) + mod - o.v);
        return *this;
    }
    Mint& operator*=(Mint o) {
        v = static_cast<uint32_t>(static_cast<uint64_t>(v) * o.v % mod);
        return *this;
    }
    friend Mint operator+(Mint a, Mint b) { return a += b; }
    friend Mint operator-(Mint a, Mint b) { return a -= b; }
    friend Mint operator*(Mint a, Mint b) { return a *= b; }
    Mint operator-() const { return raw(v ? mod - v : 0); }
    Mint pow(unsigned long long e) const {
        Mint r = raw(1), b = *this;
        while (e) {
            if (e & 1) r *= b;
            b *= b;
            e >>= 1;
        }
        return r;
    }
    Mint inv() const { return pow(mod - 2); }
    friend Mint operator/(Mint a, Mint b) { return a * b.inv(); }
    bool operator==(Mint o) const { return v == o.v; }
    bool operator!=(Mint o) const { return v != o.v; }
    bool is_zero() const { return v == 0; }
};

// Factorials and binomial coefficients modulo Mint::mod. The binomial follows
// the paper's convention: C(a, b) = 0 unless a, b are integers with 0 <= b <= a.
struct Factorials {
    std::vector<Mint> fact, inv_fact;

    explicit Factorials(int n) : fact(n + 1), inv_fact(n + 1) {
        if (n < 0 || static_cast<uint64_t>(n) >= Mint::mod)
            throw std::invalid_argument("factorial table: modulus too small");
        fact[0] = Mint::raw(1);
        for (int i = 1; i <= n; ++i) fact[i] = fact[i - 1] * Mint(i);
        inv_fact[n] = fact[n].inv();
        for (int i = n; i > 0; --i) inv_fact[i - 1] = inv_fact[i] * Mint(i);
    }
    int limit() const { return static_cast<int>(fact.size()) - 1; }
    Mint binom(long long a, long long b) const {
        if (a < 0 || b < 0 || b > a) return Mint();
        if (a > limit()) throw std::out_of_range("binom: argument exceeds the factorial table");
        return fact[a] * inv_fact[b] * inv_fact[a - b];
    }
};

inline uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {  // m < 2^32
    uint64_t r = 1;
    a %= m;
    while (e) {
        if (e & 1) r = r * a % m;
        a = a * a % m;
        e >>= 1;
    }
    return r;
}

inline bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u})
        if (n % p == 0) return n == p;
    uint32_t d = n - 1;
    int s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        ++s;
    }
    for (uint64_t a : {2ull, 7ull, 61ull}) {  // deterministic Miller-Rabin below 4759123141
        uint64_t x = powmod_u64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int i = 1; i < s && composite; ++i) {
            x = x * x % n;
            if (x == n - 1) composite = false;
        }
        if (composite) return false;
    }
    return true;
}

inline uint32_t primitive_root(uint32_t p) {
    std::vector<uint32_t> factors;
    uint32_t m = p - 1;
    for (uint32_t f = 2; static_cast<uint64_t>(f) * f <= m; ++f)
        if (m % f == 0) {
            factors.push_back(f);
            while (m % f == 0) m /= f;
        }
    if (m > 1) factors.push_back(m);
    for (uint32_t g = 2; g < p; ++g) {
        bool ok = true;
        for (uint32_t f : factors)
            if (powmod_u64(g, (p - 1) / f, p) == 1) {
                ok = false;
                break;
            }
        if (ok) return g;
    }
    throw std::runtime_error("no primitive root found");
}

// `count` distinct primes p < 2^32 with 2^min_pow2 | p - 1 (so NTTs of length
// up to 2^min_pow2 are available), largest first.
inline std::vector<uint32_t> ntt_primes(int count, int min_pow2 = 20) {
    std::vector<uint32_t> out;
    for (int k = 31; k >= min_pow2 && static_cast<int>(out.size()) < count; --k) {
        int64_t c = static_cast<int64_t>(((1ull << 32) - 2) >> k);  // c * 2^k + 1 < 2^32
        if (c % 2 == 0) --c;  // odd c: the 2-adic valuation of p - 1 is exactly k
        for (; c >= 1 && static_cast<int>(out.size()) < count; c -= 2) {
            uint64_t p = (static_cast<uint64_t>(c) << k) + 1;
            if (is_prime_u32(static_cast<uint32_t>(p))) out.push_back(static_cast<uint32_t>(p));
        }
    }
    if (static_cast<int>(out.size()) < count) throw std::runtime_error("not enough NTT-friendly primes");
    std::sort(out.rbegin(), out.rend());
    return out;
}

}  // namespace cuckoo
