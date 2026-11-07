// Reconstruction of an exact integer from residues modulo several primes
// (Garner's algorithm), with decimal and floating-point output.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "modint.hpp"

namespace cuckoo {

// Mixed-radix digits x of N = x0 + x1 p0 + x2 p0 p1 + ..., 0 <= N < prod p_i.
inline std::vector<uint32_t> garner(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& residues) {
    const size_t K = primes.size();
    std::vector<uint32_t> x(K);
    for (size_t i = 0; i < K; ++i) {
        const uint64_t p = primes[i];
        uint64_t cur = 0, prod = 1;
        for (size_t j = 0; j < i; ++j) {
            cur = (cur + x[j] % p * prod) % p;
            prod = prod * (primes[j] % p) % p;
        }
        const uint64_t t = (residues[i] % p + p - cur) % p;
        x[i] = static_cast<uint32_t>(t * powmod_u64(prod, p - 2, p) % p);
    }
    return x;
}

// log2(N) from the mixed-radix digits (-inf for N = 0), to long double precision.
inline long double log2_mixed_radix(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& x) {
    long double m = 0;  // N = m * 2^e
    long e = 0;
    for (size_t i = x.size(); i-- > 0;) {
        m = m * primes[i] + std::ldexp(static_cast<long double>(x[i]), static_cast<int>(-e));
        if (m != 0) {
            int de;
            m = std::frexp(m, &de);
            e += de;
        }
    }
    if (m == 0) return -INFINITY;
    return static_cast<long double>(e) + std::log2(m);
}

// Minimal natural number in base 10^9 for exact decimal output.
struct BigNat {
    std::vector<uint32_t> limbs{0};  // little-endian
    static constexpr uint32_t BASE = 1000000000u;

    void mul_add(uint64_t mul, uint64_t add) {
        unsigned __int128 carry = add;
        for (uint32_t& l : limbs) {
            const unsigned __int128 cur = static_cast<unsigned __int128>(l) * mul + carry;
            l = static_cast<uint32_t>(cur % BASE);
            carry = cur / BASE;
        }
        while (carry) {
            limbs.push_back(static_cast<uint32_t>(carry % BASE));
            carry /= BASE;
        }
    }
    void div_small(uint32_t q) {  // exact or floor division
        uint64_t rem = 0;
        for (size_t i = limbs.size(); i-- > 0;) {
            const uint64_t cur = rem * BASE + limbs[i];
            limbs[i] = static_cast<uint32_t>(cur / q);
            rem = cur % q;
        }
        while (limbs.size() > 1 && limbs.back() == 0) limbs.pop_back();
    }
    std::string str() const {
        std::string s = std::to_string(limbs.back());
        for (size_t i = limbs.size() - 1; i-- > 0;) {
            std::string part = std::to_string(limbs[i]);
            s += std::string(9 - part.size(), '0') + part;
        }
        return s;
    }
};

inline BigNat mixed_radix_to_bignat(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& x) {
    BigNat b;
    for (size_t i = x.size(); i-- > 0;) b.mul_add(primes[i], x[i]);
    return b;
}

}  // namespace cuckoo
