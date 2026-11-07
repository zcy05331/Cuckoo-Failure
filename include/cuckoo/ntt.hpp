// Number-theoretic transform over the current Mint::mod.
#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "modint.hpp"

namespace cuckoo {

struct Ntt {
    uint32_t mod;
    uint32_t root;
    Ntt() : mod(Mint::mod), root(primitive_root(Mint::mod)) {}

    static size_t next_pow2(size_t n) {
        size_t L = 1;
        while (L < n) L <<= 1;
        return L;
    }

    // In-place transform; the length must be a power of two dividing mod - 1.
    void transform(std::vector<Mint>& a, bool invert) const {
        const size_t n = a.size();
        if (n == 0 || (n & (n - 1)) != 0 || (mod - 1) % n != 0)
            throw std::invalid_argument("NTT length must be a power of two dividing mod - 1");
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        std::vector<Mint> w(n / 2 + 1);
        for (size_t len = 2; len <= n; len <<= 1) {
            Mint wlen = Mint(root).pow((mod - 1) / len);
            if (invert) wlen = wlen.inv();
            const size_t half = len / 2;
            w[0] = Mint::raw(1);
            for (size_t j = 1; j < half; ++j) w[j] = w[j - 1] * wlen;
            for (size_t i = 0; i < n; i += len)
                for (size_t j = 0; j < half; ++j) {
                    Mint u = a[i + j], v = a[i + j + half] * w[j];
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
        }
        if (invert) {
            Mint ninv = Mint(static_cast<long long>(n)).inv();
            for (Mint& x : a) x *= ninv;
        }
    }

    // A kernel transformed once at length L, for repeated convolutions.
    struct Kernel {
        size_t L;
        std::vector<Mint> hat;
    };
    Kernel prepare(std::vector<Mint> k, size_t L) const {
        if (k.size() > L) throw std::invalid_argument("kernel longer than the transform length");
        k.resize(L);
        transform(k, false);
        return Kernel{L, std::move(k)};
    }
    // The L coefficients of a * k (cyclic if deg a + deg k >= L; callers keep L large enough).
    std::vector<Mint> convolve(const Kernel& k, std::vector<Mint> a) const {
        if (a.size() > k.L) throw std::invalid_argument("sequence longer than the transform length");
        a.resize(k.L);
        transform(a, false);
        for (size_t i = 0; i < k.L; ++i) a[i] *= k.hat[i];
        transform(a, true);
        return a;
    }
};

}  // namespace cuckoo
