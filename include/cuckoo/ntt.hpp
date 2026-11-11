// Number-theoretic transform over the current Mint modulus.
#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "modint.hpp"

namespace cuckoo {

struct Ntt {
    uint32_t mod;
    uint32_t root;
    Ntt() : mod(Mint::modulus()), root(primitive_root(Mint::modulus())) {}

    static size_t next_pow2(size_t n) {
        size_t L = 1;
        while (L < n) L <<= 1;
        return L;
    }

    // In-place transform in natural order; the length must be a power of two dividing mod - 1.
    void transform(std::vector<Mint>& a, bool invert) const {
        const size_t n = a.size();
        check_length(n);
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
            w[0] = Mint::one();
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

    // A kernel transformed once at length L, for repeated convolutions. The spectrum is kept
    // in the bit-reversed order produced by forward_dif and already carries the factor 1/L,
    // so a convolution is one forward pass, a pointwise product and one inverse pass, with
    // no permutation and no final scaling. The twiddle tables are built once per length:
    // tw[half + j] = w_{2 half}^j and twinv[half + j] = w_{2 half}^{-j} for 0 <= j < half.
    struct Kernel {
        size_t L;
        std::vector<Mint> hat, tw, twinv;
    };
    Kernel prepare(std::vector<Mint> k, size_t L) const {
        if (k.size() > L) throw std::invalid_argument("kernel longer than the transform length");
        check_length(L);
        Kernel K{L, {}, std::vector<Mint>(L), std::vector<Mint>(L)};
        for (size_t half = 1; half < L; half <<= 1) {
            const Mint w = Mint(root).pow((mod - 1) / (2 * half)), wi = w.inv();
            K.tw[half] = K.twinv[half] = Mint::one();
            for (size_t j = 1; j < half; ++j) {
                K.tw[half + j] = K.tw[half + j - 1] * w;
                K.twinv[half + j] = K.twinv[half + j - 1] * wi;
            }
        }
        k.resize(L);
        forward_dif(k, K);
        const Mint linv = Mint(static_cast<long long>(L)).inv();
        for (Mint& x : k) x *= linv;
        K.hat = std::move(k);
        return K;
    }
    // The cyclic convolution of a and k at length L: entry i is the sum of a[s] k[t] over
    // s + t = i (mod L). Callers must make sure the entries they read are free of wrap-around.
    std::vector<Mint> convolve(const Kernel& k, std::vector<Mint> a) const {
        if (a.size() > k.L) throw std::invalid_argument("sequence longer than the transform length");
        a.resize(k.L);
        forward_dif(a, k);
        for (size_t i = 0; i < k.L; ++i) a[i] *= k.hat[i];
        inverse_dit(a, k);
        return a;
    }

private:
    void check_length(size_t n) const {
        if (n == 0 || (n & (n - 1)) != 0 || (mod - 1) % n != 0)
            throw std::invalid_argument("NTT length must be a power of two dividing mod - 1");
    }
    // Gentleman-Sande butterflies: natural order in, bit-reversed order out.
    static void forward_dif(std::vector<Mint>& a, const Kernel& K) {
        const size_t n = a.size();
        for (size_t half = n >> 1; half >= 1; half >>= 1) {
            const Mint* w = K.tw.data() + half;
            for (size_t i = 0; i < n; i += 2 * half)
                for (size_t j = 0; j < half; ++j) {
                    const Mint u = a[i + j], v = a[i + j + half];
                    a[i + j] = u + v;
                    a[i + j + half] = (u - v) * w[j];
                }
        }
    }
    // Cooley-Tukey butterflies with inverse roots: bit-reversed order in, natural order out
    // (unscaled; the factor 1/L lives in Kernel::hat).
    static void inverse_dit(std::vector<Mint>& a, const Kernel& K) {
        const size_t n = a.size();
        for (size_t half = 1; half < n; half <<= 1) {
            const Mint* w = K.twinv.data() + half;
            for (size_t i = 0; i < n; i += 2 * half)
                for (size_t j = 0; j < half; ++j) {
                    const Mint u = a[i + j], v = a[i + j + half] * w[j];
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
        }
    }
};

}  // namespace cuckoo
