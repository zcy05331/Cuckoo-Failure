// Batch engine for the one-table suffix-table algorithm (paper Section 3.3, uniform d-subset
// model) with 64-bit residues. One source, three builds:
//
//   nvcc  -O3 -std=c++17 cuckoo_fast.cu -o cuckoo_fast_gpu          CUDA: every loop is a kernel
//   g++   -O3 -std=c++17 -x c++ cuckoo_fast.cu -o cuckoo_fast_cpu   same loops on the host
//   g++   ... -mavx512f -mavx512dq -mavx512vl -mavx512ifma ...      host, hot loops in AVX-512 IFMA
//
// All builds use the same primes and print the same residues, so they check each other.
//
//   cuckoo_fast D QUERIES FIRST COUNT [--batch B] [--out FILE]
//
// QUERIES: lines "n M". Prime number i is the i-th largest prime p < 2^PB with 2^20 | p - 1
// (PB = 50, or 62 when built with -DFP_RB=64). A query is evaluated for prime i only if
// i < need(n, M), the number of primes whose product exceeds Q^n. Output lines:
// "i p n M residue". With --out the file is appended to and primes already in it are skipped,
// so an interrupted run (or another GPU) can continue it.
//
// Arithmetic: Montgomery form with R = 2^52 (default; operands < 2^50, the form AVX-512 IFMA
// needs) or R = 2^64 (-DFP_RB=64; operands < 2^62, fewer primes, scalar and CUDA only).
//
// Layout per prime: V[p][w] (0 <= w < p <= nmax) and S[v][q] (1 <= q <= v < nmax) are packed
// triangles, so a prime takes about nmax^2 words = 8 nmax^2 bytes. B primes are processed at
// once; on the GPU this is what fills the device, since the loop over v is sequential.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define DEV __host__ __device__ inline
#else
#define DEV inline
#endif
#if !defined(__CUDACC__) && defined(__AVX512IFMA__) && defined(__AVX512VL__)
#include <immintrin.h>
#define USE_IFMA 1
#else
#define USE_IFMA 0
#endif

#ifndef FP_RB
#define FP_RB 52
#endif
#if FP_RB == 52
#define PRIME_BITS 50
#elif FP_RB == 64
#define PRIME_BITS 62
#else
#error "FP_RB must be 52 or 64"
#endif
#if USE_IFMA && FP_RB != 52
#error "the IFMA build needs FP_RB = 52"
#endif
#define TWO_ADIC 20  // every prime has 2^20 | p - 1: transforms up to length 2^20

typedef uint64_t u64;

// ----------------------------------------------------------------------------------------
// Modular arithmetic (host and device)
// ----------------------------------------------------------------------------------------
struct Mod {
    u64 p;     // the prime
    u64 pinv;  // p^{-1} mod R
    u64 one;   // R mod p, the stored form of 1
    u64 linv;  // 1 / L (Montgomery form), L the current transform length
};

DEV u64 mulhi64(u64 a, u64 b) {
#ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
#else
    return (u64)(((unsigned __int128)a * b) >> 64);
#endif
}

// a b R^{-1} mod p for a, b < p. With q = (a b mod R) p^{-1} mod R the difference a b - q p is
// divisible by R, and (a b - q p) / R = floor(a b / R) - floor(q p / R) lies in (-p, p).
DEV u64 mmul(u64 a, u64 b, const Mod& m) {
    const u64 lo = a * b, hi = mulhi64(a, b);
#if FP_RB == 64
    const u64 q = lo * m.pinv;
    const u64 t = hi, h2 = mulhi64(q, m.p);
#else
    const u64 MASK = (1ull << 52) - 1;
    const u64 q = ((lo & MASK) * m.pinv) & MASK;
    const u64 t = (hi << 12) | (lo >> 52);
    const u64 h2 = (mulhi64(q, m.p) << 12) | ((q * m.p) >> 52);
#endif
    const u64 r = t - h2;
    return t >= h2 ? r : r + m.p;
}
DEV u64 madd(u64 a, u64 b, const Mod& m) {
    const u64 s = a + b;
    return s >= m.p ? s - m.p : s;
}
DEV u64 msub(u64 a, u64 b, const Mod& m) { return a >= b ? a - b : a - b + m.p; }
DEV u64 mpow(u64 base, u64 e, const Mod& m) {
    u64 r = m.one;
    while (e) {
        if (e & 1) r = mmul(r, base, m);
        base = mmul(base, base, m);
        e >>= 1;
    }
    return r;
}

// ----------------------------------------------------------------------------------------
// Device data and the loop bodies. Every body is a functor f(idx) over a flat index range;
// on the GPU each is one kernel launch, on the host a loop.
// ----------------------------------------------------------------------------------------
struct Ctx {
    Mod* mod;                // [B]
    u64 *fact, *ifact;       // [B][F1]
    u64 *tw, *twinv, *hat;   // [B][L]   tw[half + j] = w_{2 half}^j, twinv the inverses
    u64* buf;                // [B][L]   transform buffer
    u64 *V, *S;              // [B][T], [B][TS]
    u64 *glo, *ghi;          // [B][LO], [B][HI]   g^j and g^(i LO)
    u64 *gilo, *gihi;        // same for 1/g
    u64 *A, *qout;           // [B][NA]  query scratch
    size_t F1, L, T, TS, LO, HI, NA;
    int nmax, h, d;
    int lg;  // log2 L: flat indices are (b << lg) | i, so kernels split them with shifts
};
DEV size_t offV(size_t p) { return p * (p - 1) / 2; }  // row p >= 1, entries w = 0..p-1
DEV size_t offS(size_t v) { return v * (v - 1) / 2; }  // row v >= 1, entry q stored at q - 1

// Stage 1 input: a[w] = V[v][w] / (w! (v-w-1)!), w < v.
struct Stage1Pre {
    Ctx c;
    int v;
    DEV void operator()(size_t idx) const {
        const size_t b = idx >> c.lg, i = idx & (c.L - 1);
        u64 r = 0;
        if (i < (size_t)v) {
            const Mod m = c.mod[b];
            const u64* fi = c.ifact + b * c.F1;
            r = mmul(mmul(c.V[b * c.T + offV(v) + i], fi[i], m), fi[v - 1 - i], m);
        }
        c.buf[b * c.L + i] = r;
    }
};
// Stage 1 output: S[v'][v] = v'! / (v'-v)! c[v'], v <= v' < nmax.
struct Stage1Post {
    Ctx c;
    int v;
    DEV void operator()(size_t idx) const {
        const size_t b = idx >> c.lg, i = idx & (c.L - 1), vp = v + i;
        if (i >= (size_t)(c.nmax - v)) return;
        const Mod m = c.mod[b];
        const u64 *f = c.fact + b * c.F1, *fi = c.ifact + b * c.F1;
        c.S[b * c.TS + offS(vp) + v - 1] = mmul(mmul(f[vp], fi[vp - v], m), c.buf[b * c.L + vp], m);
    }
};
// Power tables of g = C(v, d) and 1/g: x^e = hi[e >> h] lo[e & (LO-1)].
struct PowTab {
    Ctx c;
    int v;
    DEV void operator()(size_t idx) const {
        const size_t half = c.LO + c.HI, b = idx / (2 * half);
        size_t i = idx % (2 * half);
        const Mod m = c.mod[b];
        const u64 *f = c.fact + b * c.F1, *fi = c.ifact + b * c.F1;
        u64 g = mmul(mmul(f[v], fi[c.d], m), fi[v - c.d], m);
        const bool inv = i >= half;
        if (inv) {
            i -= half;
            g = mpow(g, m.p - 2, m);
        }
        if (i < c.LO) (inv ? c.gilo : c.glo)[b * c.LO + i] = mpow(g, i, m);
        else (inv ? c.gihi : c.ghi)[b * c.HI + (i - c.LO)] = mpow(g, (u64)(i - c.LO) << c.h, m);
    }
};
// Stage 2 input: b[q] = S[v][q] g^{-q} / (q! (v-q)!), 1 <= q <= v.
struct Stage2Pre {
    Ctx c;
    int v;
    DEV void operator()(size_t idx) const {
        const size_t b = idx >> c.lg, i = idx & (c.L - 1);
        u64 r = 0;
        if (i >= 1 && i <= (size_t)v) {
            const Mod m = c.mod[b];
            const u64* fi = c.ifact + b * c.F1;
            const u64 gp = mmul(c.gihi[b * c.HI + (i >> c.h)], c.gilo[b * c.LO + (i & (c.LO - 1))], m);
            r = mmul(mmul(mmul(c.S[b * c.TS + offS(v) + i - 1], gp, m), fi[i], m), fi[v - i], m);
        }
        c.buf[b * c.L + i] = r;
    }
};
// Stage 2 output: V[p][v] = g^p [C(p-1, v) - p! / (p-v-1)! c[p]], v < p <= nmax.
struct Stage2Post {
    Ctx c;
    int v;
    DEV void operator()(size_t idx) const {
        const size_t b = idx >> c.lg, i = idx & (c.L - 1), p = v + 1 + i;
        if (i >= (size_t)(c.nmax - v)) return;
        const Mod m = c.mod[b];
        const u64 *f = c.fact + b * c.F1, *fi = c.ifact + b * c.F1;
        u64 t = msub(mmul(f[p - 1], fi[v], m), mmul(f[p], c.buf[b * c.L + p], m), m);
        t = mmul(t, fi[p - v - 1], m);
        const u64 gp = mmul(c.ghi[b * c.HI + (p >> c.h)], c.glo[b * c.LO + (p & (c.LO - 1))], m);
        c.V[b * c.T + offV(p) + v] = mmul(t, gp, m);
    }
};
struct Pointwise {
    Ctx c;
    DEV void operator()(size_t idx) const {
        const Mod m = c.mod[idx >> c.lg];
        c.buf[idx] = mmul(c.buf[idx], c.hat[idx], m);
    }
};
struct HatFinish {  // hat = spectrum of the kernel times 1/L
    Ctx c;
    DEV void operator()(size_t idx) const {
        const Mod m = c.mod[idx >> c.lg];
        c.hat[idx] = mmul(c.buf[idx], m.linv, m);
    }
};
// r levels of Gentleman-Sande butterflies (natural order in, bit-reversed out), halves
// s << (r-1), ..., s. One call handles the 2^r entries base + k s of one group.
struct NttFwd {
    Ctx c;
    size_t s;
    int r, ls;  // s = 1 << ls
    DEV void operator()(size_t idx) const {
        const size_t b = idx >> (c.lg - r), g = idx & ((c.L >> r) - 1), j0 = g & (s - 1);
        const Mod m = c.mod[b];
        u64* a = c.buf + b * c.L + ((g >> ls) << (ls + r)) + j0;
        const u64* tw = c.tw + b * c.L;
        const int n = 1 << r;
        u64 x[8];
        for (int k = 0; k < n; ++k) x[k] = a[k * s];
        for (int hh = n >> 1; hh >= 1; hh >>= 1)
            for (int k0 = 0; k0 < n; k0 += 2 * hh)
                for (int kk = 0; kk < hh; ++kk) {
                    const u64 u = x[k0 + kk], w = x[k0 + kk + hh];
                    x[k0 + kk] = madd(u, w, m);
                    x[k0 + kk + hh] = mmul(msub(u, w, m), tw[hh * s + j0 + kk * s], m);
                }
        for (int k = 0; k < n; ++k) a[k * s] = x[k];
    }
};
// r levels of Cooley-Tukey butterflies with inverse roots (bit-reversed in, natural out,
// unscaled), halves s, 2 s, ..., s << (r-1).
struct NttInv {
    Ctx c;
    size_t s;
    int r, ls;  // s = 1 << ls
    DEV void operator()(size_t idx) const {
        const size_t b = idx >> (c.lg - r), g = idx & ((c.L >> r) - 1), j0 = g & (s - 1);
        const Mod m = c.mod[b];
        u64* a = c.buf + b * c.L + ((g >> ls) << (ls + r)) + j0;
        const u64* tw = c.twinv + b * c.L;
        const int n = 1 << r;
        u64 x[8];
        for (int k = 0; k < n; ++k) x[k] = a[k * s];
        for (int hh = 1; hh < n; hh <<= 1)
            for (int k0 = 0; k0 < n; k0 += 2 * hh)
                for (int kk = 0; kk < hh; ++kk) {
                    const u64 u = x[k0 + kk], w = mmul(x[k0 + kk + hh], tw[hh * s + j0 + kk * s], m);
                    x[k0 + kk] = madd(u, w, m);
                    x[k0 + kk + hh] = msub(u, w, m);
                }
        for (int k = 0; k < n; ++k) a[k * s] = x[k];
    }
};
// Query (n, M): A[w] = (M-w-1)! / (w! (M-w)!), then for each q <= n
//   out[q] = sum_{w<q} A[w] V[q][w] / (q-w-1)!   ( = sum_w C(M-w-1, M-q) C(M, w) V[q][w] (M-q)! / M! ).
struct QueryA {
    Ctx c;
    int n, M;
    DEV void operator()(size_t idx) const {
        const size_t b = idx / n, w = idx % n;
        const Mod m = c.mod[b];
        const u64 *f = c.fact + b * c.F1, *fi = c.ifact + b * c.F1;
        c.A[b * c.NA + w] = mmul(mmul(f[M - w - 1], fi[w], m), fi[M - w], m);
    }
};
struct QueryInner {
    Ctx c;
    int n;
    DEV void operator()(size_t idx) const {
        const size_t b = idx / n, q = 1 + idx % n;
        const Mod m = c.mod[b];
        const u64 *fi = c.ifact + b * c.F1, *A = c.A + b * c.NA, *row = c.V + b * c.T + offV(q);
        u64 s = 0;
        for (size_t w = 0; w < q; ++w) s = madd(s, mmul(mmul(A[w], fi[q - w - 1], m), row[w], m), m);
        c.qout[b * c.NA + q - 1] = s;
    }
};

// ----------------------------------------------------------------------------------------
// Execution back ends
// ----------------------------------------------------------------------------------------
#ifdef __CUDACC__
static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class F>
__global__ void kernel_body(F f, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f(i);
}
template <class F>
static void launch(const F& f, size_t n) {
    if (n == 0) return;
    kernel_body<F><<<(unsigned)((n + 255) / 256), 256>>>(f, n);
}
static void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes ? bytes : 8), "cudaMalloc");
    cuda_check(cudaMemset(p, 0, bytes ? bytes : 8), "cudaMemset");
    return p;
}
static void to_dev(void* d, const void* h, size_t bytes) { cuda_check(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "upload"); }
static void from_dev(void* h, const void* d, size_t bytes) { cuda_check(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost), "download"); }
static void dev_sync() { cuda_check(cudaDeviceSynchronize(), "sync"); }
static size_t dev_free_bytes() {
    size_t fr = 0, tot = 0;
    cuda_check(cudaMemGetInfo(&fr, &tot), "meminfo");
    return fr;
}
static const char* backend_name() { return "cuda"; }
#else
template <class F>
static void launch(const F& f, size_t n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) f(i);
}
static void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0) {
        std::fprintf(stderr, "out of memory (%zu bytes)\n", bytes);
        std::exit(1);
    }
    std::memset(p, 0, bytes ? bytes : 64);
    return p;
}
static void to_dev(void* d, const void* h, size_t bytes) { std::memcpy(d, h, bytes); }
static void from_dev(void* h, const void* d, size_t bytes) { std::memcpy(h, d, bytes); }
static void dev_sync() {}
[[maybe_unused]] static size_t dev_free_bytes() { return (size_t)1 << 62; }
static const char* backend_name() { return USE_IFMA ? "host-ifma" : "host"; }
#endif

#if USE_IFMA
#include "ifma.inc"
#endif

// ----------------------------------------------------------------------------------------
// Host side: primes, tables, driver
// ----------------------------------------------------------------------------------------
typedef unsigned __int128 u128;
static u64 h_mulmod(u64 a, u64 b, u64 p) { return (u64)((u128)a * b % p); }
static u64 h_powmod(u64 a, u64 e, u64 p) {
    u64 r = 1;
    a %= p;
    while (e) {
        if (e & 1) r = h_mulmod(r, a, p);
        a = h_mulmod(a, a, p);
        e >>= 1;
    }
    return r;
}
static bool is_prime_u64(u64 n) {  // deterministic Miller-Rabin for n < 2^64
    if (n < 2) return false;
    for (u64 q : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull})
        if (n % q == 0) return n == q;
    u64 d = n - 1;
    int s = 0;
    while ((d & 1) == 0) d >>= 1, ++s;
    for (u64 a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        u64 x = h_powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int i = 1; i < s && comp; ++i) {
            x = h_mulmod(x, x, n);
            if (x == n - 1) comp = false;
        }
        if (comp) return false;
    }
    return true;
}
// Primes number first .. first+count-1 in the fixed list (largest first).
static std::vector<u64> prime_range(long first, long count) {
    std::vector<u64> out;
    long idx = 0;
    for (u64 c = ((1ull << PRIME_BITS) - 2) >> TWO_ADIC; c >= 1 && (long)out.size() < count; --c) {
        const u64 p = (c << TWO_ADIC) + 1;
        if (!is_prime_u64(p)) continue;
        if (idx++ >= first) out.push_back(p);
    }
    return out;
}
static Mod make_mod(u64 p) {
    Mod m;
    m.p = p;
    u64 x = p;  // Newton: p^{-1} mod 2^64
    for (int i = 0; i < 6; ++i) x *= 2 - p * x;
#if FP_RB == 52
    m.pinv = x & ((1ull << 52) - 1);
    m.one = (u64)(((u128)1 << 52) % p);
#else
    m.pinv = x;
    m.one = (u64)(((u128)1 << 64) % p);
#endif
    m.linv = 0;
    return m;
}
static u64 mont_r2(const Mod& m) { return h_mulmod(m.one, m.one, m.p); }
static u64 to_mont(u64 x, const Mod& m, u64 r2) { return mmul(x % m.p, r2, m); }
static u64 from_mont(u64 x, const Mod& m) { return mmul(x, 1, m); }

static double log2_Q(int M, int d) {
    long double b = 0;
    for (int i = 0; i < d; ++i) b += std::log2((long double)(M - i)) - std::log2((long double)(i + 1));
    return (double)b;
}
static long need(int n, int M, int d) {  // every prime used is > 2^(PRIME_BITS-1) (checked in main)
    return (long)std::ceil(((long double)n * log2_Q(M, d) + 3) / (PRIME_BITS - 1)) + 1;
}
static size_t next_pow2(size_t n) {
    size_t L = 1;
    while (L < n) L <<= 1;
    return L;
}

static void ntt_forward(const Ctx& c, size_t B) {
#if USE_IFMA
    ifma_ntt_forward(c, B);
    return;
#endif
    int rem = 0;
    while (((size_t)1 << rem) < c.L) ++rem;
    size_t H = c.L >> 1;
    while (rem > 0) {
        const int r = rem >= 3 ? 3 : rem;
        const size_t s = H >> (r - 1);
        int ls = 0;
        while (((size_t)1 << ls) < s) ++ls;
        launch(NttFwd{c, s, r, ls}, B * (c.L >> r));
        H = s >> 1;
        rem -= r;
    }
}
static void ntt_inverse(const Ctx& c, size_t B) {
#if USE_IFMA
    ifma_ntt_inverse(c, B);
    return;
#endif
    int rem = 0;
    while (((size_t)1 << rem) < c.L) ++rem;
    size_t s = 1;
    int ls = 0;
    while (rem > 0) {
        const int r = rem >= 3 ? 3 : rem;
        launch(NttInv{c, s, r, ls}, B * (c.L >> r));
        s <<= r;
        ls += r;
        rem -= r;
    }
}

struct Query {
    int n, M;
    long need;
};

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: cuckoo_fast D QUERIES FIRST COUNT [--batch B] [--out FILE]\n");
        return 2;
    }
    const int d = std::atoi(argv[1]);
    const long first = std::atol(argv[3]), count = std::atol(argv[4]);
    long batch = 0;
    const char* out_path = nullptr;
    for (int i = 5; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    std::vector<Query> qs;
    int nmax = 0, Mmax = 0;
    long maxneed = 0;
    {
        FILE* f = std::fopen(argv[2], "r");
        if (!f) {
            std::fprintf(stderr, "cannot open %s\n", argv[2]);
            return 2;
        }
        int n, M;
        while (std::fscanf(f, "%d %d", &n, &M) == 2) {
            if (n < 1 || M <= n || d < 1) {
                std::fprintf(stderr, "bad query %d %d\n", n, M);
                return 2;
            }
            qs.push_back({n, M, need(n, M, d)});
            nmax = std::max(nmax, n);
            Mmax = std::max(Mmax, M);
            maxneed = std::max(maxneed, qs.back().need);
        }
        std::fclose(f);
    }
    if (qs.empty() || nmax < 8 || nmax + 1 > (1 << TWO_ADIC)) {
        std::fprintf(stderr, "need queries with 8 <= max n < 2^%d\n", TWO_ADIC);
        return 2;
    }

    // primes already finished in the output file
    std::set<long> done;
    FILE* out = stdout;
    if (out_path) {
        if (FILE* f = std::fopen(out_path, "r")) {
            std::vector<long> cnt(first + count > 0 ? first + count : 1, 0);
            long i;
            unsigned long long p, r;
            int n, M;
            while (std::fscanf(f, "%ld %llu %d %d %llu", &i, &p, &n, &M, &r) == 5)
                if (i >= 0 && i < (long)cnt.size()) ++cnt[i];
            std::fclose(f);
            for (long i2 = first; i2 < first + count; ++i2) {
                long want = 0;
                for (const Query& q : qs) want += i2 < q.need;
                if (want > 0 && cnt[i2] == want) done.insert(i2);
            }
        }
        out = std::fopen(out_path, "a");
        if (!out) {
            std::fprintf(stderr, "cannot append to %s\n", out_path);
            return 2;
        }
    }
    std::vector<long> todo;
    for (long i = first; i < first + count && i < maxneed; ++i)
        if (!done.count(i)) todo.push_back(i);
    if (todo.empty()) {
        std::fprintf(stderr, "nothing to do (%zu primes already in the output)\n", done.size());
        return 0;
    }
    const std::vector<u64> primes = prime_range(first, count);  // primes[i - first]
    if (primes.empty() || primes.back() < (1ull << (PRIME_BITS - 1))) {
        std::fprintf(stderr, "prime list exhausted above 2^%d\n", PRIME_BITS - 1);
        return 4;
    }

    // sizes for the largest table, then the batch size
    const size_t Lmax = std::max<size_t>(16, next_pow2((size_t)nmax + 1));
    const size_t F1 = (size_t)std::max(nmax, Mmax) + 1;
    const size_t Tmax = offV((size_t)nmax + 1), TSmax = offS((size_t)nmax);
    int h = 0;
    while ((1 << (2 * h)) < nmax + 1) ++h;  // LO ~ sqrt(nmax)
    const size_t LO = (size_t)1 << h, HI = ((size_t)nmax >> h) + 1, NA = (size_t)nmax;
    const size_t per_prime = 8 * (2 * F1 + 4 * Lmax + Tmax + TSmax + 2 * (LO + HI) + 2 * NA) + sizeof(Mod);
    if (batch <= 0) {
#ifdef __CUDACC__
        batch = (long)((dev_free_bytes() - (dev_free_bytes() >> 4)) / per_prime);
        if (batch > 64) batch = 64;
#else
        batch = 1;
#endif
    }
    if (batch < 1) {
        std::fprintf(stderr, "not enough device memory: %.2f GB per prime\n", per_prime / 1073741824.0);
        return 3;
    }
    const size_t B = (size_t)std::min<long>(batch, (long)todo.size());
    std::fprintf(stderr, "[%s] R=2^%d, %d-bit primes, d=%d, nmax=%d, L=%zu, %.3f GB per prime, batch %zu, %zu primes to do\n",
                 backend_name(), FP_RB, PRIME_BITS, d, nmax, Lmax, per_prime / 1073741824.0, B, todo.size());

    Ctx c;
    c.mod = (Mod*)dev_alloc(B * sizeof(Mod));
    c.fact = (u64*)dev_alloc(8 * B * F1);
    c.ifact = (u64*)dev_alloc(8 * B * F1);
    c.tw = (u64*)dev_alloc(8 * B * Lmax);
    c.twinv = (u64*)dev_alloc(8 * B * Lmax);
    c.hat = (u64*)dev_alloc(8 * B * Lmax);
    c.buf = (u64*)dev_alloc(8 * B * Lmax);
    c.V = (u64*)dev_alloc(8 * B * Tmax);
    c.S = (u64*)dev_alloc(8 * B * TSmax);
    c.glo = (u64*)dev_alloc(8 * B * LO);
    c.ghi = (u64*)dev_alloc(8 * B * HI);
    c.gilo = (u64*)dev_alloc(8 * B * LO);
    c.gihi = (u64*)dev_alloc(8 * B * HI);
    c.A = (u64*)dev_alloc(8 * B * NA);
    c.qout = (u64*)dev_alloc(8 * B * NA);
    c.F1 = F1, c.T = Tmax, c.TS = TSmax, c.LO = LO, c.HI = HI, c.NA = NA, c.h = h, c.d = d;

    std::vector<Mod> hmod(B);
    std::vector<u64> hfact(B * F1), hifact(B * F1), htw, htwinv, hker, hq(B * NA);
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    for (size_t at = 0; at < todo.size(); at += B) {
        const size_t nb = std::min(B, todo.size() - at);
        // largest n that still needs any prime of this batch
        int nm = 0;
        for (const Query& q : qs)
            if (todo[at] < q.need) nm = std::max(nm, q.n);
        const size_t L = std::max<size_t>(16, next_pow2((size_t)nm + 1));
        c.L = L, c.nmax = nm, c.lg = 0;
        while (((size_t)1 << c.lg) < L) ++c.lg;
        htw.assign(B * L, 0), htwinv.assign(B * L, 0), hker.assign(B * L, 0);
        for (size_t b = 0; b < B; ++b) {
            const u64 p = primes[todo[at + std::min(b, nb - 1)] - first];  // pad the last batch
            Mod m = make_mod(p);
            const u64 r2 = mont_r2(m);
            m.linv = to_mont(h_powmod(L % p, p - 2, p), m, r2);
            hmod[b] = m;
            u64 *f = &hfact[b * F1], *fi = &hifact[b * F1];
            f[0] = m.one;
            u64 im = 0;  // i in Montgomery form
            for (size_t i = 1; i < F1; ++i) {
                im = madd(im, m.one, m);
                f[i] = mmul(f[i - 1], im, m);
            }
            fi[F1 - 1] = mpow(f[F1 - 1], p - 2, m);
            for (size_t i = F1 - 1; i > 0; --i) {
                fi[i - 1] = mmul(fi[i], im, m);
                im = msub(im, m.one, m);
            }
            // a root of unity of order L: x^((p-1)/L) with (.)^(L/2) = -1
            u64 w = 0;
            for (u64 x = 2;; ++x) {
                w = mpow(to_mont(x, m, r2), (p - 1) / L, m);
                if (mpow(w, L / 2, m) == p - m.one) break;
            }
            const u64 wi = mpow(w, p - 2, m);
            for (size_t half = 1; half < L; half <<= 1) {
                const u64 wh = mpow(w, L / (2 * half), m), whi = mpow(wi, L / (2 * half), m);
                u64 *t = &htw[b * L + half], *ti = &htwinv[b * L + half];
                t[0] = ti[0] = m.one;
                for (size_t j = 1; j < half; ++j) t[j] = mmul(t[j - 1], wh, m), ti[j] = mmul(ti[j - 1], whi, m);
            }
            for (int t = 1; t <= nm; ++t) hker[b * L + t] = mmul(fi[t], f[t - 1], m);  // 1/t
        }
        to_dev(c.mod, hmod.data(), B * sizeof(Mod));
        to_dev(c.fact, hfact.data(), 8 * B * F1);
        to_dev(c.ifact, hifact.data(), 8 * B * F1);
        to_dev(c.tw, htw.data(), 8 * B * L);
        to_dev(c.twinv, htwinv.data(), 8 * B * L);
        to_dev(c.buf, hker.data(), 8 * B * L);
        ntt_forward(c, B);
        launch(HatFinish{c}, B * L);

        for (int v = 1; v < nm; ++v) {
            if (v > d) {  // stage 1 (rows v <= d of V are zero, so S[.][v] stays zero)
                launch(Stage1Pre{c, v}, B * L);
                ntt_forward(c, B);
                launch(Pointwise{c}, B * L);
                ntt_inverse(c, B);
                launch(Stage1Post{c, v}, B * L);
            }
            if (v >= d) {  // stage 2 (g(v) = 0 below d: column v of V stays zero)
                launch(PowTab{c, v}, B * 2 * (LO + HI));
                launch(Stage2Pre{c, v}, B * L);
                ntt_forward(c, B);
                launch(Pointwise{c}, B * L);
                ntt_inverse(c, B);
                launch(Stage2Post{c, v}, B * L);
            }
        }
        dev_sync();
        const double t_build = secs();

        std::vector<std::string> lines(nb);
        for (const Query& q : qs) {
            if (todo[at] >= q.need) continue;  // need is the same test for the whole batch only if
            launch(QueryA{c, q.n, q.M}, B * (size_t)q.n);
            launch(QueryInner{c, q.n}, B * (size_t)q.n);
            from_dev(hq.data(), c.qout, 8 * B * NA);
            for (size_t b = 0; b < nb; ++b) {
                const long i = todo[at + b];
                if (i >= q.need) continue;
                const Mod m = hmod[b];
                const u64 *f = &hfact[b * F1], *fi = &hifact[b * F1], *o = &hq[b * NA];
                const u64 Q = mmul(mmul(f[q.M], fi[d], m), fi[q.M - d], m);
                u64 total = 0;  // Horner in Q: sum_q C(n,q) Q^(n-q) out[q] / (M-q)!
                for (int k = 1; k <= q.n; ++k) {
                    const u64 bin = mmul(mmul(f[q.n], fi[k], m), fi[q.n - k], m);
                    total = madd(mmul(total, Q, m), mmul(mmul(bin, fi[q.M - k], m), o[k - 1], m), m);
                }
                total = mmul(total, f[q.M], m);
                char line[160];
                std::snprintf(line, sizeof line, "%ld %llu %d %d %llu\n", i, (unsigned long long)m.p, q.n, q.M,
                              (unsigned long long)from_mont(total, m));
                lines[b] += line;
            }
        }
        for (size_t b = 0; b < nb; ++b) std::fputs(lines[b].c_str(), out);
        std::fflush(out);
        std::fprintf(stderr, "primes %ld..%ld (n<=%d): build done at %.1f s, queries at %.1f s\n", todo[at], todo[at + nb - 1],
                     nm, t_build, secs());
    }
    if (out != stdout) std::fclose(out);
    return 0;
}
