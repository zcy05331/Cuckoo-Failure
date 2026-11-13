// AVX-512 IFMA versions of the hot loops of cuckoo_fast.cu (host build, R = 2^52, p < 2^50).
// Included from cuckoo_fast.cu; each function replaces the generic loop over one functor and
// produces exactly the same values, eight residues per instruction.

typedef __m512i V8;
struct VMod {
    V8 p, pinv;
    explicit VMod(const Mod& m) : p(_mm512_set1_epi64((long long)m.p)), pinv(_mm512_set1_epi64((long long)m.pinv)) {}
};
// a b R^{-1} mod p, lanes < p. madd52lo/hi give the low and high 52 bits of the 104-bit product.
static inline V8 vmul(V8 a, V8 b, const VMod& m) {
    const V8 z = _mm512_setzero_si512();
    const V8 lo = _mm512_madd52lo_epu64(z, a, b), hi = _mm512_madd52hi_epu64(z, a, b);
    const V8 q = _mm512_madd52lo_epu64(z, lo, m.pinv);
    const V8 r = _mm512_sub_epi64(hi, _mm512_madd52hi_epu64(z, q, m.p));  // in (-p, p)
    return _mm512_min_epu64(r, _mm512_add_epi64(r, m.p));
}
static inline V8 vadd(V8 a, V8 b, const VMod& m) {
    const V8 s = _mm512_add_epi64(a, b);
    return _mm512_min_epu64(s, _mm512_sub_epi64(s, m.p));
}
static inline V8 vsub(V8 a, V8 b, const VMod& m) {
    const V8 d = _mm512_sub_epi64(a, b);
    return _mm512_min_epu64(d, _mm512_add_epi64(d, m.p));
}
static inline V8 vload(const u64* p) { return _mm512_loadu_si512((const void*)p); }
static inline void vstore(u64* p, V8 x) { _mm512_storeu_si512((void*)p, x); }
static inline V8 vrev(V8 x) { return _mm512_permutexvar_epi64(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), x); }

// out[k] = x^(e0 + k), 0 <= k < cnt: eight interleaved chains stepping by x^8.
static void pow_chain(u64* out, u64 x, u64 e0, size_t cnt, const Mod& m) {
    if (cnt == 0) return;
    out[0] = mpow(x, e0, m);
    for (size_t k = 1; k < cnt && k < 8; ++k) out[k] = mmul(out[k - 1], x, m);
    if (cnt <= 8) return;
    const VMod vm(m);
    const V8 x8 = _mm512_set1_epi64((long long)mpow(x, 8, m));
    size_t k = 0;
    for (; k + 16 <= cnt; k += 8) vstore(out + k + 8, vmul(vload(out + k), x8, vm));
    for (k += 8; k < cnt; ++k) out[k] = mmul(out[k - 1], x, m);
}

// ---- butterfly building blocks ----
// Gentleman-Sande (forward, natural order in, bit-reversed out) and Cooley-Tukey (inverse, with
// inverse roots, bit-reversed in, natural out). tw[half + j] = w_{2 half}^j. Levels with half >= 8
// work on whole vectors; halves 4, 2, 1 work inside each vector.
static inline void fwd_r2(u64* a, size_t len, const u64* tw, size_t half, const VMod& vm) {
    for (size_t i = 0; i < len; i += 2 * half)
        for (size_t j = 0; j < half; j += 8) {
            const V8 u = vload(a + i + j), w = vload(a + i + j + half);
            vstore(a + i + j, vadd(u, w, vm));
            vstore(a + i + j + half, vmul(vsub(u, w, vm), vload(tw + half + j), vm));
        }
}
// Levels h and h/2 in one pass (h/2 >= 8): the same butterflies as two fwd_r2 passes.
static inline void fwd_r4(u64* a, size_t len, const u64* tw, size_t h, const VMod& vm) {
    const size_t q = h >> 1;
    for (size_t i = 0; i < len; i += 2 * h)
        for (size_t j = 0; j < q; j += 8) {
            const V8 x0 = vload(a + i + j), x1 = vload(a + i + j + q), x2 = vload(a + i + j + h), x3 = vload(a + i + j + h + q);
            const V8 y0 = vadd(x0, x2, vm), y2 = vmul(vsub(x0, x2, vm), vload(tw + h + j), vm);
            const V8 y1 = vadd(x1, x3, vm), y3 = vmul(vsub(x1, x3, vm), vload(tw + h + j + q), vm);
            const V8 t = vload(tw + q + j);
            vstore(a + i + j, vadd(y0, y1, vm));
            vstore(a + i + j + q, vmul(vsub(y0, y1, vm), t, vm));
            vstore(a + i + j + h, vadd(y2, y3, vm));
            vstore(a + i + j + h + q, vmul(vsub(y2, y3, vm), t, vm));
        }
}
static inline void inv_r2(u64* a, size_t len, const u64* tw, size_t half, const VMod& vm) {
    for (size_t i = 0; i < len; i += 2 * half)
        for (size_t j = 0; j < half; j += 8) {
            const V8 u = vload(a + i + j), w = vmul(vload(a + i + j + half), vload(tw + half + j), vm);
            vstore(a + i + j, vadd(u, w, vm));
            vstore(a + i + j + half, vsub(u, w, vm));
        }
}
// Levels h and 2h in one pass (h >= 8): the same butterflies as two inv_r2 passes.
static inline void inv_r4(u64* a, size_t len, const u64* tw, size_t h, const VMod& vm) {
    for (size_t i = 0; i < len; i += 4 * h)
        for (size_t j = 0; j < h; j += 8) {
            const V8 x0 = vload(a + i + j), x1 = vload(a + i + j + h), x2 = vload(a + i + j + 2 * h), x3 = vload(a + i + j + 3 * h);
            const V8 t1 = vload(tw + h + j);
            const V8 w1 = vmul(x1, t1, vm), w3 = vmul(x3, t1, vm);
            const V8 y0 = vadd(x0, w1, vm), y1 = vsub(x0, w1, vm), y2 = vadd(x2, w3, vm), y3 = vsub(x2, w3, vm);
            const V8 w2 = vmul(y2, vload(tw + 2 * h + j), vm), w4 = vmul(y3, vload(tw + 2 * h + j + h), vm);
            vstore(a + i + j, vadd(y0, w2, vm));
            vstore(a + i + j + 2 * h, vsub(y0, w2, vm));
            vstore(a + i + j + h, vadd(y1, w4, vm));
            vstore(a + i + j + 3 * h, vsub(y1, w4, vm));
        }
}
// Forward halves 4, 2, 1 inside one vector (x = [u | w] by 256-bit halves, 128-bit pairs, lanes).
static inline V8 fwd_tail(V8 x, V8 t4, V8 t2, const VMod& vm) {
    V8 sw = _mm512_shuffle_i64x2(x, x, 0x4E);
    x = _mm512_mask_blend_epi64(0xF0, vadd(x, sw, vm), vmul(vsub(sw, x, vm), t4, vm));
    sw = _mm512_permutex_epi64(x, 0x4E);
    x = _mm512_mask_blend_epi64(0xCC, vadd(x, sw, vm), vmul(vsub(sw, x, vm), t2, vm));
    sw = _mm512_permutex_epi64(x, 0xB1);
    return _mm512_mask_blend_epi64(0xAA, vadd(x, sw, vm), vsub(sw, x, vm));
}
// Inverse halves 1, 2, 4 inside one vector.
static inline V8 inv_head(V8 x, V8 t4, V8 t2, const VMod& vm) {
    V8 sw = _mm512_permutex_epi64(x, 0xB1);  // half 1: twiddle 1
    x = _mm512_mask_blend_epi64(0xAA, vadd(x, sw, vm), vsub(sw, x, vm));
    x = _mm512_mask_blend_epi64(0xCC, x, vmul(x, t2, vm));  // half 2
    sw = _mm512_permutex_epi64(x, 0x4E);
    x = _mm512_mask_blend_epi64(0xCC, vadd(x, sw, vm), vsub(sw, x, vm));
    x = _mm512_mask_blend_epi64(0xF0, x, vmul(x, t4, vm));  // half 4
    sw = _mm512_shuffle_i64x2(x, x, 0x4E);
    return _mm512_mask_blend_epi64(0xF0, vadd(x, sw, vm), vsub(sw, x, vm));
}
static inline V8 tw4(const u64* tw) { return _mm512_set_epi64((long long)tw[7], (long long)tw[6], (long long)tw[5], (long long)tw[4], 0, 0, 0, 0); }
static inline V8 tw2(const u64* tw) { return _mm512_set_epi64((long long)tw[3], (long long)tw[2], 0, 0, (long long)tw[3], (long long)tw[2], 0, 0); }

// ---- whole transforms (used once per batch, for the kernel) ----
static void ifma_ntt_forward(const Ctx& c, size_t B) {
    const size_t L = c.L;
    for (size_t b = 0; b < B; ++b) {
        const VMod vm(c.mod[b]);
        u64* a = c.buf + b * L;
        const u64* tw = c.tw + b * L;
        for (size_t half = L >> 1; half >= 8; half >>= 1) fwd_r2(a, L, tw, half, vm);
        const V8 t4 = tw4(tw), t2 = tw2(tw);
        for (size_t i = 0; i < L; i += 8) vstore(a + i, fwd_tail(vload(a + i), t4, t2, vm));
    }
}
static void ifma_ntt_inverse(const Ctx& c, size_t B) {
    const size_t L = c.L;
    for (size_t b = 0; b < B; ++b) {
        const VMod vm(c.mod[b]);
        u64* a = c.buf + b * L;
        const u64* tw = c.twinv + b * L;
        const V8 t4 = tw4(tw), t2 = tw2(tw);
        for (size_t i = 0; i < L; i += 8) vstore(a + i, inv_head(vload(a + i), t4, t2, vm));
        for (size_t half = 8; half < L; half <<= 1) inv_r2(a, L, tw, half, vm);
    }
}

// ---- cyclic convolution of buf with the kernel, cache-blocked ----
// The forward levels with half >= BLK run over the whole buffer, two levels per pass. After them
// every block of BLK entries is independent until the inverse reaches half = BLK, so each block
// does its remaining forward levels, the pointwise product and its first inverse levels while it
// sits in L1. Then the inverse levels with half >= BLK run over the whole buffer, two per pass.
// Every butterfly is the one ifma_ntt_forward / pointwise / ifma_ntt_inverse would perform, so
// the result is identical; only the order of independent butterflies changes.
static void ifma_convolve(const Ctx& c, size_t B) {
    const size_t L = c.L, BLK = L < 4096 ? L : 4096;  // 4096 entries = 32 KB
    for (size_t b = 0; b < B; ++b) {
        const VMod vm(c.mod[b]);
        u64* a = c.buf + b * L;
        const u64 *tw = c.tw + b * L, *ti = c.twinv + b * L, *hat = c.hat + b * L;
        const V8 f4 = tw4(tw), f2 = tw2(tw), i4 = tw4(ti), i2 = tw2(ti);
        size_t h = L >> 1;
        while (h >= BLK) {
            if ((h >> 1) >= BLK) fwd_r4(a, L, tw, h, vm), h >>= 2;
            else fwd_r2(a, L, tw, h, vm), h >>= 1;
        }
        for (size_t base = 0; base < L; base += BLK) {
            u64* blk = a + base;
            for (h = BLK >> 1; h >= 8;) {
                if ((h >> 1) >= 8) fwd_r4(blk, BLK, tw, h, vm), h >>= 2;
                else fwd_r2(blk, BLK, tw, h, vm), h >>= 1;
            }
            for (size_t i = 0; i < BLK; i += 8) {
                V8 x = fwd_tail(vload(blk + i), f4, f2, vm);
                x = vmul(x, vload(hat + base + i), vm);
                vstore(blk + i, inv_head(x, i4, i2, vm));
            }
            for (h = 8; h < BLK;) {
                if ((h << 1) < BLK) inv_r4(blk, BLK, ti, h, vm), h <<= 2;
                else inv_r2(blk, BLK, ti, h, vm), h <<= 1;
            }
        }
        for (h = BLK; h < L;) {
            if ((h << 1) < L) inv_r4(a, L, ti, h, vm), h <<= 2;
            else inv_r2(a, L, ti, h, vm), h <<= 1;
        }
    }
}

// ---- stage inputs and outputs ----
// Columns v0 .. v of S and V (the current tile) live in the column-major buffers SS and VS
// (entry (row r, column v0 + t) at t NS + r); FlushTile moves them into the rows of S and V.
static inline void launch(const Stage1Pre& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v, v0 = (size_t)c.v0;
    for (size_t b = 0; b < (n >> c.lg); ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fi = c.ifact + b * c.F1, *row = c.V + b * c.T + offV(v), *vs = c.VS + b * TILE * c.NS;
        u64* a = c.buf + b * c.L;
        size_t i = 0;
        for (; i + 8 <= v0; i += 8)  // columns w < v0 are in the row of V
            vstore(a + i, vmul(vmul(vload(row + i), vload(fi + i), vm), vrev(vload(fi + v - 8 - i)), vm));
        for (; i < v0; ++i) a[i] = mmul(mmul(row[i], fi[i], m), fi[v - 1 - i], m);
        for (; i < v; ++i) a[i] = mmul(mmul(vs[(i - v0) * c.NS + v], fi[i], m), fi[v - 1 - i], m);
        std::memset(a + v, 0, 8 * (c.L - v));
    }
}
static inline void launch(const Stage1Post& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v, nmax = (size_t)c.nmax;
    for (size_t b = 0; b < (n >> c.lg); ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fa = c.fact + b * c.F1, *fi = c.ifact + b * c.F1, *a = c.buf + b * c.L;
        u64* col = c.SS + b * TILE * c.NS + (v - c.v0) * c.NS;  // S[vp][v] at col[vp]
        size_t vp = v;
        for (; vp + 8 <= nmax; vp += 8) vstore(col + vp, vmul(vmul(vload(fa + vp), vload(fi + vp - v), vm), vload(a + vp), vm));
        for (; vp < nmax; ++vp) col[vp] = mmul(mmul(fa[vp], fi[vp - v], m), a[vp], m);
    }
}
// The host build does not need the two-level power tables: Stage2Pre/Post run power chains.
static inline void launch(const PowTab&, size_t) {}
static std::vector<u64> g_pw;  // scratch for the chains
static inline void launch(const Stage2Pre& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v, v0 = (size_t)c.v0;
    if (g_pw.size() < c.L + 16) g_pw.resize(c.L + 16);
    for (size_t b = 0; b < (n >> c.lg); ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fa = c.fact + b * c.F1, *fi = c.ifact + b * c.F1, *row = c.S + b * c.TS + offS(v);  // row[q-1]
        const u64* ss = c.SS + b * TILE * c.NS;
        u64* a = c.buf + b * c.L;
        const u64 ginv = mpow(mmul(mmul(fa[v], fi[c.d], m), fi[v - c.d], m), m.p - 2, m);
        u64* pw = g_pw.data();  // pw[k] = ginv^(k+1)
        pow_chain(pw, ginv, 1, v, m);
        a[0] = 0;
        const size_t lim = v0 - 1;  // q = k + 1 < v0: the row of S; q >= v0: the tile
        size_t k = 0;
        for (; k + 8 <= lim; k += 8) {
            V8 t = vmul(vload(row + k), vload(pw + k), vm);
            t = vmul(t, vload(fi + k + 1), vm);
            vstore(a + k + 1, vmul(t, vrev(vload(fi + v - 8 - k)), vm));  // 1 / (v - q)!, q = k+1 .. k+8
        }
        for (; k < lim; ++k) a[k + 1] = mmul(mmul(mmul(row[k], pw[k], m), fi[k + 1], m), fi[v - 1 - k], m);
        for (; k < v; ++k)
            a[k + 1] = mmul(mmul(mmul(ss[(k + 1 - v0) * c.NS + v], pw[k], m), fi[k + 1], m), fi[v - 1 - k], m);
        std::memset(a + v + 1, 0, 8 * (c.L - v - 1));
    }
}
static inline void launch(const Stage2Post& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v, cnt = (size_t)c.nmax - v;
    if (g_pw.size() < c.L + 16) g_pw.resize(c.L + 16);
    for (size_t b = 0; b < (n >> c.lg); ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fa = c.fact + b * c.F1, *fi = c.ifact + b * c.F1, *a = c.buf + b * c.L;
        u64* col = c.VS + b * TILE * c.NS + (v - c.v0) * c.NS;  // V[p][v] at col[p]
        const u64 g = mmul(mmul(fa[v], fi[c.d], m), fi[v - c.d], m);
        u64* pw = g_pw.data();  // pw[k] = g^(v+1+k), p = v + 1 + k
        pow_chain(pw, g, v + 1, cnt, m);
        const V8 fiv = _mm512_set1_epi64((long long)fi[v]);
        size_t k = 0;
        for (; k + 8 <= cnt && v + 1 + k + 8 <= c.L; k += 8) {  // p + 7 < L; c[nmax] = a[0] when L = nmax
            const size_t p = v + 1 + k;
            V8 t = vsub(vmul(vload(fa + p - 1), fiv, vm), vmul(vload(fa + p), vload(a + p), vm), vm);
            vstore(col + p, vmul(vmul(t, vload(fi + k), vm), vload(pw + k), vm));  // 1 / (p-v-1)! = fi[k]
        }
        for (; k < cnt; ++k) {
            const size_t p = v + 1 + k;
            u64 t = msub(mmul(fa[p - 1], fi[v], m), mmul(fa[p], a[p & (c.L - 1)], m), m);
            col[p] = mmul(mmul(t, fi[k], m), pw[k], m);
        }
    }
}
// All queries in one pass over V (see QueryAll): per row, U[w] = V[q][w] / (w! (q-w-1)!) once,
// then one dot product with 1/(M - w) per query.
static std::vector<u64> g_u;
static inline void launch(const QueryAll& f, size_t n) {
    const Ctx& c = f.c;
    const size_t B = n / (c.NA * c.J);
    size_t qmax = 0;
    for (size_t j = 0; j < c.J; ++j) qmax = std::max<size_t>(qmax, c.qN[j]);
    if (g_u.size() < c.NA + 8) g_u.resize(c.NA + 8);
    u64* U = g_u.data();
    for (size_t b = 0; b < B; ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fi = c.ifact + b * c.F1, *inv = c.inv + b * c.F1;
        for (size_t q = 1; q <= qmax; ++q) {
            const u64* row = c.V + b * c.T + offV(q);
            size_t w = 0;
            for (; w + 8 <= q; w += 8) vstore(U + w, vmul(vmul(vload(row + w), vload(fi + w), vm), vrev(vload(fi + q - 8 - w)), vm));
            for (; w < q; ++w) U[w] = mmul(mmul(row[w], fi[w], m), fi[q - 1 - w], m);
            for (size_t j = 0; j < c.J; ++j) {
                if (q > c.qN[j]) continue;
                const size_t M = c.qM[j];
                V8 acc = _mm512_setzero_si512();
                for (w = 0; w + 8 <= q; w += 8) acc = vadd(acc, vmul(vload(U + w), vrev(vload(inv + M - w - 7)), vm), vm);
                u64 lanes[8], s = 0;
                vstore(lanes, acc);
                for (int k = 0; k < 8; ++k) s = madd(s, lanes[k], m);
                for (; w < q; ++w) s = madd(s, mmul(U[w], inv[M - w], m), m);
                c.qo[(b * c.J + j) * c.NA + q - 1] = s;
            }
        }
    }
}
