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

// ---- transforms: one level at a time over the whole buffer ----
static void ifma_ntt_forward(const Ctx& c, size_t B) {
    const size_t L = c.L;
    for (size_t b = 0; b < B; ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        u64* a = c.buf + b * L;
        const u64* tw = c.tw + b * L;
        for (size_t half = L >> 1; half >= 8; half >>= 1)
            for (size_t i = 0; i < L; i += 2 * half)
                for (size_t j = 0; j < half; j += 8) {
                    const V8 u = vload(a + i + j), w = vload(a + i + j + half);
                    vstore(a + i + j, vadd(u, w, vm));
                    vstore(a + i + j + half, vmul(vsub(u, w, vm), vload(tw + half + j), vm));
                }
        // halves 4, 2, 1 inside each vector: x = [u | w] by 256-bit halves, 128-bit pairs, lanes
        const V8 t4 = _mm512_set_epi64((long long)tw[7], (long long)tw[6], (long long)tw[5], (long long)tw[4], 0, 0, 0, 0);
        const V8 t2 = _mm512_set_epi64((long long)tw[3], (long long)tw[2], 0, 0, (long long)tw[3], (long long)tw[2], 0, 0);
        for (size_t i = 0; i < L; i += 8) {
            V8 x = vload(a + i);
            V8 sw = _mm512_shuffle_i64x2(x, x, 0x4E);
            x = _mm512_mask_blend_epi64(0xF0, vadd(x, sw, vm), vmul(vsub(sw, x, vm), t4, vm));
            sw = _mm512_permutex_epi64(x, 0x4E);
            x = _mm512_mask_blend_epi64(0xCC, vadd(x, sw, vm), vmul(vsub(sw, x, vm), t2, vm));
            sw = _mm512_permutex_epi64(x, 0xB1);
            x = _mm512_mask_blend_epi64(0xAA, vadd(x, sw, vm), vsub(sw, x, vm));
            vstore(a + i, x);
        }
    }
}
static void ifma_ntt_inverse(const Ctx& c, size_t B) {
    const size_t L = c.L;
    for (size_t b = 0; b < B; ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        u64* a = c.buf + b * L;
        const u64* tw = c.twinv + b * L;
        const V8 t4 = _mm512_set_epi64((long long)tw[7], (long long)tw[6], (long long)tw[5], (long long)tw[4], 0, 0, 0, 0);
        const V8 t2 = _mm512_set_epi64((long long)tw[3], (long long)tw[2], 0, 0, (long long)tw[3], (long long)tw[2], 0, 0);
        for (size_t i = 0; i < L; i += 8) {
            V8 x = vload(a + i);
            V8 sw = _mm512_permutex_epi64(x, 0xB1);  // half 1: twiddle 1
            x = _mm512_mask_blend_epi64(0xAA, vadd(x, sw, vm), vsub(sw, x, vm));
            x = _mm512_mask_blend_epi64(0xCC, x, vmul(x, t2, vm));  // half 2
            sw = _mm512_permutex_epi64(x, 0x4E);
            x = _mm512_mask_blend_epi64(0xCC, vadd(x, sw, vm), vsub(sw, x, vm));
            x = _mm512_mask_blend_epi64(0xF0, x, vmul(x, t4, vm));  // half 4
            sw = _mm512_shuffle_i64x2(x, x, 0x4E);
            x = _mm512_mask_blend_epi64(0xF0, vadd(x, sw, vm), vsub(sw, x, vm));
            vstore(a + i, x);
        }
        for (size_t half = 8; half < L; half <<= 1)
            for (size_t i = 0; i < L; i += 2 * half)
                for (size_t j = 0; j < half; j += 8) {
                    const V8 u = vload(a + i + j), w = vmul(vload(a + i + j + half), vload(tw + half + j), vm);
                    vstore(a + i + j, vadd(u, w, vm));
                    vstore(a + i + j + half, vsub(u, w, vm));
                }
    }
}

static inline void launch(const Pointwise& f, size_t n) {
    const Ctx& c = f.c;
    for (size_t b = 0; b < n / c.L; ++b) {
        const VMod vm(c.mod[b]);
        u64* a = c.buf + b * c.L;
        const u64* hat = c.hat + b * c.L;
        for (size_t i = 0; i < c.L; i += 8) vstore(a + i, vmul(vload(a + i), vload(hat + i), vm));
    }
}
static inline void launch(const Stage1Pre& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v;
    for (size_t b = 0; b < n / c.L; ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fi = c.ifact + b * c.F1, *row = c.V + b * c.T + offV(v);
        u64* a = c.buf + b * c.L;
        size_t i = 0;
        for (; i + 8 <= v; i += 8)
            vstore(a + i, vmul(vmul(vload(row + i), vload(fi + i), vm), vrev(vload(fi + v - 8 - i)), vm));
        for (; i < v; ++i) a[i] = mmul(mmul(row[i], fi[i], m), fi[v - 1 - i], m);
        std::memset(a + v, 0, 8 * (c.L - v));
    }
}
static inline void launch(const Stage1Post& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v, cnt = (size_t)c.nmax - v;
    for (size_t b = 0; b < (n >> c.lg); ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fa = c.fact + b * c.F1, *fi = c.ifact + b * c.F1, *a = c.buf + b * c.L;
        u64* S = c.S + b * c.TS;
        const V8 lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), vv = _mm512_set1_epi64((long long)(v - 1));
        size_t vp = v;
        for (; vp + 8 <= (size_t)c.nmax; vp += 8) {
            const V8 val = vmul(vmul(vload(fa + vp), vload(fi + vp - v), vm), vload(a + vp), vm);
            const V8 pv = _mm512_add_epi64(_mm512_set1_epi64((long long)vp), lane);
            const V8 off = _mm512_srli_epi64(_mm512_mullo_epi64(pv, _mm512_sub_epi64(pv, _mm512_set1_epi64(1))), 1);
            _mm512_i64scatter_epi64((void*)S, _mm512_add_epi64(off, vv), val, 8);
        }
        for (; vp < (size_t)c.nmax; ++vp) S[offS(vp) + v - 1] = mmul(mmul(fa[vp], fi[vp - v], m), a[vp], m);
    }
}
// The host build does not need the two-level power tables: Stage2Pre/Post run power chains.
static inline void launch(const PowTab&, size_t) {}
static std::vector<u64> g_pw;  // scratch for the chains
static inline void launch(const Stage2Pre& f, size_t n) {
    const Ctx& c = f.c;
    const size_t v = (size_t)f.v;
    if (g_pw.size() < c.L + 16) g_pw.resize(c.L + 16);
    for (size_t b = 0; b < n / c.L; ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fa = c.fact + b * c.F1, *fi = c.ifact + b * c.F1, *row = c.S + b * c.TS + offS(v);  // row[q-1]
        u64* a = c.buf + b * c.L;
        const u64 ginv = mpow(mmul(mmul(fa[v], fi[c.d], m), fi[v - c.d], m), m.p - 2, m);
        u64* pw = g_pw.data();  // pw[k] = ginv^(k+1)
        pow_chain(pw, ginv, 1, v, m);
        a[0] = 0;
        size_t k = 0;  // q = k + 1
        for (; k + 8 <= v; k += 8) {
            V8 t = vmul(vload(row + k), vload(pw + k), vm);
            t = vmul(t, vload(fi + k + 1), vm);
            vstore(a + k + 1, vmul(t, vrev(vload(fi + v - 8 - k)), vm));  // 1 / (v - q)!, q = k+1 .. k+8
        }
        for (; k < v; ++k) a[k + 1] = mmul(mmul(mmul(row[k], pw[k], m), fi[k + 1], m), fi[v - 1 - k], m);
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
        u64* Vt = c.V + b * c.T;
        const u64 g = mmul(mmul(fa[v], fi[c.d], m), fi[v - c.d], m);
        u64* pw = g_pw.data();  // pw[k] = g^(v+1+k), p = v + 1 + k
        pow_chain(pw, g, v + 1, cnt, m);
        const V8 lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), vv = _mm512_set1_epi64((long long)v);
        const V8 fiv = _mm512_set1_epi64((long long)fi[v]);
        size_t k = 0;
        for (; k + 8 <= cnt; k += 8) {
            const size_t p = v + 1 + k;
            V8 t = vsub(vmul(vload(fa + p - 1), fiv, vm), vmul(vload(fa + p), vload(a + p), vm), vm);
            t = vmul(vmul(t, vload(fi + k), vm), vload(pw + k), vm);  // 1 / (p-v-1)! = fi[k]
            const V8 pv = _mm512_add_epi64(_mm512_set1_epi64((long long)p), lane);
            const V8 off = _mm512_srli_epi64(_mm512_mullo_epi64(pv, _mm512_sub_epi64(pv, _mm512_set1_epi64(1))), 1);
            _mm512_i64scatter_epi64((void*)Vt, _mm512_add_epi64(off, vv), t, 8);
        }
        for (; k < cnt; ++k) {
            const size_t p = v + 1 + k;
            u64 t = msub(mmul(fa[p - 1], fi[v], m), mmul(fa[p], a[p], m), m);
            Vt[offV(p) + v] = mmul(mmul(t, fi[k], m), pw[k], m);
        }
    }
}
static inline void launch(const QueryInner& f, size_t n) {
    const Ctx& c = f.c;
    const size_t nq = (size_t)f.n;
    for (size_t b = 0; b < n / nq; ++b) {
        const Mod m = c.mod[b];
        const VMod vm(m);
        const u64 *fi = c.ifact + b * c.F1, *A = c.A + b * c.NA;
        for (size_t q = 1; q <= nq; ++q) {
            const u64* row = c.V + b * c.T + offV(q);
            V8 acc = _mm512_setzero_si512();
            size_t w = 0;
            for (; w + 8 <= q; w += 8)
                acc = vadd(acc, vmul(vmul(vload(A + w), vrev(vload(fi + q - 8 - w)), vm), vload(row + w), vm), vm);
            u64 lanes[8], s = 0;
            vstore(lanes, acc);
            for (int k = 0; k < 8; ++k) s = madd(s, lanes[k], m);
            for (; w < q; ++w) s = madd(s, mmul(mmul(A[w], fi[q - w - 1], m), row[w], m), m);
            c.qout[b * c.NA + q - 1] = s;
        }
    }
}
