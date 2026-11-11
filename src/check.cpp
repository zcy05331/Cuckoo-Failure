// Correctness checks: brute force against the four algorithms, plus cross-checks
// between the algorithms on instances too large for brute force.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cuckoo/brute_force.hpp"
#include "cuckoo/crt.hpp"
#include "cuckoo/separate_tables.hpp"
#include "cuckoo/single_table.hpp"

using namespace cuckoo;

namespace {

int g_pass = 0, g_fail = 0;
const std::vector<uint32_t> kPrimes = {998244353u, 1004535809u};

void report(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (ok) ++g_pass; else ++g_fail;
}

std::string vec_str(const std::vector<int>& m) {
    std::string s = "(";
    for (size_t i = 0; i < m.size(); ++i) s += (i ? "," : "") + std::to_string(m[i]);
    return s + ")";
}

std::string mod_str(uint32_t P, const std::vector<std::pair<const char*, Mint>>& vals) {
    std::string s = " [mod " + std::to_string(P) + ":";
    for (const auto& kv : vals) s += std::string(" ") + kv.first + "=" + std::to_string(kv.second.val());
    return s + "]";
}

// One table: brute force vs. forward O(n^4), cubic O(n^3) and NTT O(n^2 log n).
void one_table_vs_brute(int n, int M, int d, bool rep) {
    const BruteResult br = brute_force_count(n, M, one_table_choices(M, d, rep));
    bool ok = true;
    std::string detail;
    for (uint32_t P : kPrimes) {
        Mint::set_mod(P);
        const Mint expect = Mint(static_cast<long long>(br.nbad % P));
        const OneTableModel model{d, rep};
        const Mint a = one_table_forward(n, M, model);
        const Mint b = build_one_table_suffix_cubic(n, model).nbad(n, M);
        const Mint c = build_one_table_suffix_ntt(n, model).nbad(n, M);
        if (a != expect || b != expect || c != expect) {
            ok = false;
            detail += mod_str(P, {{"brute", expect}, {"forward", a}, {"cubic", b}, {"ntt", c}});
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "one table  n=%d M=%d d=%d %-11s brute N_bad=%llu / %llu", n, M, d,
                  rep ? "replacement" : "distinct", br.nbad, br.total);
    report(ok, buf + detail);
}

// Separate tables: brute force vs. forward O(n^{2d+2}) and suffix O(n^{d+2}).
void separate_vs_brute(int n, const std::vector<int>& m) {
    int M = 0;
    for (int x : m) M += x;
    const BruteResult br = brute_force_count(n, M, separate_table_choices(m));
    bool ok = true;
    std::string detail;
    for (uint32_t P : kPrimes) {
        Mint::set_mod(P);
        const Mint expect = Mint(static_cast<long long>(br.nbad % P));
        const Mint a = separate_forward(n, m);
        const Mint b = build_separate_suffix(n, static_cast<int>(m.size())).nbad(n, m);
        if (a != expect || b != expect) {
            ok = false;
            detail += mod_str(P, {{"brute", expect}, {"forward", a}, {"suffix", b}});
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "separate   n=%d m=%s brute N_bad=%llu / %llu", n, vec_str(m).c_str(), br.nbad,
                  br.total);
    report(ok, buf + detail);
}

void one_table_cross(int n, int M, int d, bool rep) {
    bool ok = true;
    std::string detail;
    for (uint32_t P : kPrimes) {
        Mint::set_mod(P);
        const OneTableModel model{d, rep};
        const Mint a = one_table_forward(n, M, model);
        const Mint b = build_one_table_suffix_cubic(n, model).nbad(n, M);
        const Mint c = build_one_table_suffix_ntt(n, model).nbad(n, M);
        if (a != b || b != c) {
            ok = false;
            detail += mod_str(P, {{"forward", a}, {"cubic", b}, {"ntt", c}});
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "one table  n=%d M=%d d=%d %-11s forward = cubic = ntt", n, M, d,
                  rep ? "replacement" : "distinct");
    report(ok, buf + detail);
}

void separate_cross(int n, const std::vector<int>& m) {
    bool ok = true;
    std::string detail;
    for (uint32_t P : kPrimes) {
        Mint::set_mod(P);
        const Mint a = separate_forward(n, m);
        const Mint b = build_separate_suffix(n, static_cast<int>(m.size())).nbad(n, m);
        if (a != b) {
            ok = false;
            detail += mod_str(P, {{"forward", a}, {"suffix", b}});
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "separate   n=%d m=%s forward = suffix", n, vec_str(m).c_str());
    report(ok, buf + detail);
}

// d = 1: an item has one candidate slot, so N_bad = M^n - M!/(M-n)! (birthday
// problem). The one-table (both sampling models) and the separate-table
// algorithms must all agree with it.
void birthday(int n, int M) {
    bool ok = true;
    std::string detail;
    for (uint32_t P : kPrimes) {
        Mint::set_mod(P);
        Factorials fac(M);
        const Mint expect = Mint(M).pow(n) - fac.fact[M] * fac.inv_fact[M - n];
        const Mint a = one_table_forward(n, M, {1, false});
        const Mint b = build_one_table_suffix_ntt(n, {1, false}).nbad(n, M);
        const Mint c = one_table_forward(n, M, {1, true});
        const Mint e = separate_forward(n, {M});
        const Mint f = build_separate_suffix(n, 1).nbad(n, {M});
        if (a != expect || b != expect || c != expect || e != expect || f != expect) {
            ok = false;
            detail += mod_str(P, {{"formula", expect}, {"forward", a}, {"ntt", b}, {"replacement", c},
                                  {"sep-forward", e}, {"sep-suffix", f}});
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "d=1        n=%d M=%d all algorithms = M^n - M!/(M-n)!", n, M);
    report(ok, buf + detail);
}

// One suffix table serves every (n, M) with n <= nmax, and every (n, m).
void reuse_tables() {
    Mint::set_mod(kPrimes[0]);
    bool ok = true;
    const OneTableSuffix T = build_one_table_suffix_ntt(24, {3, false});
    for (int n : {5, 12, 24})
        for (int M : {n + 1, n + 7, 2 * n + 3})
            ok = ok && T.nbad(n, M) == one_table_forward(n, M, {3, false});
    const SeparateSuffix Ts = build_separate_suffix(14, 2);
    for (int n : {4, 9, 14})
        for (const std::vector<int>& m : {std::vector<int>{n, 1}, {n / 2 + 1, n / 2 + 2}, {n + 3, n + 5}})
            ok = ok && Ts.nbad(n, m) == separate_forward(n, m);
    report(ok, "suffix tables reused across (n, M) and (n, m)");
}

// Exact reconstruction of N_bad from residues modulo several primes.
void crt_exact() {
    const int n = 6, M = 7, d = 2;
    const BruteResult br = brute_force_count(n, M, one_table_choices(M, d, false));
    const std::vector<uint32_t> primes = ntt_primes(3, 10);
    std::vector<uint32_t> res;
    for (uint32_t P : primes) {
        Mint::set_mod(P);
        res.push_back(build_one_table_suffix_ntt(n, {d, false}).nbad(n, M).val());
    }
    const std::vector<uint32_t> x = garner(primes, res);
    const std::string s = mixed_radix_to_bignat(primes, x).str();
    const long double l2 = log2_mixed_radix(primes, x);
    const bool ok = s == std::to_string(br.nbad) && std::fabs(static_cast<double>(l2 - std::log2(static_cast<long double>(br.nbad)))) < 1e-12;
    report(ok, "CRT reconstruction: N_bad = " + s + " (brute force " + std::to_string(br.nbad) + ")");
}

struct OneCase { int n, M, d; };

}  // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    std::puts("== one table, distinct candidate sets: brute force vs. O(n^4), O(n^3), O(n^2 log n) ==");
    for (OneCase c : std::vector<OneCase>{{2, 4, 1}, {3, 4, 1}, {2, 5, 2}, {3, 5, 2}, {4, 5, 2}, {2, 5, 3},
                                          {3, 6, 2}, {4, 6, 2}, {5, 6, 2}, {3, 6, 3}, {4, 6, 3}, {5, 6, 3},
                                          {3, 6, 4}, {4, 7, 2}, {5, 7, 2}, {6, 7, 2}, {4, 7, 3}, {5, 7, 3},
                                          {6, 7, 3}, {5, 8, 3}, {6, 8, 2}, {4, 8, 4}, {5, 8, 4}})
        one_table_vs_brute(c.n, c.M, c.d, false);

    std::puts("== one table, draws with replacement: brute force vs. O(n^4), O(n^3), O(n^2 log n) ==");
    for (OneCase c : std::vector<OneCase>{{2, 3, 2}, {2, 4, 2}, {3, 4, 2}, {3, 4, 3}, {3, 5, 2}, {4, 5, 2},
                                          {3, 5, 3}, {4, 5, 3}, {4, 6, 2}, {5, 6, 2}, {4, 6, 3}, {5, 6, 3},
                                          {5, 7, 2}, {4, 7, 4}})
        one_table_vs_brute(c.n, c.M, c.d, true);

    std::puts("== separate tables: brute force vs. O(n^{2d+2}), O(n^{d+2}) ==");
    for (const auto& c : std::vector<std::pair<int, std::vector<int>>>{
             {2, {2, 2}}, {3, {2, 2}}, {2, {3, 3}}, {3, {3, 3}}, {4, {3, 3}}, {5, {3, 3}}, {3, {2, 3}},
             {4, {2, 3}}, {3, {1, 3}}, {4, {3, 4}}, {5, {3, 4}}, {6, {3, 4}}, {5, {4, 4}}, {7, {4, 4}},
             {2, {1, 1, 1}}, {3, {2, 2, 2}}, {4, {2, 2, 2}}, {5, {2, 2, 2}}, {4, {2, 3, 2}}, {5, {2, 3, 2}},
             {6, {2, 3, 2}}, {5, {3, 3, 3}}, {6, {3, 3, 3}}, {4, {2, 2, 2, 2}}, {6, {2, 2, 2, 2}},
             {7, {2, 2, 2, 2}}, {3, {5}}, {4, {6}}})
        separate_vs_brute(c.first, c.second);

    std::puts("== cross-checks on larger instances ==");
    one_table_cross(30, 40, 2, false);
    one_table_cross(30, 45, 3, false);
    one_table_cross(24, 60, 4, false);
    one_table_cross(31, 32, 2, false);
    one_table_cross(30, 50, 2, true);
    one_table_cross(25, 40, 3, true);
    one_table_cross(60, 100, 3, false);
    separate_cross(16, {10, 12});
    separate_cross(20, {12, 14});
    separate_cross(12, {5, 6, 7});
    separate_cross(14, {8, 8, 8});
    separate_cross(9, {3, 3, 3, 3});
    separate_cross(10, {2, 3, 4, 5});
    for (int n : {3, 7, 12}) birthday(n, 20);
    reuse_tables();
    crt_exact();

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%d passed, %d failed (%.1f s)\n", g_pass, g_fail, secs);
    return g_fail ? 1 : 0;
}
