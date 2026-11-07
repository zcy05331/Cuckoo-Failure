// Command-line tool: exact failure probability of d-choice cuckoo hashing.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuckoo/crt.hpp"
#include "cuckoo/separate_tables.hpp"
#include "cuckoo/single_table.hpp"

using namespace cuckoo;

namespace {

void usage() {
    std::puts(
        "usage:\n"
        "  cuckoo_fail single   -n N -M M -d D [--replacement] [--algo forward|cubic|ntt] [--exact] [--mod P]\n"
        "  cuckoo_fail separate -n N -m m1,m2,...,md       [--algo forward|suffix]     [--exact] [--mod P]\n"
        "\n"
        "Computes N_bad, the number of candidate-set outcomes that admit no placement,\n"
        "and p_fail = N_bad / Q^n. The algorithms work modulo 32-bit primes; enough\n"
        "primes are used to reconstruct N_bad exactly by Chinese remaindering.\n"
        "  --algo     single: forward O(n^4) | cubic O(n^3) | ntt O(n^2 log n) (default)\n"
        "             separate: forward O(n^{2d+2}) | suffix O(n^{d+2}) (default)\n"
        "  --exact    also print N_bad and Q^n in decimal\n"
        "  --mod P    report N_bad mod P only (single run, no reconstruction)\n");
}

struct Options {
    std::string mode, algo;
    int n = 0, M = 0, d = 0;
    std::vector<int> m;
    bool replacement = false, exact = false;
    long long mod = 0;
};

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        out.push_back(std::atoi(s.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("missing mode");
    Options o;
    o.mode = argv[1];
    if (o.mode != "single" && o.mode != "separate") throw std::invalid_argument("mode must be single or separate");
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "-n") o.n = std::atoi(next().c_str());
        else if (a == "-M") o.M = std::atoi(next().c_str());
        else if (a == "-d") o.d = std::atoi(next().c_str());
        else if (a == "-m") o.m = parse_list(next());
        else if (a == "--algo") o.algo = next();
        else if (a == "--mod") o.mod = std::atoll(next().c_str());
        else if (a == "--replacement") o.replacement = true;
        else if (a == "--exact") o.exact = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    if (o.mode == "single") {
        if (o.algo.empty()) o.algo = "ntt";
        if (o.algo != "forward" && o.algo != "cubic" && o.algo != "ntt") throw std::invalid_argument("bad --algo");
        check_one_table_args(o.n, o.M, {o.d, o.replacement});
    } else {
        if (o.algo.empty()) o.algo = "suffix";
        if (o.algo != "forward" && o.algo != "suffix") throw std::invalid_argument("bad --algo");
        check_separate_args(o.n, o.m);
    }
    if (o.mod && (o.mod < 3 || o.mod >= (1ll << 32) || !is_prime_u32(static_cast<uint32_t>(o.mod))))
        throw std::invalid_argument("--mod must be a prime below 2^32");
    return o;
}

Mint run(const Options& o) {
    if (o.mode == "single") {
        const OneTableModel model{o.d, o.replacement};
        if (o.algo == "forward") return one_table_forward(o.n, o.M, model);
        if (o.algo == "cubic") return build_one_table_suffix_cubic(o.n, model).nbad(o.n, o.M);
        return build_one_table_suffix_ntt(o.n, model).nbad(o.n, o.M);
    }
    if (o.algo == "forward") return separate_forward(o.n, o.m);
    return build_separate_suffix(o.n, static_cast<int>(o.m.size())).nbad(o.n, o.m);
}

// log2 of the number Q of candidate choices per item.
long double log2_Q(const Options& o) {
    long double r = 0;
    if (o.mode == "separate") {
        for (int x : o.m) r += std::log2(static_cast<long double>(x));
    } else if (o.replacement) {
        r = o.d * std::log2(static_cast<long double>(o.M));
    } else {
        for (int i = 0; i < o.d; ++i) r += std::log2(static_cast<long double>(o.M - i)) - std::log2(static_cast<long double>(i + 1));
    }
    return r;
}

// Q^n in decimal.
BigNat exact_Qn(const Options& o) {
    BigNat b;
    b.limbs = {1};
    if (o.mode == "separate") {
        for (int x : o.m)
            for (int i = 0; i < o.n; ++i) b.mul_add(x, 0);
    } else if (o.replacement) {
        for (int i = 0; i < o.d * o.n; ++i) b.mul_add(o.M, 0);
    } else {  // C(M, d)^n = prod_i (M - i)^n / (d!)^n, every intermediate quotient is an integer
        for (int i = 0; i < o.d; ++i)
            for (int t = 0; t < o.n; ++t) b.mul_add(o.M - i, 0);
        for (int j = 2; j <= o.d; ++j)
            for (int t = 0; t < o.n; ++t) b.div_small(j);
    }
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n", e.what());
        usage();
        return 2;
    }

    if (o.mode == "single")
        std::printf("one table: n=%d items, M=%d slots, d=%d choices, %s\n", o.n, o.M, o.d,
                    o.replacement ? "d independent draws with replacement" : "uniform d-subset of candidate slots");
    else {
        std::printf("separate tables: n=%d items, m=(", o.n);
        for (size_t j = 0; j < o.m.size(); ++j) std::printf("%s%d", j ? "," : "", o.m[j]);
        std::printf("), d=%zu tables\n", o.m.size());
    }
    const long double bits = o.n * log2_Q(o);  // log2 of Q^n
    std::printf("algorithm: %s;  outcomes Q^n = 2^%.3f\n", o.algo.c_str(), static_cast<double>(bits));

    std::vector<uint32_t> primes;
    long double have = 0;
    if (o.mod) {
        primes.push_back(static_cast<uint32_t>(o.mod));
    } else {
        const size_t L = Ntt::next_pow2(2 * static_cast<size_t>(o.n) + 2);
        int min_pow2 = 0;
        while ((static_cast<size_t>(1) << min_pow2) < L) ++min_pow2;
        int count = static_cast<int>(bits / 31) + 2;
        while (true) {
            primes = ntt_primes(count, min_pow2);
            have = 0;
            for (uint32_t p : primes) have += std::log2(static_cast<long double>(p));
            if (have > bits + 2) break;  // the product of the primes exceeds Q^n >= N_bad
            count += static_cast<int>((bits + 2 - have) / 31) + 1;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> residues;
    for (uint32_t P : primes) {
        Mint::mod = P;
        residues.push_back(run(o).v);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (o.mod) {
        std::printf("N_bad mod %u = %u  (%.3f s)\n", primes[0], residues[0], secs);
        return 0;
    }
    std::printf("primes: %zu (product 2^%.1f)  time: %.3f s\n", primes.size(), static_cast<double>(have), secs);
    const std::vector<uint32_t> x = garner(primes, residues);
    const long double log2N = log2_mixed_radix(primes, x);
    if (std::isinf(static_cast<double>(log2N))) {
        std::puts("N_bad = 0,  p_fail = 0");
        return 0;
    }
    const long double log2p = log2N - bits;
    std::printf("p_fail = %.15e\n", static_cast<double>(std::exp2(log2p)));
    std::printf("log2(p_fail) = %.6f,  log10(p_fail) = %.6f\n", static_cast<double>(log2p),
                static_cast<double>(log2p * std::log10(2.0L)));
    if (o.exact) {
        std::printf("N_bad = %s\n", mixed_radix_to_bignat(primes, x).str().c_str());
        std::printf("Q^n   = %s\n", exact_Qn(o).str().c_str());
    }
    return 0;
}
