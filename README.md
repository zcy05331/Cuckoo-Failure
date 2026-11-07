# Cuckoo-Failure

Exact failure probabilities for `d`-choice cuckoo hashing, implemented in C++
after the dynamic-programming algorithms of *Precise Failure Probabilities for
d-Choice Cuckoo Hashing*.

`n` items are placed in `M` slots. Every item draws `d` candidate slots and must
be placed in one of them, with at most one item per slot. An outcome fails when
the candidate graph has no matching that covers every item. The code computes
`N_bad`, the number of failing outcomes among the `Q^n` equally likely ones, and
`p_fail = N_bad / Q^n`.

| model | choices per item `Q` | in the code |
| --- | --- | --- |
| one table, uniform `d`-subset of `[M]` (Section 3) | `C(M, d)` | `OneTableModel{d, false}` |
| one table, `d` independent draws with replacement (Section 3.4) | `M^d` | `OneTableModel{d, true}` |
| `d` separate tables of lengths `m_1, ..., m_d`, one uniform slot per table (Section 4) | `m_1 * ... * m_d` | `std::vector<int> m` |

## Algorithms

| paper | function (header) | time | what it does |
| --- | --- | --- | --- |
| Section 3.2, eqs. (13)-(15) | `one_table_forward` (`single_table.hpp`) | `O(n^4)` | forward recurrence over block boundaries `(l, b)`; only the boundaries that can carry weight, `(0, 0)` and `z <= l < b`, are visited |
| Section 3.3 | `build_one_table_suffix_cubic` (`single_table.hpp`) | `O(n^3)` | suffix table `V[p][v]` with shared inner sums (the intermediate step of Section 3.3) |
| Section 3.3 | `build_one_table_suffix_ntt` (`single_table.hpp`) | `O(n^2 log n)` | both stages are convolutions with the kernel `1/t`, evaluated by number-theoretic transforms |
| Section 4.2, eqs. (18)-(20) | `separate_forward` (`separate_tables.hpp`) | `O(n^{2d+2})` | forward recurrence over `(l, b)` with per-table forbidden counts; `O(n^{d+1})` retained states |
| Section 4.3 | `build_separate_suffix` (`separate_tables.hpp`) | `O(n^{d+2})` | suffix table over `(p, v)`; the shared inner sums form a `d`-dimensional convolution evaluated with `O(n)` multiplications by `sigma = x_1 + ... + x_d` |

A suffix table depends only on `d` and the largest `n`, not on the table sizes:
`OneTableSuffix::nbad(n, M)` costs `O(n^2)` and `SeparateSuffix::nbad(n, m)`
costs `O(n^{d+1})` per query, so one table serves many `(n, M)` or `(n, m)`.

All arithmetic is modular (`Mint`, a runtime prime below `2^32`). The
command-line tool runs the chosen algorithm modulo enough NTT-friendly primes
for their product to exceed `Q^n`, reconstructs `N_bad` exactly by Chinese
remaindering (`crt.hpp`), and prints `p_fail` in double precision, optionally
with the exact decimal values of `N_bad` and `Q^n`.

### The suffix tables

For a boundary with `l >= z` (after the block of degree 0 that holds the
`z = M - n` dummy items) put `p = M - l` (remaining items) and `v = M - b`
(available slots), and let `g(v)` be the number of candidate sets inside `v`
slots (`C(v, d)`, or `v^d` with replacement). The multiplier (11) and the
terminal factor of (15) become

```
tau = -C(v-w-1, v-q) C(p-q-1, p-v-1) C(v, w) C(p, q) g(v)^{p-q},   terminal = C(p-1, v) g(v)^p,
```

so the total weight `V[p][v]` of the blocks after the boundary satisfies

```
S[v][q] = sum_{w<q} C(v-w-1, v-q) C(v, w) V[q][w]
V[p][v] = C(p-1, v) g(v)^p - sum_{q<=v} C(p-q-1, p-v-1) C(p, q) g(v)^{p-q} S[v][q]
N_bad   = sum_{q,w} C(M-w-1, M-q) C(M, w) C(n, q) Q^{n-q} V[q][w]        (initial block of degree 0)
```

Rewriting the binomials turns both stages into convolutions with the kernel
`1/t`, which gives the `O(n^2 log n)` construction:

```
S[v][q] = v!/(v-q)! * sum_w [V[q][w] / (w! (q-w-1)!)] / (v-w)
V[p][v] = g(v)^p [ C(p-1, v) - p!/(p-v-1)! * sum_q [S[v][q] g(v)^{-q} / (q! (v-q)!)] / (p-q) ]
```

For separate tables `v` is the vector of available slots per table,
`g(v) = prod_t v_t`, and the inner sum runs over vectors `w <= v` with
`prod_t C(v_t, w_t)` in place of `C(v, w)`. It is a `d`-dimensional
convolution with the kernel `K[e] = (|e|-1)! / prod_t e_t!`, whose generating
function is `-log(1 - sigma)` with `sigma = x_1 + ... + x_d`; the product
`sum_{s>=1} sigma^s X / s` is evaluated by Horner's rule with `O(n)`
multiplications by `sigma`, each `O(d)` per coefficient, for `O(n^{d+2})` in
total. The derivations are repeated in the header comments next to the code.

## Building

Requires a C++17 compiler (tested with g++ 10, MinGW-w64/TDM-GCC on Windows).

```sh
make            # builds bin/check and bin/cuckoo_fail
make check      # builds and runs the correctness checks
```

or directly

```sh
g++ -O2 -std=c++17 -Iinclude -o check src/check.cpp
g++ -O2 -std=c++17 -Iinclude -o cuckoo_fail src/cuckoo_fail.cpp
```

The library itself is header-only (`include/cuckoo/`).

## Correctness checks

`src/check.cpp` compares every algorithm with a brute-force count
(`brute_force.hpp`: enumeration of all multisets of candidate choices with
multinomial weights, failure decided by bipartite matching) on 65 small
instances of the three models, modulo two different primes, and adds

- cross-checks between `O(n^4)`, `O(n^3)` and `O(n^2 log n)` up to `n = 60`,
  and between `O(n^{2d+2})` and `O(n^{d+2})` for `d = 2, 3, 4`;
- the closed form `N_bad = M^n - M!/(M-n)!` for `d = 1`, which all algorithms
  (both one-table models and the separate-table ones) must reproduce;
- reuse of one suffix table across several `(n, M)` and `(n, m)`;
- exact reconstruction of `N_bad` from residues modulo several primes.

```
$ ./bin/check
== one table, distinct candidate sets: brute force vs. O(n^4), O(n^3), O(n^2 log n) ==
[PASS] one table  n=2 M=4 d=1 distinct    brute N_bad=4 / 16
...
[PASS] one table  n=6 M=7 d=2 distinct    brute N_bad=23597091 / 85766121
...
== separate tables: brute force vs. O(n^{2d+2}), O(n^{d+2}) ==
...
[PASS] separate   n=6 m=(3,3,3) brute N_bad=722709 / 387420489
...
[PASS] CRT reconstruction: N_bad = 23597091 (brute force 23597091)

83 passed, 0 failed (3.5 s)
```

## Command-line tool

```
cuckoo_fail single   -n N -M M -d D [--replacement] [--algo forward|cubic|ntt] [--exact] [--mod P]
cuckoo_fail separate -n N -m m1,m2,...,md       [--algo forward|suffix]     [--exact] [--mod P]
```

`--algo` selects the algorithm (default: the fastest one), `--exact` also
prints `N_bad` and `Q^n` in decimal, and `--mod P` reports `N_bad mod P` from a
single run instead of reconstructing the integer.

```
$ ./bin/cuckoo_fail single -n 20 -M 40 -d 3 --exact
one table: n=20 items, M=40 slots, d=3 choices, uniform d-subset of candidate slots
algorithm: ntt;  outcomes Q^n = 2^265.406
primes: 10 (product 2^311.3)  time: 0.004 s
p_fail = 2.328850077437917e-07
log2(p_fail) = -22.033879,  log10(p_fail) = -6.632858
N_bad = 18292809974308866472375915138161693832233480017410161093140031066419200000
Q^n   = 78548680104103955730097329547580614987391500612545069816217600000000000000000000

$ ./bin/cuckoo_fail separate -n 40 -m 30,30,30
separate tables: n=40 items, m=(30,30,30), d=3 tables
algorithm: suffix;  outcomes Q^n = 2^588.827
primes: 20 (product 2^619.8)  time: 2.345 s
p_fail = 6.586258703572174e-09
log2(p_fail) = -27.177894,  log10(p_fail) = -8.181361
```

Timings on one core (every algorithm is run once per prime; the number of
primes grows like `n log2(Q) / 31`):

| instance | `forward` | `cubic` | `ntt` / `suffix` |
| --- | --- | --- | --- |
| one table `n=80, M=160, d=3` (51 primes) | 7.0 s | | 0.29 s |
| one table `n=200, M=400, d=3` (152 primes) | | 15.6 s | 5.0 s |
| separate `n=20, m=(15,15,15)` (9 primes) | 1.24 s | | 0.03 s |
| separate `n=40, m=(30,30,30)` (20 primes) | | | 2.3 s |

## Layout

```
include/cuckoo/modint.hpp           modular integers, factorials/binomials, NTT-friendly primes
include/cuckoo/ntt.hpp              number-theoretic transform and convolution
include/cuckoo/single_table.hpp     one table: O(n^4) forward recurrence, O(n^3) and O(n^2 log n) suffix tables
include/cuckoo/separate_tables.hpp  separate tables: O(n^{2d+2}) forward recurrence, O(n^{d+2}) suffix table
include/cuckoo/brute_force.hpp      brute-force counts for small instances
include/cuckoo/crt.hpp              Chinese remaindering and decimal output
src/check.cpp                       correctness checks
src/cuckoo_fail.cpp                 command-line tool
```

## Assumptions

- As in the paper, `d >= 1` and `1 <= n < M` (so there is at least one dummy
  item); for separate tables `M = m_1 + ... + m_d`.
- The brute force handles at most 62 slots and instances with `Q^n < 2^64`.
- Modular arithmetic needs `M` below the prime; the primes used are the largest
  NTT-friendly primes below `2^32`.
