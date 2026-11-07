# Cuckoo-Failure

C++ implementation of the algorithms of *Exact Failure Probabilities for
d-Choice Cuckoo Hashing*: dynamic programs for the exact probability
`p_fail = N_bad / Q^n` that `n` items with `d` candidate slots each admit no
valid placement into `M` slots. This is the implementation referenced by the
paper, including the accelerated `O(n^2 log n)` and `O(n^{d+2})` versions.

Models: one table with a uniform `d`-subset of candidate slots per item
(`Q = C(M, d)`, Section 3), `d` separate tables of lengths `m_1, ..., m_d`
with one uniform slot per table (`Q = m_1 ... m_d`, Section 4), and the
variant of Section 3.4 with `d` independent draws with replacement
(`Q = M^d`).

## Algorithms

| paper | function | time |
| --- | --- | --- |
| Sec. 3.2, forward recurrence `F[l, b]` | `one_table_forward` (`single_table.hpp`) | `O(n^4)` |
| Sec. 3.3, suffix table `V[p, v]` with shared inner sums | `build_one_table_suffix_cubic` | `O(n^3)` |
| Sec. 3.3, both stages as `1/t`-convolutions via NTT | `build_one_table_suffix_ntt` | `O(n^2 log n)` |
| Sec. 4.2, forward recurrence `F_d[l, b]` | `separate_forward` (`separate_tables.hpp`) | `O(n^{2d+2})` |
| Sec. 4.3, suffix table over `(p, v)` | `build_separate_suffix` | `O(n^{d+2})` |

The suffix tables depend only on `d` and the largest `n`; `nbad(n, M)` then
costs `O(n^2)` and `nbad(n, m)` costs `O(n^{d+1})` per query. In the
`O(n^{d+2})` algorithm the shared inner sums form a `d`-dimensional convolution
whose kernel has the generating function `-log(1 - sigma)`,
`sigma = x_1 + ... + x_d`, evaluated by Horner's rule with `O(n)`
multiplications by `sigma`. The formulas are derived in the header comments,
with the paper's equation numbers.

All arithmetic is modulo 32-bit primes. The command-line tool runs the
algorithm modulo enough NTT-friendly primes, reconstructs `N_bad` exactly by
Chinese remaindering and prints `p_fail`.

## Build and check

```sh
make          # bin/check and bin/cuckoo_fail (C++17 compiler)
make check    # 83 checks: brute force on 65 small instances modulo two primes,
              # cross-checks between the algorithms, the d = 1 closed form, CRT reconstruction
```

## Usage

```
cuckoo_fail single   -n N -M M -d D [--replacement] [--algo forward|cubic|ntt] [--exact] [--mod P]
cuckoo_fail separate -n N -m m1,...,md       [--algo forward|suffix]     [--exact] [--mod P]
```

```
$ ./bin/cuckoo_fail single -n 20 -M 40 -d 3
one table: n=20 items, M=40 slots, d=3 choices, uniform d-subset of candidate slots
algorithm: ntt;  outcomes Q^n = 2^265.406
primes: 10 (product 2^311.3)  time: 0.004 s
p_fail = 2.328850077437917e-07
log2(p_fail) = -22.033879,  log10(p_fail) = -6.632858
```

`--exact` also prints `N_bad` and `Q^n` in decimal; `--mod P` reports
`N_bad mod P` from a single run. One core: one table `n=200, M=400, d=3` takes
5 s with `ntt`; separate tables `n=40, m=(30,30,30)` take 2.3 s.

Assumptions: `d >= 1` and `1 <= n < M` as in the paper; the brute force
handles up to 62 slots and `Q^n < 2^64`.
