# fast/ — 64-bit batch engine (CUDA, host, AVX-512 IFMA)

`cuckoo_fast.cu` runs the one-table suffix-table algorithm (paper Section 3.3, uniform
d-subset model) with 64-bit residues. The same source builds three ways and all three print
identical residues for the same primes:

| build | command | notes |
|---|---|---|
| CUDA | `make fast-gpu` (nvcc) | every loop is a kernel; several primes per launch |
| host | `make fast` | portable scalar code, used as the reference |
| host + IFMA | `make fast-ifma` | hot loops in AVX-512 IFMA, 8 residues per instruction |

```
cuckoo_fast D QUERIES FIRST COUNT [--batch B] [--out FILE]
```

- `QUERIES` has one `n M` pair per line. Output lines are `i p n M residue`, where `i` is the
  index of the prime in a fixed list (largest first): primes below 2^50 with 2^20 | p - 1.
- A query is evaluated for prime `i` only if `i < need(n, M)`, the number of primes whose
  product exceeds Q^n. Reconstruct with `fast/post_gmp.py D out... --pbits 49` (needs gmpy2).
- `--out FILE` appends and skips primes already complete in the file, so a job cut off by a
  time limit, or a second GPU given the same range, continues where the first stopped. Disjoint
  `FIRST COUNT` ranges are independent: this is how several GPUs or nodes share one run.
- `-DFP_RB=64` switches to Montgomery R = 2^64 and 62-bit primes (about 20% fewer primes;
  scalar and CUDA only, use `--pbits 61`).

Memory is 8 n^2 bytes per prime (V and S as packed triangles): 2 GB at n = 16384, 8.6 GB at
n = 32768, 34 GB at n = 65536. `--batch` sets how many primes are processed together; the CUDA
build defaults to what fits in free device memory.

## What changed in the library (include/cuckoo)

- `Mint` stores residues in Montgomery form (R = 2^32): `Mint::set_mod(p)` replaces assigning
  `Mint::mod`, `val()` returns the residue, `Mint::one()` is 1.
- `Ntt::convolve` keeps the kernel spectrum in bit-reversed order with the 1/L factor folded
  in and caches the twiddle tables, so a convolution is one forward and one inverse pass.
- `build_one_table_suffix_ntt` uses transform length `next_pow2(nmax + 1)` (2 nmax for a power
  of two) instead of `next_pow2(2 nmax + 2)`: only entries untouched by wrap-around are read.
  The argument is in the comment above `one_table_ntt_length`.

## Checks that were run

- `make check`: 83 passed after the library changes; residues at n = 4096 are
  byte-identical to the previous version.
- host build vs the 32-bit pipeline: N_bad(512, 634) reconstructed from 50-bit primes is
  identical to the value from 31-bit primes.
- IFMA build vs host build: identical output on n = 20, 37, 512, 777, 1000, 4096.
