# fast/ — 64-bit batch engine (CUDA, host, AVX-512 IFMA)

`cuckoo_fast.cu` runs the one-table suffix-table algorithm (paper Section 3.3, uniform
d-subset model) with 64-bit residues. The same source builds three ways and all three print
identical residues for the same primes:

| build | command | notes |
|---|---|---|
| CUDA | `make fast-gpu` (nvcc) | convolutions in three shared-memory kernels; several primes per launch |
| host | `make fast` | portable scalar code, used as the reference |
| host + IFMA | `make fast-ifma` | hot loops in AVX-512 IFMA, 8 residues per instruction |

```
cuckoo_fast D QUERIES FIRST COUNT [--batch B] [--out FILE]
```

- `QUERIES` has one `n M` pair per line. Output lines are `i p n M residue`, where `i` is the
  index of the prime in a fixed list (largest first): primes below 2^50 with 2^20 | p - 1.
- A query is evaluated for prime `i` only if `i < need(n, M)`, the number of primes whose
  product exceeds Q^n. Reconstruct with `fast/post_gmp.py D out... --pbits 49` (needs gmpy2).
- One table per prime answers every query with n up to the largest n, at any M: list all the
  (n, M) you want before a run, they cost little next to the table.
- `--out FILE` appends and skips primes already complete in the file, so a job cut off by a
  time limit, or a second GPU given the same range, continues where the first stopped. Disjoint
  `FIRST COUNT` ranges are independent: this is how several GPUs or nodes share one run.
- `-DFP_RB=64` switches to Montgomery R = 2^64 and 62-bit primes (about 20% fewer primes;
  scalar and CUDA only, use `--pbits 61`).

Memory is 8 n^2 bytes per prime (V and S as packed triangles): 2 GB at n = 16384, 8.6 GB at
n = 32768, 34 GB at n = 65536. `--batch` sets how many primes are processed together; the CUDA
build defaults to what fits in free device memory.

## How the work is organized

- **Transform length** `next_pow2(nmax)`: the kernel 1/t is only needed for t <= nmax - 1, and
  with L >= nmax no entry that is read picks up a wrapped term (c[nmax] lands at index 0 when
  L = nmax). The argument is in the comment on `transform_length`. When nmax is a power of two
  this halves the transforms compared with `next_pow2(nmax + 1)`.
- **Tiles of columns**: step v writes column v of S and of V, which are stored by rows. The
  columns of eight consecutive steps are kept column-major in `SS` / `VS` and moved into the
  rows by `FlushTile`, so the writes into the 34 GB tables fill whole cache lines.
- **Convolution** (`convolve`): IFMA runs the top transform levels two per pass over the whole
  buffer, and for each block of 4096 entries the remaining forward levels, the pointwise product
  and the first inverse levels while the block is in L1. CUDA does the same with three kernels:
  top forward levels on tiles of 2048 entries in shared memory, a fused bottom-forward /
  product / bottom-inverse kernel on blocks of 2048, and top inverse levels.
- **Queries** (`QueryAll`, `k_query`): for every query out[q] = sum_w U[w] / (M - w) with
  U[w] = V[q][w] / (w! (q-w-1)!), which does not depend on (n, M). Each row of V is read once
  for all queries instead of once per query.

Every change only reorders independent butterflies or regroups products of residues, so the
output is byte-identical to the previous version; this was checked as listed below.

## What changed in the library (include/cuckoo)

- `Mint` stores residues in Montgomery form (R = 2^32): `Mint::set_mod(p)` replaces assigning
  `Mint::mod`, `val()` returns the residue, `Mint::one()` is 1.
- `Ntt::convolve` keeps the kernel spectrum in bit-reversed order with the 1/L factor folded
  in and caches the twiddle tables, so a convolution is one forward and one inverse pass.
- `build_one_table_suffix_ntt` uses transform length `next_pow2(nmax)` (see above) instead of
  `next_pow2(2 nmax + 2)`. The argument is in the comment above `one_table_ntt_length`.

## Checks that were run

All against outputs of the previous version, byte for byte:

| build | n <= 512 (40 primes) | n = 1000 / 777 / 37 | n = 4096 (16 primes) | n <= 16384 (2 primes, 18 queries) | n = 65536, 2 primes x 6 / x 126 queries |
|---|---|---|---|---|---|
| host | yes | yes | yes | yes | — |
| IFMA (Xeon 8358) | yes | — | yes | — | yes / yes |
| CUDA (H100; also batches 1, 8, 40 on n <= 512) | yes | — | yes | — | yes / yes |

The IFMA build was tested before the CUDA-only kernels (`k_fwd_top`, `k_mid`, `k_inv_top`,
`k_query`) were added; those are inside `#ifdef __CUDACC__` and do not change the host code path.

- `make check`: 83 passed; `--algo ntt` equals `--algo cubic` at n = 32, 64, 128, 200.
- host build vs the 32-bit pipeline: N_bad(512, 634) reconstructed from 50-bit primes is
  identical to the value from 31-bit primes.

## Timings at n = 65536 (one prime, 126 queries)

| build | table | queries |
|---|---|---|
| CUDA, whole H100, batch 2, previous version | 6.5 s per prime | 2.2 s per prime |
| CUDA, whole H100, batch 2, this version | 3.5 s per prime | 0.75 s per prime |
| IFMA, Xeon 8358 core, this version | 104 s | 89 s |
