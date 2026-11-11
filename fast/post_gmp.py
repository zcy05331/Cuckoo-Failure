#!/usr/bin/env python3
"""Exact N_bad by CRT from residue files, exact threshold tests, exact quotient (GMP).

usage: post_gmp.py D out_file... [--pbits 49] [--dump DIR] > table.tsv

--pbits: every prime in the files is above 2^pbits (49 for cuckoo_fast, 61 with -DFP_RB=64).
--dump DIR: also save each N_bad in decimal.

Every big-integer step (CRT merge, modular inverse, C(M,d)^n, division, radix
conversion) goes through GMP, which is subquadratic. Columns:
  log2_pfail      from 320-bit floats of the exact integers (shown to 15 decimals)
  le_2^-40/64/128 exact test  N_bad * 2^lambda <= Q^n
  p_fail          exact quotient N_bad / Q^n truncated to 40 significant digits
"""
import math
import sys
from collections import defaultdict

import gmpy2
from gmpy2 import mpfr, mpz

gmpy2.get_context().precision = 320
args = sys.argv[1:]
dump = None
pbits = 31  # every prime is above 2^pbits (31 for sweep, 49 or 61 for cuckoo_fast)
if "--pbits" in args:
    k = args.index("--pbits")
    pbits = int(args[k + 1])
    del args[k : k + 2]
if "--dump" in args:
    k = args.index("--dump")
    dump = args[k + 1]
    del args[k : k + 2]
d = int(args[0])

res = defaultdict(dict)  # (n, M) -> {prime index: (p, r)}
for fn in args[1:]:
    with open(fn) as f:
        for line in f:
            t = line.split()
            if len(t) == 5:
                res[(int(t[2]), int(t[3]))][int(t[0])] = (int(t[1]), int(t[4]))


def need(n, M):
    b = sum(math.log2(M - i) - math.log2(i + 1) for i in range(d))
    return math.ceil((n * b + 3) / float(pbits)) + 1


def crt(pairs):
    items = [(mpz(r), mpz(p)) for p, r in pairs]
    while len(items) > 1:
        nxt = []
        for k in range(0, len(items) - 1, 2):
            (x1, m1), (x2, m2) = items[k], items[k + 1]
            nxt.append((x1 + m1 * ((x2 - x1) * gmpy2.invert(m1, m2) % m2), m1 * m2))
        if len(items) % 2:
            nxt.append(items[-1])
        items = nxt
    return items[0]


def sci(N, D, digits=40):
    """N / D truncated to `digits` significant digits, by exact integer division."""
    if N == 0:
        return "0"
    e = gmpy2.num_digits(N, 10) - gmpy2.num_digits(D, 10)  # exponent estimate, off by at most 2
    while True:
        s = digits - 1 - e
        q = (N * mpz(10) ** s) // D if s >= 0 else N // (D * mpz(10) ** (-s))
        t = q.digits(10)
        if len(t) == digits:
            return t[0] + "." + t[1:] + "e%d" % e
        e += 1 if len(t) > digits else -1


print("n\tM\tratio\tlog2_pfail\tle_2^-40\tle_2^-64\tle_2^-128\tp_fail\tprimes")
for (n, M) in sorted(res):
    got = res[(n, M)]
    k = need(n, M)
    if any(i not in got for i in range(k)):
        print(f"{n}\t{M}\t{M / n:.6f}\tINCOMPLETE {len(got)}/{k}")
        continue
    x, m = crt([got[i] for i in range(k)])
    Qn = gmpy2.comb(M, d) ** n
    assert m > 4 * Qn and x <= Qn, (n, M)
    lg = gmpy2.log2(mpfr(x)) - gmpy2.log2(mpfr(Qn)) if x else mpfr("-inf")
    tests = "\t".join("yes" if (x << lam) <= Qn else "no" for lam in (40, 64, 128))
    print(f"{n}\t{M}\t{M / n:.6f}\t{lg:.15f}\t{tests}\t{sci(x, Qn)}\t{k}")
    sys.stdout.flush()
    if dump:
        with open(f"{dump}/N_{n}_{M}.txt", "w") as f:
            f.write("N_bad=" + x.digits(10) + f"\nQ^n=C({M},{d})^{n}\n")
