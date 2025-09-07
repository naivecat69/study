# -*- coding: utf-8 -*-
# (7,4) Hamming code via generator polynomial over GF(2)
# - Poly representation: nonnegative int as bitmask; LSB = x^0
# - Example: 0b1011 -> x^3 + x + 1

from typing import Tuple, List

# ---------- GF(2) polynomial utilities (bitmask representation) ----------
def deg(p: int) -> int:
    """degree of polynomial p (bit-length-1), deg(0)=-1"""
    return p.bit_length() - 1

def poly_str(p: int, var: str = "x") -> str:
    """Convert bitmask poly to human-readable string (e.g., x^3 + x + 1)."""
    if p == 0:
        return "0"
    terms = []
    i = 0
    q = p
    while q:
        if q & 1:
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append(var)
            else:
                terms.append(f"{var}^{i}")
        q >>= 1
        i += 1
    return " + ".join(reversed(terms))

def poly_add(a: int, b: int) -> int:
    """Addition over GF(2) => XOR"""
    return a ^ b

def poly_mul(a: int, b: int) -> int:
    """Multiplication over GF(2) in bitmask form."""
    res = 0
    x = a
    y = b
    while y:
        if y & 1:
            res ^= x
        y >>= 1
        x <<= 1
    return res

def poly_divmod(a: int, b: int) -> Tuple[int, int]:
    """Division a/b over GF(2): returns (q, r). Requires b != 0."""
    if b == 0:
        raise ZeroDivisionError("poly_divmod by zero")
    q = 0
    r = a
    db = deg(b)
    while r and deg(r) >= db:
        shift = deg(r) - db
        q ^= (1 << shift)
        r ^= (b << shift)
    return q, r

def poly_mod(a: int, m: int) -> int:
    """Remainder a mod m over GF(2)."""
    return poly_divmod(a, m)[1]

def poly_gcd(a: int, b: int) -> int:
    """GCD over GF(2)."""
    x, y = a, b
    while y:
        _, r = poly_divmod(x, y)
        x, y = y, r
    return x

def poly_pow_mod(a: int, e: int, m: int) -> int:
    """(a^e) mod m over GF(2)."""
    res = 1
    base = a
    exp = e
    while exp > 0:
        if exp & 1:
            res = poly_mod(poly_mul(res, base), m)
        base = poly_mod(poly_mul(base, base), m)
        exp >>= 1
    return res

# ---------- Cyclic (7,4) Hamming specifics ----------
def xn_minus_1(n: int) -> int:
    """x^n - 1 over GF(2) in bitmask: that's (1 << n) - 1 (since -1 => +1 in GF(2))."""
    # x^n + 1 in GF(2) equals x^n - 1 in usual notation, because -1 == +1.
    # But as a polynomial, x^n - 1 = x^n + 1 over GF(2).
    # Bitmask with bits n and 0 set: (1<<n) | 1. However, for division we often
    # use the modulus x^n - 1 by reducing via cyclic wrap; here we’ll compute h(x) with division.
    # For exact (x^n - 1), use x^n + 1 form: bit n and bit 0 set.
    return (1 << n) | 1  # x^n + 1  (== x^n - 1 in GF(2))

def cyclic_reduce(p: int, n: int) -> int:
    """Reduce polynomial modulo x^n - 1 (i.e., wrap-around / cyclic)."""
    # Equivalent to repeatedly folding back terms with degree >= n
    mask_n = (1 << n) - 1
    while deg(p) >= n:
        shift = deg(p) - n
        # x^{n} == 1 => subtract (== add) x^{shift} in GF(2) at degree shift
        p = (p & mask_n) ^ (1 << shift)
    return p & mask_n

def message_to_poly(m_bits: List[int]) -> int:
    """m_bits[0] is the LSB (= coefficient of x^0). Length = k."""
    p = 0
    for i, b in enumerate(m_bits):
        if b & 1:
            p |= (1 << i)
    return p

def poly_to_bits(p: int, n: int) -> List[int]:
    """Return n coefficients [c0, c1, ..., c_{n-1}] for p mod x^n-1."""
    return [(p >> i) & 1 for i in range(n)]

def encode_cyclic_hamming_7_4(m_bits: List[int], g: int) -> List[int]:
    """
    Encode 4-bit message with generator polynomial g(x) (deg 3) into 7-bit cyclic code:
    c(x) = (m(x) * g(x)) mod (x^7 - 1)
    """
    n = 7
    if len(m_bits) != 4:
        raise ValueError("message length must be 4 for (7,4) Hamming")
    m = message_to_poly(m_bits)             # 4-bit message -> poly
    prod = poly_mul(m, g)
    mod = xn_minus_1(n)                     # x^7 + 1 == x^7 - 1 in GF(2)
    # True cyclic reduction uses mod (x^n - 1). We can do long division by (x^n + 1)
    # but an easier correct approach is manual wrap-around:
    # However, dividing by (x^n + 1) as polynomial modulus is acceptable since +1 == -1 in GF(2).
    q, r = poly_divmod(prod, mod)
    c = r
    # Ensure we output exactly 7 bits:
    return poly_to_bits(c, n)

def parity_check_from_h(h: int, n: int) -> List[List[int]]:
    """Build circulant parity-check matrix H (size (n-k) x n) from h(x)."""
    r = deg(h) + 1          # number of columns with ones in first row, but we build full circulant rows
    # For cyclic codes, a standard parity-check is the circulant with first row the coefficients of h(x).
    base = poly_to_bits(h, n)  # pad to length n
    H = []
    row = base[:]
    rows = n - (n - r)  # but for Hamming (7,4), we expect 3 rows. We'll build 3 independent cyclic shifts.
    # Simpler: build 3 rows as cyclic shifts of h's length-n vector such that rank=3.
    # We'll just take 3 successive cyclic shifts.
    for _ in range(n - 4):   # n - k = 3 rows
        H.append(row[:])
        # cyclic right shift by 1 (you can also choose left; both are valid parity-checks up to permutation)
        row = [row[-1]] + row[:-1]
    return H

def syndrome_table_single_error(H: List[List[int]]) -> dict:
    """Map 3-bit syndrome -> error position (0..6)."""
    import numpy as np
    Hm = np.array(H, dtype=int)  # shape (3,7)
    table = {}
    for pos in range(Hm.shape[1]):
        e = np.zeros(Hm.shape[1], dtype=int)
        e[pos] = 1
        s = (Hm @ e) % 2
        key = tuple(s.tolist())
        table[key] = pos
    table[tuple([0]*len(H))] = None  # no error
    return table

def decode_syndrome(c_bits: List[int], H: List[List[int]], synd_tbl: dict) -> Tuple[List[int], int]:
    """Return (corrected 7-bit codeword, error_position_or_None)."""
    import numpy as np
    v = np.array(c_bits, dtype=int)
    Hm = np.array(H, dtype=int)
    s = (Hm @ v) % 2
    key = tuple(s.tolist())
    pos = synd_tbl.get(key, None)
    if pos is not None:
        v[pos] ^= 1
    return v.tolist(), pos

# ---------- Equivalence check between two generator polys ----------
def cyclic_codewords(n: int, g: int) -> List[int]:
    """Enumerate all 2^k codewords as ints (k = n - deg(g))."""
    k = n - deg(g) - 1  # deg(g)=3 => k should be 7-3=4; but deg(g) returns 3, so k = n - 3 - 1 = 3 ? -> fix
    # Correction:
    k = n - (deg(g) + 1)
    words = []
    for m in range(1 << k):
        # encode via poly multiplication mod (x^n - 1)
        prod = poly_mul(m, g)
        # reduce mod x^n - 1 by folding: use polynomial division by x^n + 1 works in GF(2)
        mod = (1 << n) | 1
        q, r = poly_divmod(prod, mod)
        words.append(r & ((1 << n) - 1))
    return words

def code_equivalent_by_permutation(n: int, g1: int, g2: int) -> bool:
    """
    Very small n=7 check: are the codeword sets the same up to column permutation?
    Brute-force all 7! is heavy; instead we check simple dihedral permutations
    (cyclic shifts and reversals) which cover equivalence between x^3+x+1 and x^3+x^2+1 for n=7.
    """
    import numpy as np
    C1 = set(cyclic_codewords(n, g1))
    C2 = set(cyclic_codewords(n, g2))

    def permute_bits(word: int, perm: List[int]) -> int:
        out = 0
        for new_pos, old_pos in enumerate(perm):
            if (word >> old_pos) & 1:
                out |= (1 << new_pos)
        return out

    # generate dihedral group D_n permutations (n rotations * 2 reflections)
    perms = []
    base = list(range(n))
    for r in range(n):
        rot = [(i + r) % n for i in base]
        perms.append(rot)
        rev = list(reversed(rot))
        perms.append(rev)

    # Try to match C1 under a perm to C2
    for perm in perms:
        mapped = {permute_bits(w, perm) for w in C1}
        if mapped == C2:
            return True
    return False

# ---------- Demo / Self-test ----------
if __name__ == "__main__":
    n = 7
    g = 0b1011            # x^3 + x + 1 (LSB first)
    print("[g(x)] =", poly_str(g))  # should be x^3 + x + 1

    # h(x) = (x^n - 1) / g(x)
    Xn_1 = (1 << n) | 1   # x^7 + 1
    h, rem = poly_divmod(Xn_1, g)
    assert rem == 0
    print("[h(x)] =", poly_str(h))  # should be x^4 + x^2 + x + 1

    # Encode a sample message (m3 m2 m1 m0) with LSB-first list: [m0, m1, m2, m3]
    m_bits = [1, 0, 1, 1]  # m(x) = 1 + x^2 + x^3 (= 0b1101)
    c_bits = encode_cyclic_hamming_7_4(m_bits, g)
    print("m bits (LSB->):", m_bits)
    print("c bits (len=7, LSB->):", c_bits)

    # Build a simple parity-check H from h(x) as a circulant (3 rows)
    H = parity_check_from_h(h, n)
    print("H (3x7) rows:")
    for row in H:
        print(row)

    # Build syndrome table and correct a single-bit error
    synd_tbl = syndrome_table_single_error(H)
    # introduce error at position 5 (0-based, LSB side is position 0)
    v = c_bits[:]
    v[5] ^= 1
    v_corr, pos = decode_syndrome(v, H, synd_tbl)
    print("received v with 1-bit error at pos 5:", v)
    print("syndrome-corrected ->", v_corr, " (fixed pos:", pos, ")")
    assert v_corr == c_bits

    # Equivalence check: g1 = x^3 + x + 1  vs g2 = x^3 + x^2 + 1
    g1 = 0b1011        # x^3 + x + 1
    g2 = 0b1101        # x^3 + x^2 + 1
    eq = code_equivalent_by_permutation(n, g1, g2)
    print("Equivalence (g1 vs g2) under rotation/reflection:", eq)