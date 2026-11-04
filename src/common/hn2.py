"""House-number parser used by the fusion features.

  - leading zeros stripped everywhere (numbers compared as numbers)
  - numbers that are NOT house numbers are excluded from the candidates: ordinals (22nd, 5th), numbers after
    unit/apt/suite/ste/floor/fl/room/rm/po box/box/sector/phase/lane no/road no/ward/gate no, and 5-6 digit postal codes
  - house number = first remaining candidate; comparisons are also done set-wise so order does not matter
Country-agnostic; French 'bis/ter', 'appartement', 'bp' (boite postale), 'etage' handled the same way.
"""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # package root (code/business_entity_resolution)

import re

_ORD = re.compile(r"\b\d+\s*(?:st|nd|rd|th|er|e|eme)\b", re.I)
_NONHN = re.compile(r"\b(?:unit|apt|apartment|appartement|suite|ste|floor|flr|fl|etage|room|rm|p\.?\s*o\.?\s*box|box|bp|sector|sec|phase|ward|"
                    r"gate\s*no|lane\s*no|road\s*no|rd\s*no|street\s*no|cross|main|stage|block\s*no|km|mile|loop|route|hwy|highway|us|sr|cr|i)"
                    r"\s*[#:.\-]?\s*(?:no\.?\s*)?[a-z]?\d+[a-z]?", re.I)
_POSTAL = re.compile(r"\b\d{5,6}\b\s*$")
_DIG = re.compile(r"\d+")


def z(x):
    return x.lstrip("0") or "0"


def parse_hn(addr):
    """-> (hn, candidates, all_numbers): hn = primary house number ('' if none); candidates = house-number-like numbers;
    all_numbers = every number in the address (zero-stripped)."""
    a = (addr or "").replace("<NULL>", " ").lower()
    alln = [z(x) for x in _DIG.findall(a)]
    b = _ORD.sub(" ", a)
    b = _NONHN.sub(" ", b)
    b = _POSTAL.sub(" ", b.strip())
    cand = [z(x) for x in _DIG.findall(b)]
    return (cand[0] if cand else ""), cand, alln


def compare(a1, a2):
    """Pair features from two addresses (S1 first). Returns a dict (all floats, -1 = not applicable)."""
    h1, c1, n1 = parse_hn(a1); h2, c2, n2 = parse_hn(a2)
    S1c, S2c, N1, N2 = set(c1), set(c2), set(n1), set(n2)
    out = {"hn2_state": 0.0, "hn2_eq": -1.0, "hn2_in_other": -1.0, "hn2_mindiff": -1.0, "hn2_trunc": -1.0,
           "nums2_shared": float(len(N1 & N2)), "nums2_only_s1": float(len(N1 - N2)), "nums2_only_aux": float(len(N2 - N1))}
    if not h1 or not h2:
        out["hn2_state"] = 1 + (not h1) + 2 * (not h2)                   # 2: S1 missing, 3: aux missing, 4: both missing
        if h1 and N2: out["hn2_in_other"] = float(h1 in N2)
        if h2 and N1: out["hn2_in_other"] = float(h2 in N1)
        return out
    out["hn2_eq"] = float(h1 == h2)
    out["hn2_in_other"] = float(h1 in S2c or h2 in S1c or h1 in N2)      # same number present, whatever the order
    d = min(abs(int(h1[:12]) - int(y[:12])) for y in (c2 or [h2]))
    out["hn2_mindiff"] = float(min(d, 10**6))
    out["hn2_trunc"] = float(h1 != h2 and (h1.endswith(h2) or h2.endswith(h1) or h1.startswith(h2) or h2.startswith(h1)))
    return out
