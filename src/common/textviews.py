"""Text views used by the pipeline: normalised, transliterated (Indic -> Latin), skeleton names and address views, plus GPU top-K helpers."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # package root (code/business_entity_resolution)

import re
import unicodedata

import numpy as np
import scipy.sparse as sp
import torch
from anyascii import anyascii
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 2 ** 23

from unidecode import unidecode

_WS = re.compile(r"\s+")
_NULL = re.compile(r"<null>")
# punctuation/symbols/control chars (Unicode categories P*, S*, C*) -> space; keeps Indic vowel signs intact
_PUNCT_TABLE = {cp: " " for cp in range(0x10000) if unicodedata.category(chr(cp))[0] in "PSC" and chr(cp) not in "\t\n "}


def _clean(s):
    # lowercase, drop literal <NULL>, punctuation/symbols -> space, collapse whitespace
    s = _NULL.sub(" ", s.lower()).translate(_PUNCT_TABLE)
    return _WS.sub(" ", s).strip()


def v_raw(s):
    return s.lower()


def v_norm(s):
    # accent-fold Latin characters only (Indic scripts untouched), then clean
    return _clean("".join(unidecode(ch) if 0x80 <= ord(ch) < 0x250 else ch for ch in s))


def v_translit(s):
    # every script -> ASCII (Indic transliterated, accents folded), then clean
    return _clean(anyascii(s))


# phonetic skeleton: aligns English spellings with transliterated Indic spellings
_SK = [(re.compile(p), r) for p, r in [
    (r"[^a-z0-9 ]", " "),
    (r"ph", "f"), (r"sh", "s"), (r"th", "t"), (r"dh", "d"), (r"bh", "b"), (r"kh", "k"), (r"gh", "g"), (r"ck", "k"), (r"ch", "c"),
    (r"c(?=[eiy])", "s"), (r"c", "k"), (r"q", "k"), (r"x", "ks"), (r"z", "j"), (r"w", "v"),
    (r"g", "k"), (r"d", "t"), (r"b", "p"),                      # voicing merge (Tamil script has no voiced/unvoiced split)
    (r"(?<=[a-z])m(?=[bcdfghjklmnpqrstvwxyz])", "n"),                      # anusvara nasal
    (r"(.)\1+", r"\1"),                                          # doubled letters
    (r"\B[aeiouy]", "")]]                                         # non-initial vowels


def v_skel(s):
    s = _NULL.sub(" ", anyascii(s).lower())
    for p, r in _SK:
        s = p.sub(r, s)
    return _WS.sub(" ", s).strip()


VIEWS = {"raw": v_raw, "norm": v_norm, "translit": v_translit, "skel": v_skel}


# ---------------- address views: abbreviation canonicalization + number awareness -----------------
# One table for every record, whatever its country (no country branching). Everything is mapped to the SHORT form,
# so an ambiguous abbreviation (st = street / saint) never breaks a match: both sides receive the same mapping.
_ABBR = [
    # street types (US / India / France)
    (r"street|str", "st"), (r"avenue|ave|av", "av"), (r"road|rd", "rd"), (r"drive|dr", "dr"), (r"boulevard|blvd|bd", "bd"),
    (r"lane|ln", "ln"), (r"court|ct", "ct"), (r"circle|cir", "cir"), (r"place|pl", "pl"), (r"highway|hwy", "hwy"),
    (r"parkway|pkwy", "pkwy"), (r"rue|r", "r"), (r"allee|all", "all"), (r"impasse|imp", "imp"), (r"route|rte", "rte"),
    (r"chemin|chem|che", "che"), (r"square|sq", "sq"), (r"saint|st", "st"), (r"sainte|ste", "ste"),
    # India locality words
    (r"nagar|ngr", "ngr"), (r"colony|clny", "clny"), (r"industrial|indl", "indl"), (r"near|nr", "nr"), (r"opposite|opp", "opp"),
    (r"sector|sec", "sec"), (r"phase|ph", "ph"), (r"extension|extn|ext", "ext"), (r"building|bldg", "bldg"),
    (r"apartments?|apts?", "apt"), (r"society|soc", "soc"), (r"district|dist", "dist"), (r"floor|flr|fl", "fl"),
    # house-number markers carry no information once the number itself is kept
    (r"house\s*no|h\.?\s*no|hno|plot\s*no|flat\s*no|no|number|n", ""),
]
_ABBR_RX = [(re.compile(rf"(?<![\w]){p}(?![\w])\.?"), r) for p, r in [(f"(?:{a})", b) for a, b in _ABBR]]
_NUMSIGN = re.compile(r"n[°º]\s*|#")
_DIGITS = re.compile(r"\d+")


def _fold_latin(s):
    return "".join(unidecode(ch) if 0x80 <= ord(ch) < 0x250 else ch for ch in s)


def v_addr_abbr(s):
    s = _NUMSIGN.sub(" ", _NULL.sub(" ", s.lower()))
    s = _fold_latin(s)
    for rx, rep in _ABBR_RX:
        s = rx.sub(rep, s)
    return _WS.sub(" ", s).strip()


def _num_tags(s):
    return " ".join(f"#{n}#" for n in _DIGITS.findall(s))


def v_addr_numboost(s):
    s = s.lower()
    tags = _num_tags(s)
    return f"{s} {tags} {tags}".strip() if tags else s


def v_addr_abbr_numboost(s):
    a = v_addr_abbr(s)
    tags = _num_tags(a)
    return f"{a} {tags} {tags}".strip() if tags else a


def v_addr_numkey(s):
    """First numbered address part plus the part after it, keyed by the house number:
    '#25# #25# 25 r des tanneurs lille'. Empty when the address has no number."""
    parts = [p.strip() for p in v_addr_abbr(s).split(",")]
    for j, part in enumerate(parts):
        m = _DIGITS.search(part)
        if m:
            nxt = parts[j + 1] if j + 1 < len(parts) else ""
            return f"#{m.group()}# #{m.group()}# {part} {nxt}".strip()
    return ""


VIEWS.update({"addr_abbr": v_addr_abbr, "addr_numboost": v_addr_numboost, "addr_abbr_numboost": v_addr_abbr_numboost,
              "addr_numkey": v_addr_numkey})


def _count_chunk(texts, view, ngram):
    hv = HashingVectorizer(analyzer="char", ngram_range=ngram, lowercase=False, alternate_sign=False,
                           norm=None, n_features=N_FEATURES, dtype=np.float32)
    return hv.transform([VIEWS[view](t) for t in texts])


def counts(texts, view, ngram, n_jobs=32):
    step = max(1, len(texts) // n_jobs + 1)
    parts = Parallel(n_jobs=n_jobs)(delayed(_count_chunk)(texts[i:i + step], view, ngram)
                                    for i in range(0, len(texts), step))
    return sp.vstack(parts).tocsr()


def _tfidf(X, idf):
    X = X.copy()
    X.data = 1.0 + np.log(X.data)
    X = X @ sp.diags(idf.astype(np.float32))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms.astype(np.float32)) @ X


def _topk_dim0(S, k):
    """Top-k along dim 0 of a (n_docs, block) score matrix. torch.topk is very slow on this layout; for small k, k passes of
    max + masking are ~30x faster, for larger k a transposed contiguous topk is ~3x faster. Results identical."""
    if k <= 5:
        vs, ids = [], []
        for j in range(k):
            v, i = S.max(dim=0)
            vs.append(v); ids.append(i)
            if j < k - 1:
                S.scatter_(0, i[None, :], float("-inf"))
        return torch.stack(vs), torch.stack(ids)
    v, i = torch.topk(S.T.contiguous(), k, dim=1)
    return v.T, i.T


def topk_search(qtexts, dtexts, view, ngram, k=20, device="cuda:0", qblock=512):
    """Returns (row_idx, doc_idx, score, rank) arrays of the exact cosine top-k docs for each query."""
    Xd = counts(dtexts, view, ngram)
    Xq = counts(qtexts, view, ngram)
    n = Xd.shape[0]
    df = np.bincount(Xd.indices, minlength=N_FEATURES)
    idf = np.log((1 + n) / (1 + df)) + 1.0
    Xd, Xq = _tfidf(Xd, idf).tocsr(), _tfidf(Xq, idf).tocsr()
    # only columns used by some query can contribute to a dot product (norms were computed on full vectors)
    used = np.unique(Xq.indices)
    remap = np.full(N_FEATURES, -1, dtype=np.int64); remap[used] = np.arange(len(used))
    Xd = Xd.tocsc()[:, used].tocsr(); Xd.sort_indices()
    Xq = Xq[:, used].tocsr()
    D = torch.sparse_csr_tensor(torch.from_numpy(Xd.indptr.astype(np.int64)), torch.from_numpy(Xd.indices.astype(np.int64)),
                                torch.from_numpy(Xd.data), size=Xd.shape, device=device)
    k = min(k, n)
    rows, cols, vals, ranks = [], [], [], []
    for s in range(0, Xq.shape[0], qblock):
        blk = Xq[s:s + qblock].tocoo()                         # densify (used_features, block) on the GPU
        qb = torch.zeros((len(used), blk.shape[0]), dtype=torch.float32, device=device)
        qb[torch.from_numpy(blk.col.astype(np.int64)).to(device), torch.from_numpy(blk.row.astype(np.int64)).to(device)] = \
            torch.from_numpy(blk.data).to(device)
        scores = torch.sparse.mm(D, qb)                        # (n_docs, block)
        v, i = _topk_dim0(scores, k)                    # (k, block)
        v, i = v.T.cpu().numpy(), i.T.cpu().numpy()
        keep = v > 1e-6
        b = v.shape[0]
        rows.append(np.repeat(np.arange(s, s + b), k).reshape(b, k)[keep])
        cols.append(i[keep]); vals.append(v[keep])
        ranks.append(np.tile(np.arange(1, k + 1), (b, 1))[keep])
        del scores
    torch.cuda.empty_cache()
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks).astype(np.int32)


class GpuIndex:
    """Char-TF-IDF index of `dtexts` resident on one GPU; exact cosine top-k for arbitrarily many queries (streamed in chunks).
    Same weighting as topk_search (hashed n-grams, sublinear TF, smoothed IDF from the docs, L2 norm)."""

    def __init__(self, dtexts, view, ngram, device="cuda:0"):
        self.view, self.ngram, self.device = view, ngram, device
        Xd = counts(dtexts, view, ngram)
        n = Xd.shape[0]
        df = np.bincount(Xd.indices, minlength=N_FEATURES)
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        Xd = _tfidf(Xd, self.idf).tocsr()
        self.used = np.unique(Xd.indices)                        # query columns absent from docs cannot score
        self.remap = np.full(N_FEATURES, -1, dtype=np.int64); self.remap[self.used] = np.arange(len(self.used))
        Xd.indices = self.remap[Xd.indices].astype(np.int32); Xd._shape = (n, len(self.used))
        self.n = n
        self.D = torch.sparse_csr_tensor(torch.from_numpy(Xd.indptr.astype(np.int64)), torch.from_numpy(Xd.indices.astype(np.int64)),
                                         torch.from_numpy(Xd.data), size=Xd.shape, device=device)

    def search(self, qtexts, k=1, qblock=1024, chunk=500_000):
        """Yields (row_offset_array, doc_idx, score, rank) per chunk of queries (row indices are global)."""
        k = min(k, self.n)
        for c0 in range(0, len(qtexts), chunk):
            Xq = _tfidf(counts(qtexts[c0:c0 + chunk], self.view, self.ngram), self.idf).tocoo()
            keep = self.remap[Xq.col] >= 0
            Xq = sp.csr_matrix((Xq.data[keep], (Xq.row[keep], self.remap[Xq.col[keep]])), shape=(Xq.shape[0], len(self.used)))
            rows, cols, vals, ranks = [], [], [], []
            for s in range(0, Xq.shape[0], qblock):
                blk = Xq[s:s + qblock].tocoo()
                qb = torch.zeros((len(self.used), blk.shape[0]), dtype=torch.float32, device=self.device)
                qb[torch.from_numpy(blk.col.astype(np.int64)).to(self.device), torch.from_numpy(blk.row.astype(np.int64)).to(self.device)] = \
                    torch.from_numpy(blk.data).to(self.device)
                v, i = _topk_dim0(torch.sparse.mm(self.D, qb), k)
                v, i = v.T.cpu().numpy(), i.T.cpu().numpy()
                ok = v > 1e-6; b = v.shape[0]
                rows.append(np.repeat(np.arange(c0 + s, c0 + s + b), k).reshape(b, k)[ok])
                cols.append(i[ok]); vals.append(v[ok]); ranks.append(np.tile(np.arange(1, k + 1), (b, 1))[ok])
                del qb
            yield np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks).astype(np.int32)
