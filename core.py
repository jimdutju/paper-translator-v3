"""Core of the v2 translation pipeline: text lines -> paragraphs -> kinds.

Two text sources
----------------
* vector : pages with a real text layer (PyMuPDF blocks/lines/spans)
* ocr    : scanned pages (tesseract TSV at 400 dpi, cached)

Everything downstream works on the resulting paragraph list, so the rest of
the pipeline does not care where the text came from.
"""
from __future__ import annotations

import hashlib
import os
import re
import statistics
import subprocess
from dataclasses import dataclass, field

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_OCR = os.path.join(HERE, "_ocr")

LIG = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
    "\u00ad": "", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2009": " ", "\u00a0": " ",
}

MATH_FONTS = (
    "MTMI", "MTSY", "MTEX", "MSAM", "MSBM", "CMMI", "CMSY", "CMEX",
    "Symbol", "Euclid", "MTExtra", "MTSYN", "STIX", "cmmi", "cmsy", "cmex",
    "rsfs", "wasy", "eusm", "eurm", "eufm", "msam", "msbm", "MathPi",
    "MathIt", "cmmib", "cmbsy", "dsrom", "dsss", "MTMI", "Gen_Math",
)

CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?s?\.?|f[il1][cg]\.?|tab(?:le)?s?\.?|chart|plate|scheme|"
    r"exhibit|panel|box)\s*[\.\-–—:]?\s*"
    r"(?:[A-Z]?\s*[\.\-–—:]?\s*)?(?:[0-9]+|[IVXLC]+)\b",
    re.I,
)
HEAD_RE = re.compile(
    r"^\s*(?:"
    r"[IVXLC]+\.|[A-Z]\.(?:\s|\d)|"           # I.  A.
    r"\d+(?:\.\d+)*\.?\s|"                      # 1.  2.3
    r"(?:section|chapter|part|appendix)\b"
    r")",
    re.I,
)
NOTE_MARK_RE = re.compile(
    r"^\s*(?:[*\u2020\u2021\u00a7\u00b6\u2751\u2752\u2660\u2663\u2665\u2666{}\[\]]+\s*"
    r"|\d{1,2}\s+(?=[A-Z\u4e00-\u9fff]))")

# "Table 2 shows that ..." / "Tables III and IV present ..." is body text that
# merely starts with a table number
TABLE_SENTENCE_RE = re.compile(
    r"(?i)^\s*(?:tables?|figures?|fig\.?|panels?|charts?)\s*[0-9IVX]+"
    r"(?:\s*(?:,|and|through|-|–)\s*[0-9IVX]+)*\s*"
    r"(?:also\s+|further\s+)?(?:provides?|shows?|presents?|reports?|gives?|lists?"
    r"|summari[sz]es?|contains?|displays?|indicates?|depicts?|compares?|is|are|was|were)\b")
MATH_CHARS = set("=+\u2212\u2013\u2014<>\u2264\u2265\u00d7\u00f7\u2211\u222b\u221a|{}\\^_~`$")
SYM_CHARS = set("=+\u2212<>\u2264\u2265\u00d7\u00f7\u00b1\u2211\u222b\u221a\u221e\u2248\u2260\u2202\u2207\u2192\u2190")

STOP = set("""the and but for than that this these those its was were been being can could
may might must shall should will would not only also more most such which whose when where while
there here have has had one two three all any both each few other some over under about into out
down very much many between among during within from with without they them their then thus
however therefore because assume given consider suppose note let see also same well make makes
using used uses figure table section equation appendix theorem lemma proof result results
if when as at by in on of to a an is are be it we i you he she his her our your
""".split())


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def clean(text: str) -> str:
    for a, b in LIG.items():
        text = text.replace(a, b)
    text = text.replace("\ufffd", " ")
    text = "".join(ch if ch >= " " or ch in "\t" else " " for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def join_fragments(parts):
    out = ""
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if not out:
            out = s
        elif out.endswith("-") and s[:1].islower():
            out = out[:-1] + s
        else:
            out += " " + s
    return out


BOILER_RE = re.compile(
    r"(?i)(this content downloaded|all use subject to|jstor terms|terms and conditions"
    r"|downloaded from \d+\.\d+\.\d+\.\d+|downloaded by|subscription information"
    r"|registered in england|informa ltd|taylor & francis)")


def is_garbage(text: str) -> bool:
    if not text:
        return True
    bad = sum(1 for c in text if ord(c) < 32 and c not in "\t")
    bad += text.count("\ufffd")
    return bad / max(1, len(text)) > 0.12


def words(text):
    return re.findall(r"[A-Za-z]{2,}", text)


def has_prose(text) -> bool:
    w = [x.lower() for x in words(text)]
    return sum(1 for x in w if x in STOP) >= 2 and len(w) >= 6


def sym_ratio(text):
    s = [c for c in text if not c.isspace()]
    if not s:
        return 0.0
    return sum(1 for c in s if c in MATH_CHARS) / len(s)


def centered(rect, x0, x1):
    c = (rect.x0 + rect.x1) / 2
    return abs(c - (x0 + x1) / 2) < 0.12 * (x1 - x0) + 12


# --------------------------------------------------------------------------- #
# line extraction
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    rect: pymupdf.Rect
    text: str
    size: float
    bold: bool = False
    math: float = 0.0          # share of chars set in maths fonts
    superscript: bool = False


@dataclass
class Para:
    rect: pymupdf.Rect
    lines: list
    text: str
    size: float
    bold: bool = False
    math: float = 0.0
    kind: str = "body"
    col: int = 0
    band: int = 0
    order: int = 0
    reason: str = ""

    @property
    def n_lines(self):
        return len(self.lines)


def _span_math(font: str) -> bool:
    return any(f in font for f in MATH_FONTS)


def _find_gutter(chars, W):
    """x position of the empty band between two text columns (or None)."""
    if not chars:
        return None
    rows = 40
    H = max(c[3] for c in chars) or 1
    grid = [bytearray(int(W) + 2) for _ in range(rows)]
    used = [0] * rows
    for x0, y0, x1, y1 in chars:
        r = min(rows - 1, max(0, int(y0 / H * rows)))
        used[r] += 1
        a, b = max(0, int(x0) - 1), min(int(W) + 1, int(x1) + 1)
        for x in range(a, b):
            grid[r][x] = 1
    counts = [sum(grid[r][x] for r in range(rows)) for x in range(int(W) + 2)]
    maxc = max(counts) or 1
    thresh = 0.12 * maxc
    best, cur = None, None
    for x in range(int(0.28 * W), int(0.72 * W)):
        if counts[x] <= thresh:
            if cur is None:
                cur = x
        else:
            if cur is not None:
                if best is None or x - cur > best[1] - best[0]:
                    best = (cur, x)
                cur = None
    if cur is not None and (best is None or int(0.72 * W) - cur > best[1] - best[0]):
        best = (cur, int(0.72 * W))
    if best and best[1] - best[0] >= 10:
        return (best[0] + best[1]) / 2
    return None


def page_chars(page):
    """Every character box of the page's text layer."""
    chars = []
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            dirx, diry = l["dir"]
            if abs(dirx - 1) > 1e-3 or abs(diry) > 1e-3:
                continue
            for s in l["spans"]:
                for c in s["chars"]:
                    chars.append((c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3]))
    return chars


def doc_gutter(doc, pages=None):
    """The page gutter this whole document agrees on (or None).

    Front pages (title, abstract running the full width) have no clean band of
    their own; without the document's value their lines merge the two columns
    into one paragraph, which then covers — and erases — whatever sits beside
    it.
    """
    import statistics
    gs = []
    for i in (pages if pages is not None else range(doc.page_count)):
        g = _find_gutter(page_chars(doc[i]), doc[i].rect.width)
        if g:
            gs.append(g)
    if len(gs) < 3:
        return None
    med = statistics.median(gs)
    if sum(1 for g in gs if abs(g - med) <= 4.0) < 0.6 * len(gs):
        return None
    return med


def _straddles(chars, x, W):
    """Is there text on both sides of x, and little on it?"""
    left = sum(1 for c in chars if c[2] <= x - 3)
    right = sum(1 for c in chars if c[0] >= x + 3)
    across = sum(1 for c in chars if c[0] < x - 3 < x + 3 < c[2])
    return left > 30 and right > 30 and across < 0.1 * (left + right)


def extract_vector(page, gutter_hint=None):
    """Line list from a real text layer.

    Word-level spans are common in this corpus, so lines are rebuilt from the
    character boxes: find the column gutter first, then group characters into
    visual lines inside each column.
    """
    chars = []
    spans = []
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            dirx, diry = l["dir"]
            if abs(dirx - 1) > 1e-3 or abs(diry) > 1e-3:
                continue
            for s in l["spans"]:
                t = "".join(c["c"] for c in s["chars"])
                if not t:
                    continue
                for c in s["chars"]:
                    chars.append((c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3]))
                spans.append(dict(rect=pymupdf.Rect(s["bbox"]), text=t,
                                  size=s["size"], font=s["font"], flags=s["flags"]))
    if not spans:
        return []
    W = page.rect.width
    gutter = _find_gutter(chars, W)
    if gutter is None and gutter_hint is not None and _straddles(chars, gutter_hint, W):
        gutter = gutter_hint

    groups = []
    for s in sorted(spans, key=lambda s: (s["rect"].y0, s["rect"].x0)):
        cx = (s["rect"].x0 + s["rect"].x1) / 2
        col = 0 if (gutter is None or cx < gutter) else 1
        if gutter is not None and s["rect"].x0 < gutter < s["rect"].x1:
            # a span crossing the gutter: assign to the side holding most of it
            col = 0 if (gutter - s["rect"].x0) > (s["rect"].x1 - gutter) else 1
        for g in groups:
            if g["col"] != col:
                continue
            r = g["rect"]
            ov = min(r.y1, s["rect"].y1) - max(r.y0, s["rect"].y0)
            if ov > 0.45 * min(r.height, s["rect"].height) and \
                    abs(r.y0 - s["rect"].y0) < 0.6 * max(r.height, s["rect"].height) and \
                    (s["rect"].x0 - r.x1 < 30) and (r.x0 - s["rect"].x1 < 30):
                g["rect"] |= s["rect"]
                g["spans"].append(s)
                break
        else:
            groups.append(dict(rect=+s["rect"], spans=[s], col=col))

    out = []
    for g in groups:
        ss = sorted(g["spans"], key=lambda s: s["rect"].x0)
        size = max(s["size"] for s in ss)
        parts, prev = [], None
        for s in ss:
            sup = bool(s["flags"] & 1) or (s["size"] < 0.82 * size and
                                           re.fullmatch(r"[0-9*†‡]{1,3}", s["text"].strip() or "x"))
            if prev is not None and s["rect"].x0 - prev > 0.12 * size and not sup:
                parts.append(" ")
            if sup and re.fullmatch(r"[0-9*]{1,3}", s["text"].strip() or ""):
                parts.append("\u27e6" + s["text"].strip() + "\u27e7")
            else:
                parts.append(s["text"])
            prev = s["rect"].x1
        txt = clean("".join(parts))
        if not txt:
            continue
        math = sum(len(s["text"]) for s in ss if _span_math(s["font"])) / max(1, len(txt))
        bold = any("bold" in s["font"].lower() or "heavy" in s["font"].lower() for s in ss)
        out.append(Line(+g["rect"], txt, round(size, 1), bold, math))
    return out


def _ocr_png(page, tag, mask=None):
    d = os.path.join(CACHE_OCR, "v2_" + tag)
    os.makedirs(d, exist_ok=True)
    suffix = "_fig" if mask else ""
    png = os.path.join(d, "p%03d%s.png" % (page.number, suffix))
    tsv = os.path.join(d, "p%03d%s.tsv" % (page.number, suffix))
    if not os.path.exists(tsv):
        if not os.path.exists(png):
            if mask:
                import numpy as np
                s = 400 / 72
                pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
                for r in mask:
                    x0, y0, x1, y1 = [int(v * s) for v in (r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2)]
                    img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
                pymupdf.Pixmap(pymupdf.csGRAY, img.shape[1], img.shape[0],
                               img.tobytes(), 0).save(png)
            else:
                page.get_pixmap(matrix=pymupdf.Matrix(400 / 72, 400 / 72)).save(png)
        subprocess.run(["tesseract", png, tsv[:-4], "-l", "eng", "--psm", "3", "tsv"],
                       check=True, capture_output=True)
    return tsv


FUNC_WORDS = {"of", "if", "in", "is", "to", "at", "by", "as", "be", "it",
              "on", "or", "an", "we", "no", "so", "up", "my", "its"}
REPAIR_LOG = []          # (tag, page, old, new, applied|review)


def _reread_word(page, x, y, w, h, nchars, lang="eng", dpi=600):
    """OCR a single short token from a tight crop, bypassing line context."""
    import os
    import subprocess
    import tempfile
    try:
        pm = page.get_pixmap(dpi=dpi,
                             clip=pymupdf.Rect(x - 1.0, y - 1.0, x + w + 1.0, y + h + 1.0))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pm.save(path)
        psm = "10" if nchars == 1 else "8"
        got = subprocess.run(["tesseract", path, "stdout", "--psm", psm, "-l", lang],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout.strip()
        os.unlink(path)
        return got
    except Exception:                          # noqa: BLE001
        return ""


# Greek letters that look exactly like a Latin one; without this the Greek
# reading of "T" would look like a disagreement and the repair would be dropped.
HOMOGLYPH = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o", "α": "a", "ε": "e", "κ": "k", "ρ": "p", "τ": "t", "υ": "u",
    "ι": "i", "ν": "v", "γ": "y", "χ": "x",
})


def _skeleton(s):
    return (s or "").strip().rstrip(".,;:!?'\"\u2019\u201d").translate(HOMOGLYPH)


GREEK_SYMBOLS = set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")

# (Greek letter the ell model reported, what the English model reported).
# Only pairs whose precision was checked against the pixels belong here; see
# _diag/beta_precision_01.png for the beta measurements on paper 01.
#
# Rationale for trusting ell here at all: "-l eng" has no Greek class in its
# output alphabet, so "eng says 8, ell says beta" is not two models
# disagreeing about one hypothesis -- ell is the only model that can even
# propose Greek. Everything outside this table still goes to review.
GREEK_APPLY = {
    ("β", "8"), ("β", "6"), ("β", "B"), ("β", "b"),
    ("β", "6B"), ("β", "8B"), ("β", "6b"), ("β", "8b"),
}


def _greek_is_safe(old, greek):
    """True only when this (reading, Greek) pair has been verified by hand."""
    return (greek, (old or "").rstrip(".,;:!?'")) in GREEK_APPLY


def _accept_greek(old, ell):
    """A lone Greek letter where the OCR saw a digit or a digit+capital is a
    maths symbol: "8" / "6B" are both ways of mis-reading beta."""
    e = (ell or "").strip().rstrip(".,;:!?'\"")
    if len(e) != 1 or e not in GREEK_SYMBOLS:
        return None
    o = old.rstrip(".,;:!?'")
    if len(o) == 1 and re.fullmatch(r"[A-Za-z0-9]", o):
        return e
    if len(o) == 2 and re.fullmatch(r"\d[A-Z]", o):
        return e
    return None


def _accept(old, new):
    """Only the two safest corrections are applied automatically."""
    if not new or new == old:
        return False
    if not re.fullmatch(r"[A-Za-z0-9]{1,2}[.,;:!?']?", new):
        return False
    o = old.rstrip(".,;:!?'")
    n = new.rstrip(".,;:!?'")
    if not n or len(n) > 2:
        return False
    if re.fullmatch(r"\d", o) and re.fullmatch(r"[A-Za-z]", n):
        return True                                    # a digit that is a variable
    if (re.fullmatch(r"[A-Za-z]{1,2}", o) and len(o) == len(n)
            and n.lower() in FUNC_WORDS and o.lower() not in FUNC_WORDS):
        return True                                    # a broken function word
    return False


FIXED_MAP = {"¢": "t", "€": "t"}   # italic t, read as a currency sign


def _apply_fixed_map(rows, tag, pno):
    """Deterministic substitutions that need no second opinion.

    Applied per character, not per token: the mis-read usually sits inside a
    longer token such as "Z(¢)".
    """
    for r in rows:
        t = r["text"] or ""
        if not t:
            continue
        new = t
        for ch, rep in FIXED_MAP.items():
            if ch in new:
                new = new.replace(ch, rep)
        if new != t:
            r["text"] = new
            REPAIR_LOG.append((tag, pno, t.strip(), new.strip(), "applied"))
    return rows


def _repair_short_tokens(page, rows, scale, tag, dpi=600, max_conf=85.0):
    """Re-read short, low-confidence tokens from a tight crop.

    Tesseract mis-reads italic letters on long lines ("T" -> "7") because the
    crop it works on decides the answer; the same glyph alone is read correctly.
    Results are cached per paper, and every change is logged for review.
    """
    import json
    import os
    cache_path = os.path.join(CACHE_OCR, "repair_%s.json" % tag)
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:                      # noqa: BLE001
            cache = {}
    fresh = False
    rows = _apply_fixed_map(rows, tag, page.number + 1)
    out = []
    for r in rows:
        t = (r["text"] or "").strip()
        try:
            conf = float(r["conf"])
        except (TypeError, ValueError):
            conf = 0.0
        # A lone glyph is where the maths symbols live, and there the confidence
        # is worthless: beta read as "8" scores 96.9 (OCR调查记录.md 3.5). Gating
        # the second opinion on low confidence skipped 679 of the 871
        # one-character tokens in paper 01. Re-read a lone glyph unconditionally;
        # keep the confidence gate for two-character tokens.
        is_alnum = bool(0 < len(t) <= 2 and re.search(r"[A-Za-z0-9]", t))
        if not (is_alnum and (len(t) == 1 or conf < max_conf)):
            out.append(r)
            continue
        x = int(r["left"]) / scale
        y = int(r["top"]) / scale
        w = int(r["width"]) / scale
        h = int(r["height"]) / scale
        key = "%d:%.0f:%.0f" % (page.number, x, y)
        hit = cache.get(key)
        if hit is None or len(hit) != 2:
            hit = [_reread_word(page, x, y, w, h, len(t), "eng", dpi),
                   _reread_word(page, x, y, w, h, len(t), "ell", dpi)]
            cache[key] = hit
            fresh = True
        eng, ell = hit
        # both models must agree (after folding Greek look-alikes) before a
        # correction is applied; otherwise it only goes to the review log
        agree = bool(_skeleton(eng)) and _skeleton(eng) == _skeleton(ell)
        gk = _accept_greek(t, ell)
        if gk:
            suffix = t[len(t.rstrip(".,;:!?'")):]
            if _greek_is_safe(t, gk):
                # Verified pairs: the glyph really is that Greek letter. Apply
                # it; the English reading is a digit only because -l eng has no
                # Greek class at all.
                r["text"] = gk + suffix
                REPAIR_LOG.append((tag, page.number + 1, t, gk + suffix, "applied"))
            else:
                # Unverified pair: the Greek model still confuses accents and
                # accent-like shapes, so record it and leave the text alone.
                REPAIR_LOG.append((tag, page.number + 1, t, gk + suffix, "review"))
        elif agree and _accept(t, eng):
            r["text"] = eng
            REPAIR_LOG.append((tag, page.number + 1, t, eng, "applied"))
        elif not agree and (eng or ell) and t not in (eng, ell):
            REPAIR_LOG.append((tag, page.number + 1, t,
                               "%s|%s" % (eng, ell), "review"))
        elif not agree and _accept(t, eng):
            REPAIR_LOG.append((tag, page.number + 1, t, eng, "review"))
        out.append(r)
    if fresh:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception:                      # noqa: BLE001
            pass
    return out


def extract_ocr(page, tag, mask=None):
    """Line list from a tesseract TSV (400 dpi)."""
    import csv
    tsv = _ocr_png(page, tag, mask)
    scale = 400 / 72.0
    rows = [r for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t")
            if r["level"] == "5" and (r["text"] or "").strip()]
    groups = {}
    for r in rows:
        key = (int(r["block_num"]), int(r["par_num"]), int(r["line_num"]))
        groups.setdefault(key, []).append(r)
    rows = _repair_short_tokens(page, rows, scale, tag)
    out = []
    raws = []
    for key in sorted(groups):
        ws = sorted(groups[key], key=lambda r: int(r["left"]))
        txt = clean(" ".join(w["text"] for w in ws))
        if not txt:
            continue
        x0 = min(int(w["left"]) for w in ws) / scale
        x1 = max(int(w["left"]) + int(w["width"]) for w in ws) / scale
        y0 = min(int(w["top"]) for w in ws) / scale
        y1 = max(int(w["top"]) + int(w["height"]) for w in ws) / scale
        hts = [int(w["height"]) for w in ws if re.search(r"[A-Za-z]{3}", w["text"])] \
            or [int(w["height"]) for w in ws]
        h = max(hts) / scale
        size = round(h * 0.75, 1)
        line = Line(pymupdf.Rect(x0, y0, x1, y1), txt, size)
        out.append(line)
        raws.append(ws)

    # tesseract sometimes joins the two columns into one "line": split such a
    # line at the page gutter when the words leave a gap there
    g = _gutter([l.rect for l in out], page.rect.width, page.rect.height)
    if g:
        merged, raw_merged = [], []
        for line, ws in zip(out, raws):
            if not (line.rect.x0 < g - 2 < line.rect.x1 + 2):
                merged.append(line)
                raw_merged.append(ws)
                continue
            left = [w for w in ws if int(w["left"]) + int(w["width"]) / 2 < g]
            right = [w for w in ws if int(w["left"]) + int(w["width"]) / 2 >= g]
            if not left or not right:
                merged.append(line)
                raw_merged.append(ws)
                continue

            def mk(part):
                if not part:
                    return None
                a = min(int(w["left"]) for w in part) / scale
                b = max(int(w["left"]) + int(w["width"]) for w in part) / scale
                c = min(int(w["top"]) for w in part) / scale
                d = max(int(w["top"]) + int(w["height"]) for w in part) / scale
                txt = clean(" ".join(w["text"] for w in part))
                return Line(pymupdf.Rect(a, c, b, d), txt, line.size) if txt else None

            for part in (left, right):
                ln = mk(part)
                if ln is not None:
                    merged.append(ln)
        out = merged
    return out


# --------------------------------------------------------------------------- #
# columns and paragraphs
# --------------------------------------------------------------------------- #
def _gutter(boxes, W, H=None):
    """Widest x-band in the middle of the page that (almost) no box covers."""
    if H:
        boxes = [b for b in boxes if b.y0 > 0.05 * H and b.y1 < 0.95 * H]
    if len(boxes) < 4:
        return None
    n = len(boxes)
    occ = [0] * (int(W) + 2)
    for r in boxes:
        a, b = max(0, int(r.x0)), min(int(W) + 1, int(r.x1))
        for x in range(a, b):
            occ[x] += 1
    thresh = 0.25 * n
    best, cur = None, None
    for x in range(int(0.28 * W), int(0.72 * W)):
        if occ[x] <= thresh:
            if cur is None:
                cur = x
        else:
            if cur is not None:
                if best is None or x - cur > best[1] - best[0]:
                    best = (cur, x)
                cur = None
    if cur is not None and (best is None or int(0.72 * W) - cur > best[1] - best[0]):
        best = (cur, int(0.72 * W))
    if best and best[1] - best[0] >= 10:
        return (best[0] + best[1]) / 2
    return None


def find_columns(lines, page_w, page_h=None):
    """Column x-ranges: split at the widest empty vertical band, if any."""
    if not lines:
        return [(0, page_w)]
    boxes = [l.rect for l in lines]
    g = _gutter(boxes, page_w, page_h)
    sides = [(0, g, lines), (g, page_w, lines)] if g else [(0, page_w, lines)]
    cols = []
    for lo, hi, _ in sides:
        mem = [l for l in lines if lo <= (l.rect.x0 + l.rect.x1) / 2 < hi]
        if len(mem) < 2:
            continue
        x0 = sorted(l.rect.x0 for l in mem)[int(0.04 * (len(mem) - 1))]
        x1 = sorted(l.rect.x1 for l in mem)[int(0.90 * (len(mem) - 1))]
        cols.append((x0 - 1, max(x1, x0 + 60) + 1))
    if not cols:
        x0 = sorted(b[0] for b in boxes)[0]
        x1 = sorted(b[2] for b in boxes)[-1]
        cols = [(x0, x1)]
    # ranges that overlap are not two columns: merging them is what keeps the
    # two text streams from being painted on top of each other
    cols.sort()
    merged = []
    for c in cols:
        if merged and c[0] < merged[-1][1] - 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], c[1]))
        else:
            merged.append(c)
    return merged or [(0, page_w)]


def _col_of(line, cols):
    cx = (line.rect.x0 + line.rect.x1) / 2
    best, dist = 0, 1e9
    for i, (x0, x1) in enumerate(cols):
        d = 0 if x0 - 10 <= cx <= x1 + 10 else min(abs(cx - x0), abs(cx - x1))
        if d < dist:
            best, dist = i, d
    return best


def _is_cjk_char(c):
    return bool(c) and ("\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
                        or "\uff00" <= c <= "\uffef")


def build_paragraphs(lines, page_w, use_size=True, page_h=None):
    if not lines:
        return [], []
    cols = find_columns(lines, page_w, page_h)
    by_col = {}
    for l in lines:
        by_col.setdefault(_col_of(l, cols), []).append(l)

    paras = []
    for ci in sorted(by_col):
        ls = by_col[ci]
        ls.sort(key=lambda l: (l.rect.y0, l.rect.x0))
        # merge fragments sitting on the same visual line
        vlines = []
        for l in ls:
            # fragments of one printed line: same baseline, and the next box
            # starts where the previous one ended.  The boxes very often
            # overlap by a point or two (italic maths has slanted, padded
            # boxes), so a negative gap is still the same line — without this
            # an inline formula such as "r(t, T ) =" becomes its own paragraph
            # and is later left behind while its sentence reflows.
            gap = l.rect.x0 - vlines[-1].rect.x1 if vlines else 0.0
            # a formula fragment has a taller box than the prose around it and
            # therefore sits a few points lower: allow that, but never a whole
            # line pitch (adjacent lines of 10 pt text are ~12 pt apart)
            dy_ok = (abs(vlines[-1].rect.y0 - l.rect.y0)
                     < max(2.5, 0.45 * max(l.size, vlines[-1].size))) if vlines else False
            if vlines and dy_ok and -4.0 <= gap < 30:
                v = vlines[-1]
                # two Latin fragments butted together need the space the box
                # gap lost ("returnsr(t, T )"); two CJK fragments must not get
                # one, and punctuation never takes a space before it
                sep = " "
                if gap <= 0.8:
                    before = v.text[-1:] or " "
                    after = l.text[:1] or " "
                    if _is_cjk_char(before) or _is_cjk_char(after):
                        sep = ""
                    elif after in ",.;:)]}%'’”":
                        sep = ""
                v.text = clean(v.text + sep + l.text)
                v.rect |= l.rect
                v.size = max(v.size, l.size)
                v.bold = v.bold or l.bold
            else:
                vlines.append(Line(+l.rect, l.text, l.size, l.bold, l.math, l.superscript))
        if not vlines:
            continue
        # median line advance of this column
        dys = [b.rect.y0 - a.rect.y0 for a, b in zip(vlines, vlines[1:])]
        lh = statistics.median([v.rect.height for v in vlines]) or 10.0
        adv = statistics.median([d for d in dys if 3 < d < 3 * lh] or [1.25 * lh])
        right = sorted(v.rect.x1 for v in vlines)
        col_right = right[int(0.92 * (len(right) - 1))] if right else 0
        cur = []
        for v in vlines:
            if not cur:
                cur = [v]
                continue
            p = cur[-1]
            dy = v.rect.y0 - p.rect.y0
            para_x0 = min(x.rect.x0 for x in cur)
            indented = (v.rect.x0 > para_x0 + 10 and p.rect.x1 < col_right - 25
                        and v.rect.x0 > p.rect.x0 + 8)
            same_size = (not use_size) or abs(v.size - p.size) < 1.6
            same_bold = v.bold == p.bold
            if dy < 1.45 * adv and not indented and same_size and same_bold:
                cur.append(v)
            else:
                paras.append(_mk(cur, ci))
                cur = [v]
        if cur:
            paras.append(_mk(cur, ci))
    paras.sort(key=lambda p: (p.col, p.rect.y0))
    for i, p in enumerate(paras):
        p.order = i
    return paras, cols


def _mk(vlines, col):
    rect = +vlines[0].rect
    for v in vlines[1:]:
        rect |= v.rect
    sizes = sorted(v.size for v in vlines)
    text = join_fragments(v.text for v in vlines)
    return Para(rect=rect,
                lines=[(+v.rect, v.text) for v in vlines],
                text=text,
                size=round(statistics.median(sizes), 1),
                bold=any(v.bold for v in vlines),
                math=sum(v.math for v in vlines) / max(1, len(vlines)),
                col=col)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _merge_rects(rects, pad=6):
    """Union-find merge of rects that come within `pad` points of each other."""
    rects = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        for r in rects:
            for i, m in enumerate(out):
                g = pymupdf.Rect(m.x0 - pad, m.y0 - pad, m.x1 + pad, m.y1 + pad)
                if (r & g).get_area() > 1:
                    out[i] = m | r
                    changed = True
                    break
            else:
                out.append(+r)
        rects = out
    return rects


def figure_rects_vector(page):
    area = page.rect.get_area()
    elems = []
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            if 20 < r.width < page.rect.width and 20 < r.height < page.rect.height:
                elems.append(+r)
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width >= 12 and r.height >= 12 and r.get_area() < 0.75 * area:
            elems.append(+r)
    # drop table rules / underlines: very wide and very flat
    elems = [r for r in elems if not (r.height < 4 or r.width < 4)]
    return _merge_rects(elems, pad=8)


_LINE_CACHE = {}


def _h_runs(ink, minrun):
    """Vectorised: every horizontal True run of length >= minrun."""
    import numpy as np
    pad = np.pad(ink.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(pad, axis=1)
    rs, cs = np.nonzero(d == 1)
    re_, ce = np.nonzero(d == -1)
    keep = (ce - cs) >= minrun
    return list(zip(rs[keep], cs[keep], ce[keep]))


def frame_boxes(page):
    """Boxes delimited by long straight ink runs (chart frames, table rules)."""
    import numpy as np
    # key on the document too: batch.py walks many papers in one process and a
    # bare page number would hand paper B's page 2 the frames of paper A's.
    key = (getattr(page.parent, "name", "") or id(page.parent), page.number)
    if key in _LINE_CACHE:
        return _LINE_CACHE[key]
    s = 2.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = img < 170
    H, W = ink.shape
    minrun = int(55 * s)

    hsegs = [(y * 2, a, b) for y, a, b in _h_runs(ink[::2], minrun)]
    vsegs = [(x * 2, a, b) for x, a, b in _h_runs(ink.T[::2], minrun)]

    # 1) join collinear horizontal pieces (ticks break a frame into pieces)
    rules = []
    for y, a, b in sorted(hsegs):
        for g in rules:
            if abs(y - g[0]) <= 4 and not (b < g[1] - 2 or a > g[2] + 40):
                g[1] = min(g[1], a)
                g[2] = max(g[2], b)
                g[0] = y
                break
        else:
            rules.append([y, a, b])

    # 2) stack rules that share their horizontal extent: this is a plot frame
    groups = []
    for y, a, b in rules:
        for g in groups:
            overlap = min(b, g[3]) - max(a, g[2])
            if abs(y - g[1]) <= 320 and overlap > 0.5 * min(b - a, g[3] - g[2]):
                g[1] = max(g[1], y)
                g[2] = min(g[2], a)
                g[3] = max(g[3], b)
                break
        else:
            groups.append([y, y, a, b])
    boxes = [pymupdf.Rect(g[2] / s, g[0] / s, g[3] / s, g[1] / s)
             for g in groups if g[1] - g[0] >= 25 * s and g[3] - g[2] >= 45 * s]
    for y, a, b in rules:
        boxes.append(pymupdf.Rect(a / s, y / s - 0.5, b / s, y / s + 0.5))
    for x, a, b in vsegs:
        boxes.append(pymupdf.Rect(x / s - 0.5, a / s, x / s + 0.5, b / s))
    if len(_LINE_CACHE) > 400:
        _LINE_CACHE.clear()
    _LINE_CACHE[key] = boxes
    return boxes


def _dilate(mask, k):
    import numpy as np
    m = mask
    for _ in range(k):
        p = np.pad(m, 1)
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def figure_rects_scan(page, lines):
    """Figure blobs on a scanned page: dense ink that no OCR line covers."""
    import numpy as np
    s = 2.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = img < 170
    H, W = ink.shape
    cover = np.zeros_like(ink)
    for l in lines:
        x0, y0, x1, y1 = [int(v * s) for v in (l.rect.x0, l.rect.y0, l.rect.x1, l.rect.y1)]
        cover[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = True
    cand = ink & ~cover
    cand = _dilate(cand, 4)
    cell = 6
    gh, gw = H // cell, W // cell
    frac = cand[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    grid = frac > 0.08
    seen = np.zeros_like(grid)
    rects = []
    for y in range(gh):
        if not grid[y].any():
            continue
        for x in np.nonzero(grid[y] & ~seen[y])[0]:
            stack = [(y, int(x))]
            seen[y, x] = True
            xs, ys = [], []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            r = pymupdf.Rect(min(xs) * cell / s, min(ys) * cell / s,
                             (max(xs) + 1) * cell / s, (max(ys) + 1) * cell / s)
            if r.width >= 45 and r.height >= 45 and r.get_area() > 2500:
                rects.append(r)
    rects += [r for r in frame_boxes(page) if r.width >= 40 and r.height >= 20]
    return _merge_rects(rects, pad=12)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def doc_body_size(pages):
    """Main text size: the largest size that still carries a real share of the
    text mass (footnotes are smaller, headings/titles are too few)."""
    mass = {}
    for paras in pages:
        for p in paras:
            if len(p.text) > 40 and 4.0 <= p.size <= 30.0:
                k = round(p.size * 2) / 2
                mass[k] = mass.get(k, 0) + len(p.text)
    if not mass:
        return 10.0
    top = max(mass.values())
    cand = [k for k, v in mass.items() if v >= 0.15 * top]
    return round(max(cand), 1)


def classify(paras, rect, body_size, figs, extra=None):
    """Assign a kind to every paragraph. Mutates and returns `paras`.

    `extra`: refs (bool, this page is part of the reference list),
             ocr (bool, font sizes are tesseract estimates -> text rules only),
             rules (y positions of horizontal rules; a rule above the bottom
             half starts the footnote block).
    """
    extra = extra or {}
    ocr = extra.get("ocr", False)
    refs = extra.get("refs", False)
    W, H = rect.width, rect.height
    texty = [p for p in paras if len(p.text) > 50
             and body_size * 0.92 <= p.size <= body_size * 1.12]
    body_top = min([p.rect.y0 for p in texty] or [0])
    body_bot = max([p.rect.y1 for p in texty] or [H])

    rules = [r for r in extra.get("rules", []) if r[0] > 0.45 * H
             and not any(f.y0 - 3 <= r[0] <= f.y1 + 3 and r[1] < f.x1 and r[2] > f.x0
                         for f in figs)]
    cols = extra.get("cols") or []

    def col_of(p):
        cx = (p.rect.x0 + p.rect.x1) / 2
        for i, (a, b) in enumerate(cols):
            if a - 10 <= cx <= b + 10:
                return i
        return -1

    # where the footnote block starts, per column
    note_top = {}
    for p in paras:
        if NOTE_MARK_RE.match(p.text) and p.rect.y0 > 0.30 * H and \
                not any((p.rect & pymupdf.Rect(f.x0 - 6, f.y0 - 6, f.x1 + 6, f.y1 + 6)).get_area() > 0
                        for f in figs):
            c = col_of(p)
            note_top[c] = min(note_top.get(c, 1e9), p.rect.y0)
    for y, rx0, rx1 in rules:
        for i, (a, b) in enumerate(cols):
            if min(rx1, b) - max(rx0, a) > 0.30 * (b - a):
                note_top[i] = min(note_top.get(i, 1e9), y)
    if not cols:
        note_top[-1] = min([v for v in note_top.values()] or [1e9])

    for p in paras:
        t = p.text
        p.kind = "body"
        p.reason = ""
        w = words(t)

        def keep(kind, why):
            p.kind, p.reason = kind, why

        if is_garbage(t):
            keep("garbage", "garbage"); continue
        if len(t) < 2:
            keep("garbage", "short"); continue
        if BOILER_RE.search(t):
            keep("footer", "publisher boilerplate"); continue

        if refs and (extra.get("pno", -1) > refs[0] or
                     (extra.get("pno", -1) == refs[0] and p.order >= refs[1])):
            keep("ref", "reference section"); continue

        if CAPTION_RE.match(t) and len(t) < 700 and not TABLE_SENTENCE_RE.match(t):
            keep("caption", "caption"); continue
        if len(t) < 500 and any(
                (p.rect & pymupdf.Rect(f.x0 - 30, f.y0 - 45, f.x1 + 30, f.y1 + 45)).get_area() > 0
                for f in figs):
            cm = re.search(r"(?i)\b(fi[gcl]|f[il1][cg]|figures?|tables?|chart)\s*\.?\s*[-–—:]?\s*[0-9IVX]",
                           t[:80])
            if cm and not re.search(r"[A-Za-z]{3,}", t[:cm.start()]):
                keep("caption", "caption near figure"); continue

        if NOTE_MARK_RE.match(t) and p.rect.y0 > 0.35 * H:
            keep("note", "marker"); continue
        if p.rect.y0 >= note_top.get(col_of(p), 1e9) - 2:
            keep("note", "below rule"); continue

        # A footnote that began on the previous page continues here without a
        # number, so nothing marks it; but it is still set in the small note
        # type, starts in the middle of a word, and sits below the body block.
        if small_type(p, body_size) and p.size <= body_size * 0.88 \
                and p.rect.y0 > 0.45 * H and len(t) >= 60 \
                and re.match(r"[a-z,;]", t) and not equation_like(p, body_size):
            keep("note", "continuation"); p.cont = True; continue

        if equation_like(p, body_size):
            keep("equation", "math"); continue

        if not ocr and small_type(p, body_size):
            if p.rect.y0 > 0.60 * H or p.rect.y0 >= body_bot - 4:
                keep("note", "small type"); continue

        if p.rect.y1 <= body_top + 2 and p.rect.y0 < 0.12 * H and len(t) < 160:
            keep("header", "above body"); continue
        if p.rect.y0 >= body_bot - 2 and not ocr:
            keep("footer", "below body"); continue

        # heading (font sizes are trustworthy on vector pages only)
        if (not ocr) and p.size >= body_size * 1.22 and len(t) < 160 \
                and not t.rstrip().endswith("."):
            keep("heading", "large font"); continue
        if (p.bold or t.isupper()) and len(t) < 120 and not t.endswith(".") \
                and len(w) <= 12 and (HEAD_RE.match(t) or t.isupper()):
            keep("heading", "bold/numbered"); continue
        if HEAD_RE.match(t) and len(t) < 90 and len(w) <= 10 \
                and not t.rstrip().endswith("."):
            keep("heading", "numbered short"); continue

        inside = False
        for f in figs:
            inter = (p.rect & f).get_area()
            if inter > 0.55 * p.rect.get_area():
                keep("figlabel", "inside figure")
                inside = True
                break
            g = pymupdf.Rect(f.x0 - 10, f.y0 - 10, f.x1 + 10, f.y1 + 10)
            if len(t) <= 60 and (p.rect & g).get_area() > 0.5 * p.rect.get_area():
                keep("figlabel", "beside figure")
                inside = True
                break
        if inside:
            continue

        if re.fullmatch(r"[\(\[]?\s*[0-9]{1,3}[a-z]?\s*[\)\]]?", t):
            keep("equation", "number"); continue

    # A continuation only counts when it really is the tail of the column: it
    # must sit below every body paragraph and above a numbered note.
    for p in paras:
        if not getattr(p, "cont", False):
            continue
        col = col_of(p)
        below_body = [q for q in paras
                      if q is not p and q.kind == "body" and col_of(q) == col
                      and q.rect.y0 > p.rect.y1 - 4]
        below_note = [q for q in paras
                      if q.kind == "note" and q.reason == "marker" and col_of(q) == col
                      and q.rect.y0 > p.rect.y0]
        if below_body or not below_note:
            p.kind, p.reason = "body", ""
    return paras


def equation_like(p, body_size):
    t = p.text
    w = [x.lower() for x in words(t)]
    if not w:
        return sym_ratio(t) > 0.05 or len(re.sub(r"[^0-9]", "", t)) > 2
    nstop = sum(1 for x in w if x in STOP)
    if nstop >= 2 and len(w) >= 5:
        return False
    if t.rstrip().endswith((".", "?", "!")) and nstop >= 1 and len(w) >= 4:
        return False
    sr = sym_ratio(t)
    if p.math > 0.5 and sr >= 0.02 and nstop == 0:
        return True
    if sr >= 0.16:
        return True
    if len(w) <= 5 and (sr >= 0.05 or any(c in t for c in "=<>\u2264\u2265")):
        return True
    if len(w) <= 3 and sr >= 0.03:
        return True
    if nstop == 0 and len(w) <= 6 and sr >= 0.08:
        return True
    return False


def small_type(p, body_size):
    return p.size <= body_size * 0.93


def is_note(p, body_size, body_bot, H):
    t = p.text
    small = p.size <= body_size * 0.93
    if not small:
        return False
    if p.rect.y0 < 0.55 * H:
        return False
    if NOTE_MARK_RE.match(t):
        return True
    return p.rect.y0 >= body_bot - 4


ENTRY = re.compile(r"^\s*[A-Z][A-Za-z'\u2019\-]+,\s*(?:[A-Z]\.|[A-Z][a-z])")
YEAR = re.compile(r"\b(?:1[89]|20)\d{2}\b")


def _ref_entry(p):
    """A paragraph that reads like a bibliography entry (author, initials,
    year) even when the OCR ran the whole list into one block."""
    return has_bibliography(p) or (ENTRY.match(p.text) is not None
                                   and len(YEAR.findall(p.text)) >= 2)


def detect_refs_start(models):
    """(absolute page number, paragraph order) of the reference-list heading."""
    for m in models:
        for p in m.paras:
            t = p.text.strip()
            if re.fullmatch(r"(?i)(references?|bibliography|literature cited|works cited|"
                            r"references and notes|reference list)\.?", t) and len(t) < 40:
                return (m.pno, p.order)
            # tesseract often glues the heading onto the first entry
            # ("References Barone-Adesi, G., and R. E. Whalley, 1987, ...")
            if re.match(r"(?i)(references?|bibliography|literature cited|works cited)\b",
                        t):
                return (m.pno, p.order)
    # No heading survived the scan: take the tail pages that are nothing but
    # bibliography entries, as long as they run to the end of the paper.
    live = [m for m in models if any(p.kind in ("body", "heading") for p in m.paras)]
    start = None
    for m in reversed(live):
        if m.pno < 0.55 * max(1, len(live)):
            break
        ps = [p for p in m.paras if p.kind in ("body", "heading")]
        if len(ps) < 2:
            break
        if sum(1 for p in ps if _ref_entry(p)) < 0.5 * len(ps):
            break
        start = (m.pno, min(p.order for p in ps))
    return start


def has_bibliography(p, body_size=None):
    """True only for a *pure* reference entry (no running prose): such a note or
    paragraph stays in English. Notes that mix prose and citations are
    translated, with the cited titles kept in English by the translator."""
    t = p.text
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", t)]
    if not words:
        return True
    nstop = sum(1 for w in words if w in STOP)
    if nstop >= 3:
        return False
    years = re.findall(r"\b(?:1[89]|20)\d{2}\b", t)
    quoted = re.findall(r'"[^"]{2,}"|\u201c[^\u201d]{2,}\u201d', t)
    if quoted and (years or nstop <= 2):
        return True
    if years and nstop <= 2:
        return True
    return False
