"""Page composition: wipe the English body text, typeset the Chinese in place.

The original page is kept as it is (figures, equations, captions, rules,
headers all untouched); only the paragraphs we translate are erased, then the
Chinese is flowed into the band those paragraphs occupied.  The band is filled
by choosing the largest possible body size first and only then by opening the
leading, which is what keeps the page from ending in a block of white space.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pymupdf
from pathlib import Path
import settings

LEAD = 1.22
PAINT_STATS = []
OVERFLOW = []   # lines that would leave their box   # one line pitch for the whole document

_win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_noto = Path("/usr/share/fonts/opentype/noto")
_dejavu = Path("/usr/share/fonts/truetype/dejavu")
FONT_R = settings.font_file("TR_FONT_REGULAR", _win / "simsun.ttc", _noto / "NotoSerifCJK-Regular.ttc")
FONT_B = settings.font_file("TR_FONT_BOLD", _win / "simhei.ttf", _noto / "NotoSerifCJK-Bold.ttc", FONT_R)
FONT_M = settings.font_file("TR_FONT_MATH", _win / "times.ttf", _dejavu / "DejaVuSerif.ttf")
FONT_MB = settings.font_file("TR_FONT_MATH_BOLD", _win / "timesbd.ttf", _dejavu / "DejaVuSerif-Bold.ttf")
FONT_MI = settings.font_file("TR_FONT_MATH_ITALIC", _win / "timesi.ttf", _dejavu / "DejaVuSerif-Italic.ttf")
FONT_MBI = settings.font_file("TR_FONT_MATH_BOLD_ITALIC", _win / "timesbi.ttf", _dejavu / "DejaVuSerif-BoldItalic.ttf")
FONT = {
    "zh": pymupdf.Font(fontfile=FONT_R),
    "zhb": pymupdf.Font(fontfile=FONT_B),
    "math": pymupdf.Font(fontfile=FONT_M),
    "mathb": pymupdf.Font(fontfile=FONT_MB),
    "mathi": pymupdf.Font(fontfile=FONT_MI),
    "mathbi": pymupdf.Font(fontfile=FONT_MBI),
}
NO_LEAD = set("，。、；：？！）】》”’%,.;:?!)]}")
NO_TAIL = set("（【《“‘([")
MATH_RE = re.compile(r"\$([^\$]+)\$")
MATH_TOK = re.compile(r"\ue000(\d+)\ue001")
MATH = {}


GLUE = re.compile(r"(?<=[\u0370-\u03ff])\s*(and|or)\s*(?=[\u0370-\u03ff])")


def _glue_fix(text):
    """τandβ -> τ 和 β: an English connective stranded between two symbols."""
    return GLUE.sub(lambda m: " 和 " if m.group(1) == "and" else " 或 ", text)


def prepare_math(text):
    """Replace $...$ with opaque tokens and compile the formulas."""
    import texmath

    def repl(m):
        latex = m.group(1).strip()
        try:
            d = texmath.render(latex)
        except Exception:                      # noqa: BLE001
            return " " + latex + " "
        k = str(len(MATH) + 1)
        MATH[k] = {"pdf": d["pdf"], "w": d["w"], "h": d["h"], "depth": d["depth"],
                   "latex": latex,
                   "crop": pymupdf.Rect(d["x0"], d["y0"],
                                        d["x0"] + d["w"], d["y0"] + d["h"])}
        return "\ue000" + k + "\ue001"

    return MATH_RE.sub(repl, text)


def _open_math(t, d):
    """Open a compiled formula, recompiling it if the cache entry went away.

    11 died on a _tex entry that existed as a path but could not be opened;
    the token only knows the path, so the LaTeX is kept in MATH as well and the
    formula is set once more rather than crashing the build.
    """
    try:
        return pymupdf.open(d["pdf"])
    except Exception as e:                     # noqa: BLE001
        import os as _os
        import sys as _sys
        p = d["pdf"]
        print("  !! open failed %r exists=%s size=%s err=%s"
              % (p, _os.path.exists(p),
                 _os.path.getsize(p) if _os.path.exists(p) else -1,
                 str(e)[:60]), file=_sys.stderr)
        import texmath
        d2 = texmath.render(d["latex"])
        MATH[t[1:-1]] = dict(d, pdf=d2["pdf"], w=d2["w"], h=d2["h"],
                             depth=d2["depth"],
                             crop=pymupdf.Rect(d2["x0"], d2["y0"],
                                               d2["x0"] + d2["w"],
                                               d2["y0"] + d2["h"]))
        return pymupdf.open(d2["pdf"])


def _is_math(t):
    return bool(t) and t[0] == "\ue000"


MATH_MAX_W = 1e9


def math_size(t, size):
    """(metrics, scale) for a formula at this type size, shrunk if it is wider
    than the column."""
    d = MATH[MATH_TOK.match(t).group(1)]
    sc = size / 12.0
    w = d["w"] * sc
    if w > MATH_MAX_W * 0.98:
        sc *= MATH_MAX_W * 0.98 / w
    return d, sc


TOKEN_RE = re.compile(
    r"\ue000\d+\ue001"
    r"|\u27e6[0-9A-Za-z*]{1,3}\u27e7"
    r"|[A-Za-z\u00c0-\u024f0-9][A-Za-z\u00c0-\u024f0-9'\u2019\-\./,]*"
    r"|[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]|.")


_GLYPH_ANY = {}

# glyphs that neither Times nor SimSun carries: draw the nearest sign instead of
# an empty box (≃ silently became a box in every formula it appeared in)
GLYPH_SUB = {"\u2243": "\u2248", "\u2245": "\u2248", "\u2213": "\u00b1",
             "\u2207": "\u25bd", "\u22c6": "*", "\u2219": "\u00b7",
             "\u2272": "\u2264", "\u2273": "\u2265", "\u227a": "\u003c",
             "\u227b": "\u003e", "\u2a7d": "\u2264", "\u2a7e": "\u2265"}


def _glyph_any(c):
    ok = _GLYPH_ANY.get(c)
    if ok is None:
        ok = FONT["math"].has_glyph(ord(c)) or FONT["zh"].has_glyph(ord(c))
        _GLYPH_ANY[c] = ok
    return ok


def _fix_glyphs(text):
    if not any(c in text for c in GLYPH_SUB):
        return text
    return "".join(GLYPH_SUB[c] if (c in GLYPH_SUB and not _glyph_any(c)) else c
                   for c in text)


def tokens(text):
    text = _fix_glyphs(text)
    out = []
    for m in TOKEN_RE.finditer(text):
        t = m.group(0)
        if t.isspace():
            out.append(" ")
        else:
            out.append(t)
    return out


def is_cjk(t):
    return bool(t) and "\u4e00" <= t[0] <= "\u9fff"


def _sup(t):
    return t.startswith("\u27e6") and t.endswith("\u27e7")


def _is_cjk(t):
    return any("\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
               or "\uff00" <= c <= "\uffef" for c in t)


def tok_font(t, bold=False):
    return FONT[tok_name(t, bold)]


ITALIC_WORDS = {"a", "an", "of", "in", "on", "to", "is", "or", "if", "at",
                "by", "as", "be", "we", "it", "the", "and", "for", "not", "are"}


def _italic(t):
    """Maths variables are set in italic: one letter always, two letters only
    when they are not a common English word ("eu" in "Q'(eu)" is a variable,
    "of" is not)."""
    core = t.rstrip("'\u2032\u2033")
    if not core.isalpha():
        return False
    if core.lower() in ITALIC_WORDS:
        return False
    return len(core) <= 2


_GLYPH_OK = {}


def _has_glyphs(name, t):
    """Does this font actually have every character of the token?

    has_glyph() is not cheap on a .ttc, so the answer is cached per character:
    a line of Chinese is 30 single-character tokens over a handful of glyphs.
    """
    f = FONT[name]
    for c in t:
        if c.isspace():
            continue
        k = (name, c)
        ok = _GLYPH_OK.get(k)
        if ok is None:
            ok = bool(f.has_glyph(ord(c)))
            _GLYPH_OK[k] = ok
        if not ok:
            return False
    return True


BOLD_TAG = "\u0001"          # marks one token as part of an inline bold run


def tok_name(t, bold=False):
    if t.startswith(BOLD_TAG):
        bold, t = True, t[len(BOLD_TAG):]
    if not t:
        return "zh"
    if _is_cjk(t):
        return "zhb" if bold else "zh"
    if _italic(t):
        name = "mathbi" if bold else "mathi"
    else:
        name = "mathb" if bold else "math"
    if not _has_glyphs(name, t):
        # Times has no ∈, ∅, ⋯ and friends: what it cannot draw, SimSun can
        zh = "zhb" if bold else "zh"
        if _has_glyphs(zh, t):
            return zh
    return name


def tok_width(t, size, bold=False):
    if t.startswith(BOLD_TAG):
        bold, t = True, t[len(BOLD_TAG):]
    if not t:
        return 0.0
    if _is_math(t):
        d, sc = math_size(t, size)
        return d["w"] * sc
    if _sup(t):
        return tok_font(t, bold).text_length(t[1:-1], size * 0.62)
    return tok_font(t, bold).text_length(t, size)


def wrap(text, size, width, bold=False, first_width=None):
    """Greedy CJK wrap that never splits a Latin word.

    Spaces count towards the line width (otherwise math/Latin-heavy lines are
    measured short and overrun the column), and closing punctuation is never
    left at the start of a line: when it would overflow, the character before
    it moves down with it instead of being forced onto the full line.
    """
    sp = tok_width(" ", size, bold)
    toks = tokens(text)
    lines, cur, curw = [], [], 0.0
    limit = first_width if first_width else width
    for t in toks:
        if t == " ":
            if cur and cur[-1] != " ":
                cur.append(t)
                curw += sp
            continue
        w = tok_width(t, size, bold)
        if w > limit and not cur:
            # one token wider than the column (glued words, a long formula):
            # break it by character, never let it cross into the next column
            for ch in t:
                cw = tok_width(ch, size, bold)
                if cur and curw + cw > width:
                    lines.append(cur)
                    cur, curw = [], 0.0
                cur.append(ch)
                curw += cw
            continue
        if cur and curw + w > limit + 0.01:
            if t and t[0] in NO_LEAD and len(cur) >= 2 and cur[-1] != " ":
                last = cur.pop()
                lastw = tok_width(last, size, bold)
                while cur and cur[-1] == " ":
                    cur.pop()
                    curw -= sp
                lines.append(cur)
                limit = width
                cur, curw = [last, t], lastw + w
            else:
                while cur and cur[-1] == " ":
                    cur.pop()
                    curw -= sp
                lines.append(cur)
                limit = width
                cur, curw = [t], w
        else:
            cur.append(t)
            curw += w
    if cur:
        while cur and cur[-1] == " ":
            cur.pop()
        if cur:
            lines.append(cur)
    return lines


def line_width(line, size, bold=False):
    tot, cur = 0.0, bold
    for t in line:
        if t == BOLD_TAG:
            cur = not cur
            continue
        tot += tok_width(t, size, cur)
    return tot


_DOCS = {}
_DOC_ORDER = []
DOC_CAP = 96          # open formula documents kept at once; see _remember()


def _doc_for(path):
    """One open document per compiled formula file.

    Keyed by path, not by token: half a page's inline maths can point at the
    same \"i\", and opening it once per occurrence is what exhausted MuPDF's
    file handles on 11.  On a failed open the bytes are read and handed over as
    a stream -- a sharing glitch then costs a read, not the build.
    """
    doc = _DOCS.get(path)
    if doc is None:
        try:
            doc = pymupdf.open(path)
        except Exception:                      # noqa: BLE001
            doc = pymupdf.open(stream=open(path, "rb").read(), filetype="pdf")
        _remember(path, doc)
    return doc


def _remember(t, doc):
    _DOCS[t] = doc
    _DOC_ORDER.append(t)
    while len(_DOC_ORDER) > DOC_CAP:
        old = _DOC_ORDER.pop(0)
        d = _DOCS.pop(old, None)
        if d is not None:
            try:
                d.close()
            except Exception:              # noqa: BLE001
                pass
    return doc

import threading
_LOCAL = threading.local()


def begin_page(page):
    """One content-stream write per page instead of one per line.

    insert_text re-scans the page's q/Q balance on every call, so a 600-line
    page costs seconds; the writer accumulates and flushes once.
    """
    _LOCAL.tw = pymupdf.TextWriter(page.rect)
    _LOCAL.page = page


def end_page(page):
    tw = getattr(_LOCAL, "tw", None)
    if tw is not None and getattr(_LOCAL, "page", None) is page:
        # NOT page.write_text(rect, writers=[tw]): that scales the writer's
        # content to fit rect (12 pt text came out 35.4 pt and overlapped)
        tw.write_text(page)
    _LOCAL.tw = None
    _LOCAL.page = None


def _tw_append(page, x, y, text, fontname, size):
    if not text:
        return
    if getattr(_LOCAL, "tw", None) is None or getattr(_LOCAL, "page", None) is not page:
        page.insert_text((x, y), text, fontname=fontname, fontsize=size)
        return
    _LOCAL.tw.append((x, y), text, font=FONT.get(fontname) or FONT["zh"], fontsize=size)


def draw_line(page, x, y, line, size, bold=False, color=(0, 0, 0), justify=0.0):
    """Justify by distributing `justify` points over the gaps of the line."""
    sp = tok_width(" ", size, bold)
    seq = []
    cur = bold
    for t in line:
        if t == BOLD_TAG:               # a zero-width toggle, not a glyph
            cur = not cur
            continue
        if t == " ":
            seq.append((" ", sp, cur))
        elif t:
            seq.append((t, tok_width(t, size, cur), cur))
    drawn = [i for i, it in enumerate(seq) if it[0] != " "]
    if not drawn:
        return
    extra = justify / max(1, len(drawn) - 1) if justify > 0 else 0.0
    cx = x
    k = 0
    i = 0
    n = len(seq)
    while i < n:
        t, w, b = seq[i]
        if t == " ":
            cx += w
            i += 1
            continue
        if _is_math(t):
            d, sc = math_size(t, size)
            doc = _doc_for(d["pdf"])
            # sit the ink bottom on the baseline: an extra 0.14*size
            # offset raised every formula ~1.5 pt, so supscripts and
            # subscripts floated above the text line
            top = y - (d["h"] - d["depth"]) * sc
            page.show_pdf_page(
                pymupdf.Rect(cx, top, cx + d["w"] * sc, top + d["h"] * sc),
                doc, 0, clip=d["crop"])
            cx += w
            k += 1
            if k < len(drawn):
                cx += extra
            i += 1
            continue
        if _sup(t):
            _tw_append(page, cx, y - size * 0.34, t[1:-1],
                       tok_name(t[1:-1], b), size * 0.62)
            cx += w
            k += 1
            if k < len(drawn):
                cx += extra
            i += 1
            continue
        # one insert_text per run of tokens that share a font and need no extra
        # spacing: a Chinese line is 30 single characters, and 30 calls to
        # insert_text cost ~0.5 s (measured) against ~0.02 s for one run
        fn = tok_name(t, b)
        buf = [t]
        adv = w
        k += 1
        j = i + 1
        while j < n and extra == 0.0:
            t2, w2, b2 = seq[j]
            if t2 == " ":
                buf.append(" ")
                adv += w2
                j += 1
                continue
            if _is_math(t2) or _sup(t2) or b2 != b or tok_name(t2, b2) != fn:
                break
            buf.append(t2)
            adv += w2
            k += 1
            j += 1
        _tw_append(page, cx, y, "".join(buf).replace(BOLD_TAG, ""), fn, size)
        cx += adv
        if k < len(drawn):
            cx += extra
        i = j


def _subtract(rect, blockers):
    """Parts of `rect` not covered by any blocker rect."""
    parts = [pymupdf.Rect(rect)]
    for b in blockers:
        out = []
        for r in parts:
            if (r & b).get_area() <= 1:
                out.append(r)
                continue
            if b.y0 > r.y0 + 0.5:
                out.append(pymupdf.Rect(r.x0, r.y0, r.x1, min(b.y0, r.y1)))
            if b.y1 < r.y1 - 0.5:
                out.append(pymupdf.Rect(r.x0, max(b.y1, r.y0), r.x1, r.y1))
            ya, yb = max(r.y0, b.y0), min(r.y1, b.y1)
            if b.x0 > r.x0 + 0.5:
                out.append(pymupdf.Rect(r.x0, ya, min(b.x0, r.x1), yb))
            if b.x1 < r.x1 - 0.5:
                out.append(pymupdf.Rect(max(b.x1, r.x0), ya, r.x1, yb))
        parts = out
    return [p for p in parts if p.width > 1 and p.height > 0.5]


class PageWriter:
    def __init__(self, out_doc, src_doc):
        self.out = out_doc
        self.src = src_doc

    # -- helpers ------------------------------------------------------------
    def new_page(self, m):
        page = self.out.new_page(width=m.rect.width, height=m.rect.height)
        return page

    def wipe_scan(self, page, paras, s=3.0, blockers=()):
        """White rectangles exactly over the ink of the paragraphs we replace."""
        if not paras:
            return
        rects = [pymupdf.Rect(p.rect) for p in paras]
        for r in rects:
            for q in _subtract(r, blockers):
                q = q & page.rect
                if q.width > 1 and q.height > 0.5:
                    page.add_redact_annot(q, fill=(1, 1, 1))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        for r in rects:
            for q in _subtract(r, blockers):
                q = q & page.rect
                if q.width > 1 and q.height > 0.5:
                    page.draw_rect(q, color=None, fill=(1, 1, 1), width=0)
        return rects

    def wipe_vector(self, page, paras, blockers=()):
        for p in paras:
            for r in _subtract(p.rect, blockers):
                if r.width > 1 and r.height > 0.5:
                    page.add_redact_annot(r)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)

    # -- text layout --------------------------------------------------------
    def layout(self, page, rect, items, body_pt, gap_ratio=0.42, size=None,
               lead_floor=1.08, size_floor=0.92):
        """items: [{'zh','kind','ratio'}] -> typeset inside rect, return used height."""
        if not items or rect.height < 4:
            return 0.0
        for it in items:
            it["zh"] = _glue_fix(it["zh"])
        if size is None:
            size = self.fit(page, rect, items, body_pt, gap_ratio)
        return self.paint(page, rect, items, body_pt, size, gap_ratio,
                          lead_floor, size_floor)

    def fit(self, page, rect, items, body_pt, gap_ratio):
        lo, hi = body_pt * 0.80, body_pt * 1.55
        best = lo
        for _ in range(14):
            mid = (lo + hi) / 2
            if self.height(items, rect.width, body_pt, mid, gap_ratio) <= rect.height:
                best = mid
                lo = mid
            else:
                hi = mid
        return best

    def height(self, items, width, body_pt, size, gap_ratio):
        total = 0.0
        for i, it in enumerate(items):
            sz = size * it.get("ratio", 1.0)
            n = len(wrap(it["zh"], sz, width, it["kind"] != "body"))
            total += n * sz * 1.22
            total += gap_ratio * sz
        return total

    def paint(self, page, rect, items, body_pt, size, gap_ratio,
              lead_floor=1.08, size_floor=0.92):
        page.insert_font(fontname="zh", fontfile=FONT_R)
        page.insert_font(fontname="zhb", fontfile=FONT_B)
        page.insert_font(fontname="math", fontfile=FONT_M)
        page.insert_font(fontname="mathb", fontfile=FONT_MB)
        page.insert_font(fontname="mathi", fontfile=FONT_MI)
        page.insert_font(fontname="mathbi", fontfile=FONT_MBI)

        global MATH_MAX_W
        MATH_MAX_W = rect.width

        def measure(lead_f, sz_size):
            rows, total = [], 0.0
            for it in items:
                if it.get("kind") == "display":
                    d = __import__("texmath").render(it["latex"])
                    dh = d["h"] * sz_size / 12.0
                    rows.append((it, sz_size, False, []))
                    total += dh + sz_size * 0.45
                    continue
                sz = sz_size * it.get("ratio", 1.0)
                bold = it["kind"] != "body"
                src = _glue_fix(it["zh"])
                zt = prepare_math(src) if "$" in src else src
                lines = wrap(zt, sz, rect.width, bold)
                rows.append((it, sz, bold, lines))
                total += len(lines) * sz * lead_f + gap_ratio * sz
            return rows, total

        # One line pitch for the whole document: only squeeze the pitch (never
        # the type size) when a band is too short, and never stretch it to fill.
        lead_f = LEAD
        rows, total = measure(lead_f, size)
        while total > rect.height * 1.005 and lead_f > lead_floor:
            lead_f -= 0.02
            rows, total = measure(lead_f, size)
        while total > rect.height * 1.005 and size > body_pt * size_floor:
            size -= 0.10
            rows, total = measure(lead_f, size)

        PAINT_STATS.append({"page": page.number + 1, "x0": round(rect.x0),
                            "y0": round(rect.y0), "y1": round(rect.y1),
                            "size": round(size, 2), "lead": round(lead_f, 3),
                            "total": round(total, 1), "height": round(rect.height, 1),
                            "n": len(rows), "para": items[0].get("zh", "")[:14]})
        extra = max(0.0, rect.height - total)
        ngaps = len(rows) or 1
        # slack goes into paragraph spacing only: line pitch and type size stay
        # identical across the document
        gap_add = min(extra * 0.92 / ngaps, size * 3.0)

        y = rect.y0 + size * 0.12
        for it, sz, bold, lines in rows:
            if it.get("kind") == "display":
                y = __import__("display").draw(page, rect, it["latex"], sz, y + sz * 0.1)
                y += gap_ratio * sz
                continue
            ink = sz * FONT["zhb" if bold else "zh"].ascender
            for i, ln in enumerate(lines):
                base = y + ink
                if base > rect.y1 + 0.4:
                    return rect.height * 3   # overflow: caller reports it
                w = line_width(ln, sz, bold)
                just = 0.0
                if w > rect.width + 0.5:          # never let a line leave the box
                    OVERFLOW.append((page.number + 1, round(rect.x1, 1),
                                     round(rect.x0 + w, 1), "".join(ln)[:40]))
                elif i < len(lines) - 1 and rect.width - w > 0 and rect.width - w < 0.30 * rect.width:
                    just = rect.width - w
                draw_line(page, rect.x0, base, ln, sz, bold, (0, 0, 0), just)
                y += sz * lead_f
            y += gap_ratio * sz + gap_add
        return min(rect.height, y - rect.y0)

    def title(self, page, m, zh, f_body):
        """Big, centred title on the first page."""
        t = [p for p in m.paras if p.kind == "title"]
        if not t or not zh:
            return
        p = t[0]
        span = [q for q in m.paras if q.kind in ("body", "heading", "note")]
        x0 = min([q.rect.x0 for q in span] or [m.rect.x0 + 40])
        x1 = max([q.rect.x1 for q in span] or [m.rect.x1 - 40])
        below = [q.rect.y0 for q in m.paras
                 if q.rect.y0 > p.rect.y0 + 4 and q.kind != "title"]
        y1 = min(below) - 3 if below else p.rect.y1 + 3.2 * f_body
        area = pymupdf.Rect(x0, p.rect.y0, x1, max(y1, p.rect.y0 + 1.5 * f_body))
        ratio = 1.45
        if m.body_size:
            ratio = min(2.8, max(1.25, p.size / m.body_size))
        size = min(30.0, f_body * ratio)
        while size > f_body * 1.02:
            lines = wrap(zh, size, area.width, True)
            if len(lines) * size * 1.3 <= area.height:
                break
            size -= 0.2
        lines = wrap(zh, size, area.width, True)
        # avoid an orphaned last character: shrink a little until the last
        # line carries a real share of the width
        if len(lines) > 1 and line_width(lines[-1], size, True) < 0.32 * area.width:
            for k in range(1, 14):
                s2 = size - 0.25 * k
                if s2 < f_body * 1.02:
                    break
                l2 = wrap(zh, s2, area.width, True)
                if len(l2) * s2 * 1.3 <= area.height and \
                        (len(l2) < len(lines) or
                         line_width(l2[-1], s2, True) >= 0.32 * area.width):
                    size, lines = s2, l2
                    break
        page.insert_font(fontname="zhb", fontfile=FONT_B)
        h = len(lines) * size * 1.3
        y = area.y0 + max(0.0, (area.height - h) / 2) + size * FONT["zhb"].ascender
        for ln in lines:
            w = line_width(ln, size, True)
            draw_line(page, (area.x0 + area.x1 - w) / 2, y, ln, size, True)
            y += size * 1.3

    def heading(self, page, rect, zh, size, center=False):
        page.insert_font(fontname="zhb", fontfile=FONT_B)
        size0 = size
        lines = wrap(zh, size, rect.width, True)
        while len(lines) > 1 and size > size0 * 0.72:
            size -= 0.2
            lines = wrap(zh, size, rect.width, True)
        if os.environ.get("V2DEBUG"):
            print("   heading %r rect=%s size=%.1f lines=%d" %
                  (zh[:20], [round(v, 1) for v in rect], size, len(lines)))
        # sit on the original baseline so the line below never collides
        y = rect.y1 - size * 0.22
        for ln in lines:
            w = line_width(ln, size, True)
            x = rect.x0 if not center else (rect.x0 + rect.x1 - w) / 2
            draw_line(page, x, y, ln, size, True)
            y += size * 1.32
