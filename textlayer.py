"""What kind of text layer does this PDF have?

v3 only trusts the *positions* in a text layer, never the characters, so the
question that matters is not "are the characters right" but "where did they
come from".  Three shapes:

  真文字       the glyphs are the ink (Tr=0), set by a typesetting engine.
               Positions exact, characters trustworthy.
  扫描+隐形层   the ink is a bitmap; the text objects are invisible (Tr=3) OCR
               annotations laid over it.  Positions approximate, characters
               unreliable -- formulas especially.
  纯图         no text objects at all.

Run it before geom.py to know what to expect, and before giving a file to
pdf2zh: pdf2zh only replaces text objects, so on an OCR-layer page its Chinese
lands on top of the scanned English.

    python textlayer.py                    # every paper under papers/
    python textlayer.py 04 07 11           # tags (prefix match, like run_paper)
"""
import collections
import glob
import os
import statistics
import sys
import unicodedata

import pymupdf

import cache

TRUE = "真文字"
OCR = "扫描+隐形层"
SCANVIS = "扫描+可见文字"
IMAGE = "纯图"
OTHER = "空白/矢量"

ADVICE = {
    TRUE: "pdf2zh 可跑；v3 的 P2/P3 全开",
    OCR: "pdf2zh 会叠印，别用；v3 只用几何、字符丢弃",
    SCANVIS: "层只是水印；要几何得靠 _ocr 兄弟件",
    IMAGE: "没有层；v3 退回 v2，除非有 _ocr 兄弟件",
    OTHER: "没有可用行框；按无层处理",
}


def advice_for(name, kind):
    if name.endswith("_ocr.pdf"):
        return "OCR 兄弟件：给 v3 供几何，不要直接翻译"
    return ADVICE[kind]


def page_kind(page):
    """(kind, visible spans, invisible spans, image coverage 0..1)."""
    tr = collections.Counter(t["type"] for t in page.get_texttrace())
    n0, n3 = tr.get(0, 0), tr.get(3, 0)
    area = abs(page.rect.width * page.rect.height) or 1.0
    cover = 0.0
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            cover += abs(r.width * r.height)
    cover = min(1.0, cover / area)          # strips add up, a page cannot exceed 1
    has = (n0 + n3) > 0
    if cover >= 0.5:
        if not has:
            return IMAGE, n0, n3, cover
        return (OCR if n3 > n0 else SCANVIS), n0, n3, cover
    return (TRUE if has else OTHER), n0, n3, cover


def is_math_font(name):
    n = name.lower()
    return any(k in n for k in ("cmmi", "cmsy", "cmex", "msam", "msbm", "math", "symbol"))


def analyze(path):
    doc = pymupdf.open(path)
    kinds = collections.Counter()
    chars = []
    fonts = set()
    for page in doc:
        kind, _, _, _ = page_kind(page)
        kinds[kind] += 1
        chars.append(len(page.get_text().strip()))
        for f in page.get_fonts():
            fonts.add(f[3])
    meta = doc.metadata or {}
    kind, n = kinds.most_common(1)[0]
    return {
        "pages": doc.page_count,
        "kind": kind,
        "share": n / doc.page_count,
        "chars": statistics.median(chars) if chars else 0,
        "fonts": len(fonts),
        "glyphless": sum("glyphless" in f.lower() for f in fonts),
        "math": sum(is_math_font(f) for f in fonts),
        "producer": (meta.get("producer") or "")[:26],
    }


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, w):
    return s + " " * max(0, w - width(s))


def resolve(tag):
    """A prefix -> the source paper; an explicit _ocr name or path stays as given."""
    if os.path.isfile(tag):
        return [os.path.abspath(tag)]
    hits = [f for f in glob.glob(os.path.join(cache.PAPERS, "*.pdf"))
            if os.path.basename(f).startswith(tag)]
    hits = sorted(hits, key=lambda f: "_ocr" in f, reverse=False)   # _ocr last
    plain = [h for h in hits if "_ocr" not in h]
    if plain:
        return plain
    if hits:
        return hits[:1]                       # asked for a sibling by name
    raise SystemExit("no paper matches %r" % tag)


def main(argv):
    files = []
    for tag in argv:
        files.extend(resolve(tag))
    if not argv:
        files = [f for f in sorted(glob.glob(os.path.join(cache.PAPERS, "*.pdf")))
                 if not os.path.basename(f).startswith("【")]   # outputs, not sources
    head = ["file", "pg", "kind", "share", "chars", "font", "math", "producer", "advice"]
    rows = []
    for path in files:
        try:
            a = analyze(path)
        except Exception as e:                    # noqa: BLE001
            rows.append([os.path.basename(path), "?", "ERR", str(e)[:40], "", "", "", "", ""])
            continue
        name = os.path.basename(path)
        rows.append([name[:34], str(a["pages"]), a["kind"],
                     "%.2f" % a["share"], "%.0f" % a["chars"], str(a["fonts"]),
                     str(a["math"]), a["producer"], advice_for(name, a["kind"])])
    w = [max(width(r[i]) for r in rows + [head]) for i in range(len(head))]
    print("  ".join(pad(h, w[i]) for i, h in enumerate(head[:-1])) + "  " + head[-1])
    for r in rows:
        print("  ".join(pad(c, w[i]) for i, c in enumerate(r[:-1])) + "  " + r[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))