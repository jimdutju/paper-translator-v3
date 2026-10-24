"""Geometry from the PDF's own text layer -- positions only, never characters.

An OCR layer says the wrong thing in the right place (07: "dx(t) =[r+ upvldt+
V v(pdz,@, (14)"), and a born-digital layer says the right thing in the right
place.  Both are equally good at the only job here: where a line sits, where it
starts, how tall it is, whether it is rotated.  The characters are discarded on
purpose -- nothing downstream may read them, so an OCR error cannot leak into
the translation.

What each page gives:

    frame     the body's left and right edge (5th/95th percentile of line ends)
    pitch     median baseline-to-baseline distance
    size      median font size
    lines     [x0, y0, x1, y1, size, chars]  -- chars is a count, not the text
    images    the bitmap rectangles, exact (better than any detector)
    rules     horizontal/vertical rules, which is how a table or the note
              separator can be found without looking at the pixels

A scanned paper with no layer of its own is read through its _ocr.pdf twin
(papers.ocr_sibling), whose page grid is checked to match.
"""
import json
import os
import statistics
import sys

import pymupdf

import cache
import papers
import settings

MIN_LINES = 6.0            # below this a page has no usable layer


def _lines(page):
    out, rotated = [], 0
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if not txt:
                continue
            d = ln.get("dir", (1, 0))
            if abs(d[1]) > 0.2:                 # rotated: a watermark, a stamp
                rotated += 1
                continue
            x0, y0, x1, y1 = ln["bbox"]
            size = statistics.median([s["size"] for s in ln["spans"]]) or 0.0
            out.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1),
                        round(size, 2), len(txt)])
    out.sort(key=lambda r: (round(r[1]), r[0]))
    return out, rotated


def _page_geom(page):
    lines, rotated = _lines(page)
    if not lines:
        return {"nlines": 0, "rotated": rotated, "lines": [],
                "images": [], "rules": [], "frame": None, "pitch": None,
                "size": None}
    xs0 = sorted(r[0] for r in lines)
    xs1 = sorted(r[2] for r in lines)
    ys0 = sorted(r[1] for r in lines)
    ys1 = sorted(r[3] for r in lines)
    frame = [round(xs0[len(xs0) // 20], 1), round(ys0[len(ys0) // 20], 1),
             round(xs1[-max(1, len(xs1) // 20)], 1),
             round(ys1[-max(1, len(ys1) // 20)], 1)]
    centers = sorted((r[1] + r[3]) / 2.0 for r in lines)
    gaps = [b - a for a, b in zip(centers, centers[1:]) if 3 < b - a < 40]
    images = []
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            images.append([round(r.x0, 1), round(r.y0, 1),
                           round(r.x1, 1), round(r.y1, 1)])
    rules = []
    for dr in page.get_drawings():
        for it in dr.get("items", []):
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 0.8 and abs(a.x - b.x) > 20:
                    rules.append(["h", round(min(a.x, b.x), 1), round(a.y, 1),
                                  round(max(a.x, b.x), 1), round(a.y, 1)])
                elif abs(a.x - b.x) < 0.8 and abs(a.y - b.y) > 10:
                    rules.append(["v", round(a.x, 1), round(min(a.y, b.y), 1),
                                  round(a.x, 1), round(max(a.y, b.y), 1)])
            elif it[0] == "re":
                r = it[1]
                if 0 < r.height < 1.2 and r.width > 20:
                    rules.append(["h", round(r.x0, 1), round(r.y0, 1),
                                  round(r.x1, 1), round(r.y1, 1)])
    return {"nlines": len(lines), "rotated": rotated, "lines": lines,
            "images": images, "rules": rules[:60], "frame": frame,
            "pitch": round(statistics.median(gaps), 2) if gaps else None,
            "size": round(statistics.median(r[4] for r in lines), 2)}


def build(tag, verbose=True):
    src = papers.find_paper(tag)
    doc = pymupdf.open(src)
    own = [_page_geom(p) for p in doc]
    good = sum(1 for g in own if g["nlines"] >= MIN_LINES)
    used = src
    pages = own
    if good < 0.6 * len(doc):
        twin = papers.ocr_sibling(src)
        if twin:
            used = twin
            pages = [_page_geom(p) for p in pymupdf.open(twin)]
            good = sum(1 for g in pages if g["nlines"] >= MIN_LINES)
    name = os.path.basename(src)[:-4]
    out = {"tag": name, "source": settings.display_path(src, cache.ROOT),
           "geom_source": settings.display_path(used, cache.ROOT),
           "pages_total": len(pages), "pages_with_layer": good,
           "usable": good >= 0.6 * len(pages),
           "pages": {str(i + 1): g for i, g in enumerate(pages)}}
    os.makedirs(cache.GEOM, exist_ok=True)
    path = os.path.join(cache.GEOM, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    if verbose:
        print("%-22s pages %2d  layer %3d/%d  %s"
              % (name, len(pages), good, len(pages),
                 "usable" if out["usable"] else "NO LAYER"))
        print("   geometry from %s  (%.0f KB)"
              % (out["geom_source"], os.path.getsize(path) / 1024.0))
        for p in list(out["pages"])[:3]:
            g = out["pages"][p]
            print("   p%-2s lines %-3d pitch %-5s size %-5s frame %s"
                  % (p, g["nlines"], g["pitch"], g["size"], g["frame"]))
    return out


_MEM = {}


def load(name):
    """Geometry for a paper, or None when the paper has no usable layer."""
    if name in _MEM:
        return _MEM[name]
    path = os.path.join(cache.GEOM, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    g = g if g.get("usable") else None
    _MEM[name] = g
    return g


def page(g, n):
    return (g or {}).get("pages", {}).get(str(n))


def lines(g, n):
    p = page(g, n)
    if not p:
        return []
    return [pymupdf.Rect(r[0], r[1], r[2], r[3]) for r in p["lines"]]


def images(g, n, page_rect):
    """Bitmap rects that are not the page background.

    A scanned paper is one image the size of the page; that is the page, not
    a figure, and snapping a figure box to it would claim the whole sheet.
    """
    p = page(g, n)
    if not p:
        return []
    out = []
    full = page_rect.width * page_rect.height
    for r in p["images"]:
        q = pymupdf.Rect(*r)
        if q.get_area() > 0.6 * full:
            continue
        out.append(q)
    return out


def snap_figs(figs, imgs):
    """Give a figure box the bitmap's exact rectangle when one fits inside."""
    if not imgs:
        return figs
    used, out = set(), []
    for r in figs:
        rect = pymupdf.Rect(*r["rect"])
        best, bi = None, -1
        for i, im in enumerate(imgs):
            if i in used:
                continue
            cover = (rect & im).get_area() / max(1.0, im.get_area())
            if cover > 0.55 and (best is None or im.get_area() > best.get_area()):
                best, bi = im, i
        if best is not None:
            used.add(bi)
            out.append({"cls": r["cls"], "rect": [best.x0, best.y0,
                                               best.x1, best.y1]})
        else:
            out.append(r)
    return out


def snap_top(rect, g, n, window=120.0, avoid=()):
    """Move a block's top edge onto the nearest printed line.

    The vision model's y for a block can be tens of points out (07 p11: a
    caption box 84 pt above the caption it describes, which then failed to
    bind and landed on the picture).  The text layer knows where the line
    really is; only the top edge is corrected, the rest is left alone.
    """
    best = None
    for r in lines(g, n):
        if r.x1 < rect.x0 - 2 or r.x0 > rect.x1 + 2:
            continue
        if any((r & q).get_area() > 0.5 * r.get_area() for q in avoid):
            continue                       # text printed inside a picture: not it
        d = abs(r.y0 - rect.y0)
        if d <= window and (best is None or d < best[0]):
            best = (d, r)
    if best is None:
        return rect
    out = pymupdf.Rect(rect)
    out.y0 = best[1].y0
    return out


def display_hint(g, n, rect):
    """How the printed journal broke this display: line widths and starts.

    Only geometry: a display's lines are text-layer lines that sit inside the
    block the model called a formula.  Returns None when the layer says nothing
    useful, and every caller then behaves as if there were no text layer.
    """
    bands = display_lines(g, n, rect)
    if not bands or len(bands) > 5:
        return None
    if min(b.width for b in bands) < 12:
        return None                     # a stray fragment, not a printed line
    if any(b.width > rect.width + 40 for b in bands):
        return None
    widths = [b.width for b in bands]
    total = sum(widths) or 1.0
    # cumulative ink where the journal ended each line, as a fraction: the only
    # size-independent way to say "break here" to a formula set at another size
    cuts, acc = [], 0.0
    for w in widths[:-1]:
        acc += w
        cuts.append(round(acc / total, 3))
    return {"n": len(bands), "cuts": cuts,
            "widths": [round(w, 1) for w in widths],
            "starts": [round(b.x0, 1) for b in bands]}


def display_lines(g, n, rect):
    """The printed lines of one display, top to bottom.

    A display's lines sit inside the block the model marked as a formula;
    body lines can share the band when the box is loose, so a line has to be
    mostly inside the box to count.  Returned as (x0, y0, x1, y1).
    """
    out = []
    for r in lines(g, n):
        a = r.get_area()
        if a <= 0:
            continue
        if (r & rect).get_area() < 0.55 * a:
            continue
        out.append(r)
    out.sort(key=lambda r: (round(r.y1, 0), r.x0))
    # A printed line arrives as several fragments -- a superscript, the "cov"
    # before the bracket, the tail after it -- and they all sit on the same
    # baseline.  Cluster by the bottom edge, not the top: a superscript's top
    # is ten points above the baseline it belongs to.
    bands = []
    for r in out:
        for band in bands:
            if abs(band[0].y1 - r.y1) < 3.0:
                band.append(r)
                break
        else:
            bands.append([r])
    bands.sort(key=lambda b: min(r.y0 for r in b))
    res = []
    for b in bands:
        res.append(pymupdf.Rect(min(r.x0 for r in b), min(r.y0 for r in b),
                                max(r.x1 for r in b), max(r.y1 for r in b)))
    return res


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["08_abdl2003"]):
        build(t)
