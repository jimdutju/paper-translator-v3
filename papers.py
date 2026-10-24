"""Find the source PDF for a tag, and its OCR sibling if there is one."""
import glob
import os

import pymupdf

import cache


def find_paper(tag):
    """A path (absolute or relative) if that is what was given, else a tag."""
    if os.path.isfile(tag):
        return os.path.abspath(tag)
    if tag.lower().endswith(".pdf"):
        return os.path.join(cache.PAPERS, tag)
    hits = [f for f in glob.glob(os.path.join(cache.PAPERS, "*.pdf"))
            if os.path.basename(f).startswith(tag)]
    hits = [h for h in hits if "_ocr" not in h]
    if len(hits) != 1:
        raise SystemExit("ambiguous paper %r: %s" % (tag, hits))
    return hits[0]


def ocr_sibling(pdf):
    """The OCR'd copy of the same scan, if it exists and lines up.

    A scanned paper has a text layer that is only the library's watermark.  The
    companion _ocr.pdf is the same bitmap (07: 2550x3300 on both sides, same
    612x792 page) with an OCR layer on top, so its line boxes can be read as the
    geometry of the original.
    """
    stem, ext = os.path.splitext(pdf)
    cand = stem + "_ocr" + ext
    if not os.path.exists(cand):
        return None
    try:
        a, b = pymupdf.open(pdf), pymupdf.open(cand)
    except Exception:                        # noqa: BLE001
        return None
    if len(a) != len(b):
        return None
    for i in range(min(len(a), 8)):
        ra, rb = a[i].rect, b[i].rect
        if abs(ra.width - rb.width) > 2 or abs(ra.height - rb.height) > 2:
            return None
    return cand
