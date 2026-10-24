"""Geometric audit of a built PDF: no text may sit on a picture, and no two
text blocks may share the same strip.  The builder prevents both by booking
rects before it sets a line; this is the check that the booking worked.
"""
import sys

import pymupdf


def rects(page):
    out = []
    for b in page.get_text("blocks"):
        if b[6] != 0:            # 0 = text block, 1 = image block
            continue
        out.append(("text", pymupdf.Rect(b[:4]), (b[4] or "").strip()[:24]))
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            out.append(("image", pymupdf.Rect(r), ""))
    return out


def main(path, min_overlap=2.0):
    doc = pymupdf.open(path)
    bad = 0
    for i, page in enumerate(doc):
        items = rects(page)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ka, ra, ta = items[a]
                kb, rb, tb = items[b]
                if ka == "image" and kb == "image":
                    continue
                x = min(ra.x1, rb.x1) - max(ra.x0, rb.x0)
                y = min(ra.y1, rb.y1) - max(ra.y0, rb.y0)
                if x <= 0 or y <= 0:
                    continue
                if y < min_overlap:
                    continue
                bad += 1
                print("p%d %s/%s overlap %.1fx%.1fpt  %s | %s"
                      % (i + 1, ka, kb, x, y, ta, tb))
    print("overlaps: %d" % bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
