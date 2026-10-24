"""The one painter the engine needs (the rest of the old flow engine is not).

vlm_yolo calls pipeline._draw_block() to put a laid-out block on the page; that
is the only function this package ever used.  Copying it here keeps the new
version free of the old engine's model/content/recover modules, which are not
part of this pipeline at all.
"""
from __future__ import annotations

import pymupdf            # noqa: F401  (kept for parity with the old module)

import render


def _fonts(page):
    page.insert_font(fontname="zh", fontfile=render.FONT_R)
    page.insert_font(fontname="zhb", fontfile=render.FONT_B)
    page.insert_font(fontname="math", fontfile=render.FONT_M)
    page.insert_font(fontname="mathb", fontfile=render.FONT_MB)
    page.insert_font(fontname="mathi", fontfile=render.FONT_MI)
    page.insert_font(fontname="mathbi", fontfile=render.FONT_MBI)


def _draw_block(page, b, size, lead):
    sz = size * b.ratio
    y = b.y0 + sz * 0.12
    width = b.x1 - b.x0
    rows = []
    for ln in b.lines:
        w = render.line_width(ln, sz, b.bold)
        if w <= width + 0.5:
            rows.append(ln)
            continue
        # hard guarantee at the pen: nothing may cross into the next column
        cur, curw = [], 0.0
        for tok in ln:
            tw = render.tok_width(tok, sz, b.bold)
            if tw > width:
                if tok and tok[0] == "\ue000":     # a formula: cannot be split
                    if cur:
                        rows.append(cur)
                        cur, curw = [], 0.0
                    rows.append([tok])
                    curw = 0.0
                    continue
                # a single glued token: break it by character
                for ch in tok:
                    cw = render.tok_width(ch, sz, b.bold)
                    if cur and curw + cw > width:
                        rows.append(cur)
                        cur, curw = [], 0.0
                    cur.append(ch)
                    curw += cw
                continue
            if cur and curw + tw > width:
                rows.append(cur)
                cur, curw = [], 0.0
                if tok == " ":
                    continue
            cur.append(tok)
            curw += tw
        if cur:
            rows.append(cur)
    n = len(rows)
    center = (b.align == "center")
    ind = b.indent * sz if (b.indent and not b.continued) else 0.0
    for i, ln in enumerate(rows):
        base = y + sz * render.FONT["zh"].ascender
        w = render.line_width(ln, sz, b.bold)
        x_left = b.x0 + (ind if i == 0 else 0.0)
        room = width - (ind if i == 0 else 0.0)
        just, x = 0.0, x_left
        if center:
            x = x_left + (room - w) / 2.0
        elif (b.align == "justify" and i < n - 1
              and 0 < room - w < 0.30 * room):
            # only a block the original set as justified is stretched: a
            # left-aligned block keeps its ragged right edge
            just = room - w
        render.draw_line(page, x, base, ln, sz, b.bold, (0, 0, 0), just)
        y += sz * lead
