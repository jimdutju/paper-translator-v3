"""Document flow layout: measure -> place -> verify.  No drawing, no IO.

The page is a set of columns interrupted by fixed objects (figures, tables,
captions, kept formulas, title, notes).  Text is a single stream that fills the
free intervals in reading order; a paragraph is cut at a line boundary and
continues in the next interval, so no text can ever be pushed into a fixed
object or into another paragraph.

Everything here is deterministic and measurable before the PDF is touched:
solve() returns placements plus a list of hard violations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import pymupdf

import render
import texmath

LEAD_MIN, LEAD_MAX = 1.22, 1.45
GAP_MIN, GAP_MAX = 0.30, 1.20          # paragraph gap, in lines
WIDOW = 2                              # never leave one line alone


@dataclass
class Run:
    """One unit of content in the reading stream.

    A run has no coordinates.  Where it ends up is decided by the cursor, not
    stored here -- that is the whole point: content and geometry are separate.
    """
    kind: str                 # text | heading | display | note | title
    zh: str = ""
    latex: str = ""
    ratio: float = 1.0        # font size relative to body size
    bold: bool = False
    align: str = "left"       # left | center, inherited from the printed page
    indent: float = 0.0       # first-line indent, in em
    keep_next: bool = False
    continued: bool = False   # the tail of a paragraph broken earlier
    src: list = field(default_factory=list)     # core.Para objects (originals)
    src_page: int = -1        # where it was printed (reading order, nothing else)
    src_rect: pymupdf.Rect | None = None        # where it was printed, to erase it
    number: str = ""          # printed equation number

    @property
    def label(self):
        if self.kind == "display":
            return "式:" + self.latex[:22]
        return (self.zh or "")[:22].replace("\n", " ")


@dataclass
class Block:
    """A run that has been placed.

    Everything geometric lives here and only here, and it exists only after
    solving.  The painter reads it; the solver writes it; content never sees it.
    """
    run: Run
    lines: list = field(default_factory=list)
    page: int = -1
    col: int = 0
    x0: float = 0.0
    x1: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    slot: int = -1
    gap: float = 0.0
    scale: float = 1.0        # display formulas may shrink a little to fit

    # -- content pass-through, so callers can keep saying b.zh
    kind = property(lambda self: self.run.kind)
    zh = property(lambda self: self.run.zh)
    latex = property(lambda self: self.run.latex)
    ratio = property(lambda self: self.run.ratio)
    bold = property(lambda self: self.run.bold)
    align = property(lambda self: self.run.align)
    keep_next = property(lambda self: self.run.keep_next)
    indent = property(lambda self: self.run.indent)
    continued = property(lambda self: self.run.continued)
    src = property(lambda self: self.run.src)
    src_page = property(lambda self: self.run.src_page)
    number = property(lambda self: self.run.number)
    label = property(lambda self: self.run.label)


@dataclass
class Frame:
    page: int
    idx: int                   # column index on that page
    x0: float
    x1: float
    top: float
    bottom: float
    obstacles: list = field(default_factory=list)


@dataclass
class Slot:
    page: int
    col: int
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def h(self):
        return self.y1 - self.y0


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
_WRAP = {}


def wrap_zh(zh, size, width, bold=False, indent=0.0):
    """Tokens of the wrapped lines, using the same wrapping as the painter."""
    k = (zh, round(size, 2), round(width, 1), bold, round(indent, 1))
    hit = _WRAP.get(k)
    if hit is not None:
        return hit
    render.MATH_MAX_W = width
    txt = render.prepare_math(zh) if "$" in zh else zh
    out = render.wrap(txt, size, width, bold,
                      first_width=(width - indent) if indent > 0 else None)
    _WRAP[k] = out
    return out


def line_text(line):
    return "".join(t for t in line if t != " ")


def display_height(latex, size):
    d = texmath.render(latex)
    return d["h"] * size / 12.0


# --------------------------------------------------------------------------- #
# frames and slots
# --------------------------------------------------------------------------- #
def covered_by(rect, others, share=0.5):
    a = rect.get_area() or 1.0
    return any((rect & o).get_area() > share * a for o in others)


def build_frames(m, eq_rects=(), note_top=None):
    """Columns of this page and the fixed objects that interrupt them."""
    eqs = [pymupdf.Rect(r) for r in eq_rects]
    obs = [pymupdf.Rect(f) for f in m.figs]
    for p in m.paras:
        r = pymupdf.Rect(p.rect)
        if p.kind in ("caption", "table", "figlabel", "name", "note", "title",
                      "header", "footer", "ref", "garbage", "skip"):
            obs.append(r)
        elif p.kind in ("body", "heading") and not getattr(p, "done", False):
            obs.append(r)                 # left in English: text must avoid it
        elif p.kind == "equation" and not getattr(p, "done", False)                 and not covered_by(r, eqs, 0.45):
            obs.append(r)
    # footnotes are re-flowed across BOTH columns, so reserve one strip over
    # the whole text width instead of each note's own (usually one-sided) rect
    notes = [p for p in m.paras if p.kind == "note"]
    if notes:
        # the notes are re-flowed across BOTH columns, so the reserved strip
        # must span the whole text width, not just the column they sat in
        body_all = [p for p in m.paras if p.kind in ("body", "heading")]
        nx0 = min([p.rect.x0 for p in body_all] + [p.rect.x0 for p in notes])
        nx1 = max([p.rect.x1 for p in body_all] + [p.rect.x1 for p in notes])
        ny1 = max(p.rect.y1 for p in notes)
        # the notes are re-flowed into two columns, so they need about half of
        # their original total height, not the scattered blocks the scan has
        if note_top is not None:
            ny0 = note_top
        else:
            need = sum(p.rect.height for p in notes) / 2.0 + 18.0
            ny0 = max(min(p.rect.y0 for p in notes), ny1 - need)
        obs = [o for o in obs if not any((o & pymupdf.Rect(p.rect)).get_area() > 0.9 * o.get_area()
                                         for p in notes)]
        obs.append(pymupdf.Rect(nx0, ny0, nx1, ny1))
    body = [p for p in m.paras if p.kind in ("body", "heading")]
    frames = []
    for ci, (x0, x1) in enumerate(m.cols):
        colp = [p for p in body if x0 - 6 <= (p.rect.x0 + p.rect.x1) / 2 <= x1 + 6]
        if not colp:
            continue
        top = min(p.rect.y0 for p in colp)
        bot = max(p.rect.y1 for p in colp)
        frames.append(Frame(m.pno, ci, x0, x1, top, bot, obs))
    return frames


def cut(f, min_gap=8.0):
    """Free intervals of a column, top to bottom."""
    cuts = []
    for o in f.obstacles:
        a, b = max(o.x0, f.x0), min(o.x1, f.x1)
        if b - a < 3:
            continue
        y0, y1 = max(o.y0, f.top), min(o.y1, f.bottom)
        if y1 - y0 >= 1:
            cuts.append([y0, y1])
    cuts.sort()
    merged = []
    for a, b in cuts:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, y = [], f.top
    for a, b in merged:
        if a - y >= min_gap:
            out.append((y, a))
        y = max(y, b)
    if f.bottom - y >= min_gap:
        out.append((y, f.bottom))
    return out


def slots_of(frames):
    out = []
    for f in frames:
        for a, b in cut(f):
            out.append(Slot(f.page, f.idx, f.x0, f.x1, a, b))
    return out


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
def _mk(run, lines, continued):
    """The same content, cut down to these lines."""
    return replace(run, zh="".join(line_text(l) for l in lines),
                   continued=continued)



def _fill(items, page, slots, size, lead, gap, col=None):
    """Fill the free intervals of one page; the rest is handed back."""
    pending = deque(items)
    placed = []
    for si, s in enumerate(slots):
        if page is not None and s.page != page:
            continue
        if col is not None and s.col != col:
            continue
        y = s.y0
        w = s.x1 - s.x0
        first = True
        while pending:
            r = pending[0]
            sz = size * r.ratio
            g = 0.0 if (first or r.continued) else gap * sz
            if r.kind == "display":
                g = gap * sz if not first else 0.0
                h = display_height(r.latex, sz) + sz * 0.35
                avail = s.y1 - (y + g)
                scale = 1.0
                if h > avail:
                    # rather than leave a hole, shrink the formula a little
                    if avail >= 0.85 * h:
                        scale = avail / h
                        h = avail
                    else:
                        break
                placed.append(Block(r, page=s.page, col=s.col, x0=s.x0, x1=s.x1,
                                    y0=y + g, y1=y + g + h, slot=si, gap=g,
                                    scale=scale))
                pending.popleft()
                y = placed[-1].y1
                first = False
                continue
            lines = wrap_zh(r.zh, sz, w, r.bold,
                            0.0 if r.continued else r.indent * sz)
            if not lines:
                pending.popleft()
                continue
            lh = sz * lead
            need = len(lines) * lh
            avail = s.y1 - y - g
            if r.keep_next and y + g + need + WIDOW * size * lead > s.y1:
                break
            nfit = int((avail + 1e-6) // lh)
            if nfit <= 0:
                break
            if nfit >= len(lines):
                placed.append(Block(r, lines=lines, page=s.page, col=s.col,
                                    x0=s.x0, x1=s.x1, y0=y + g, y1=y + g + need,
                                    slot=si, gap=g))
                pending.popleft()
                y = placed[-1].y1
                first = False
                continue
            if len(lines) - nfit == 1 and nfit > 1:
                nfit -= 1
            if nfit <= 0:
                break
            head = _mk(r, lines[:nfit], r.continued)
            tail = _mk(r, lines[nfit:], True)
            placed.append(Block(head, lines=lines[:nfit], page=s.page, col=s.col,
                                x0=s.x0, x1=s.x1, y0=y + g, y1=y + g + nfit * lh,
                                slot=si, gap=g))
            y = placed[-1].y1
            pending.popleft()
            pending.appendleft(tail)
            break
    return placed, list(pending)


def stretch(placed, slots, size, lead):
    """Give the spare room of every interval back to its paragraph gaps."""
    by_slot = {}
    for b in placed:
        by_slot.setdefault(b.slot, []).append(b)
    for si, bs in by_slot.items():
        bs.sort(key=lambda b: b.y0)
        spare = slots[si].y1 - bs[-1].y1
        breaks = [i for i, b in enumerate(bs) if i > 0 and not b.continued]
        if spare <= 1 or not breaks:
            continue
        add = min(spare / len(breaks), size * lead * (GAP_MAX - GAP_MIN))
        if add <= 0.2:
            continue
        shift = 0.0
        for i, b in enumerate(bs):
            if i > 0 and not b.continued:
                shift += add
            b.y0 += shift
            b.y1 += shift
    return placed



def solve_doc(blocks, frames, size, lead, gap=GAP_MIN):
    """One continuous stream over the whole document (default).

    Every page is filled to its last free point and the text runs on; with a
    size chosen so the stream ends on the last page, no page is left with a
    hole.  Returns (placed, leftover, slots).
    """
    slots = slots_of(frames)
    placed, rest = _fill(blocks, None, slots, size, lead, gap)
    placed = stretch(placed, slots, size, lead)
    return placed, rest, slots


def _used(placed):
    return sum(b.y1 - b.y0 for b in placed)


def _cap(slots, page, col):
    return sum(s.h for s in slots if s.page == page and s.col == col)


def _clone_blocks(items):
    """Fresh copies of the runs, so a trial pass cannot write into the real ones."""
    return [replace(r) for r in items]





def fill_page(items, page, slots, size, lead, gap):
    """Balance the two columns down to the line.

    Whole-block splits leave up to a paragraph of difference between the
    column feet; the block at the boundary is therefore tried line by line and
    the cut that brings both feet closest together wins.
    """
    cols = sorted({s.col for s in slots if s.page == page})
    if len(cols) < 2:
        return _fill(items, page, slots, size, lead, gap)
    colw = {c: max((s.x1 - s.x0) for s in slots if s.page == page and s.col == c)
            for c in cols}

    def score(a, b, rest):
        ya = max([x.y1 for x in a] + [0.0])
        yb = max([x.y1 for x in b] + [0.0])
        return abs(ya - yb) + 60.0 * len(rest)

    n = len(items)
    best = None
    for k in range(n + 1):
        cl = _clone_blocks(items)
        a, rest = _fill(cl[:k], page, slots, size, lead, gap, col=cols[0])
        b, rest2 = _fill(rest + cl[k:], page, slots, size, lead, gap, col=cols[1])
        sc = score(a, b, rest2)
        if best is None or sc < best[0] - 1e-6:
            best = (sc, k, 0)
        if k < n and cl[k].kind == "text":
            blk = cl[k]
            sz = size * blk.ratio
            lines = wrap_zh(blk.zh, sz, colw[cols[0]], blk.bold,
                            blk.indent * sz)
            if 2 <= len(lines) <= 40:
                for t in range(1, len(lines)):
                    head = _mk(blk, lines[:t], blk.continued)
                    tail = _mk(blk, lines[t:], True)
                    a2, r2 = _fill(cl[:k] + [head], page, slots, size, lead, gap,
                                   col=cols[0])
                    b2, r3 = _fill(r2 + [tail] + cl[k + 1:], page, slots, size,
                                   lead, gap, col=cols[1])
                    sc2 = score(a2, b2, r3)
                    if sc2 < best[0] - 1e-6:
                        best = (sc2, k, t)

    sc, k, t = best
    if not t:
        a, rest = _fill(items[:k], page, slots, size, lead, gap, col=cols[0])
        b, rest2 = _fill(rest + items[k:], page, slots, size, lead, gap, col=cols[1])
        return a + b, rest2
    blk = items[k]
    sz = size * blk.ratio
    lines = wrap_zh(blk.zh, sz, colw[cols[0]], blk.bold, blk.indent * sz)
    head = _mk(blk, lines[:t], blk.continued)
    tail = _mk(blk, lines[t:], True)
    tail.src = blk.src
    a, rest = _fill(items[:k] + [head], page, slots, size, lead, gap, col=cols[0])
    b, rest2 = _fill(rest + [tail] + items[k + 1:], page, slots, size, lead, gap,
                     col=cols[1])
    return a + b, rest2

def solve(blocks, frames, size, lead, gap=GAP_MIN, do_stretch=True):
    """Page-local flow: a page takes what it can hold, the rest goes on.

    Returns (placed, leftover, slots, spill) where spill is [(page, n)].
    """
    slots = slots_of(frames)
    by_page = {}
    for r in blocks:
        by_page.setdefault(r.src_page, []).append(r)
    page_order = sorted(set(by_page) | {f.page for f in frames})
    placed, spill = [], []
    carry = []
    for pg in page_order:
        items = carry + by_page.get(pg, [])
        if not items:
            continue
        got, rest = fill_page(items, pg, slots, size, lead, gap)
        placed.extend(got)
        if rest:
            spill.append((pg, len(rest)))
        carry = rest
    carry = _backfill(placed, carry, slots, size, lead, gap)
    if do_stretch:
        trial = _copy_blocks(placed)
        trial = stretch(trial, slots, size, lead)
        if not _collide(trial):
            placed = trial
    return placed, carry, slots, spill


def _copy_blocks(placed):
    """Same blocks, same geometry: the stretch works on a copy so it can be
    thrown away if it would push a block into its neighbour."""
    return [replace(b, run=replace(b.run)) for b in placed]


def _collide(placed):
    """True when two blocks on the same page touch each other."""
    for i in range(len(placed)):
        a = placed[i]
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if a.page != b.page:
                continue
            if b.y0 >= a.y1 - 0.5 or a.y0 >= b.y1 - 0.5:
                continue
            inter = (pymupdf.Rect(a.x0, a.y0, a.x1, a.y1) &
                     pymupdf.Rect(b.x0, b.y0, b.x1, b.y1))
            if inter.get_area() > 1.0:
                return True
    return False


def _backfill(placed, carry, slots, size, lead, gap):
    """Second chance for blocks that did not fit when their slot came up.

    A block is skipped when it does not fit the *current* slot, but an earlier
    slot may still have room (typically the last page of a paper).  Putting it
    there is the correct reading order, so try same-or-earlier pages.
    """
    if not carry:
        return carry
    used = {}
    for b in placed:
        used[b.slot] = max(used.get(b.slot, 0.0), b.y1)
    rest = []
    for r in carry:
        placed_now = False
        for si in range(len(slots) - 1, -1, -1):
            s = slots[si]
            if s.page > r.src_page:
                continue
            w = s.x1 - s.x0
            sz = size * r.ratio
            lines = []
            if r.kind == "display":
                h = display_height(r.latex, sz) + sz * 0.45
            else:
                lines = wrap_zh(r.zh, sz, w, r.bold)
                if not lines:
                    placed_now = True
                    break
                h = len(lines) * sz * lead
            top = used.get(si, s.y0)
            if s.y1 - top >= h + gap * sz:
                placed.append(Block(r, lines=lines, page=s.page, col=s.col,
                                    x0=s.x0, x1=s.x1, y0=s.y1 - h, y1=s.y1,
                                    slot=si, gap=gap * sz))
                used[si] = s.y1
                placed_now = True
                break
        if not placed_now:
            rest.append(r)
    return rest



def verify(placed, slots, frames):
    """Hard checks: no overlap, inside the slot, not over a fixed object."""
    bad = []
    by_page_obs = {}
    for f in frames:
        by_page_obs.setdefault(f.page, []).extend(f.obstacles)
    for b in placed:
        if b.slot < 0:
            bad.append(("unplaced", b.label))
            continue
        s = slots[b.slot]
        if b.x0 < s.x0 - 0.5 or b.x1 > s.x1 + 0.5:
            bad.append(("x-outside", b.label, round(b.x0), round(b.x1)))
        if b.y0 < s.y0 - 0.5 or b.y1 > s.y1 + 0.5:
            bad.append(("y-outside", b.label, round(b.y0), round(b.y1), round(s.y0), round(s.y1)))
        box = pymupdf.Rect(b.x0, b.y0, b.x1, b.y1)
        for o in by_page_obs.get(b.page, []):
            hit = box & o
            if hit.get_area() > 30.0 and hit.width > 3.0 and hit.height > 1.0:
                bad.append(("over-obstacle", b.page + 1, b.kind, b.label,
                            [round(v) for v in o]))
                break
    for i in range(len(placed)):
        a = placed[i]
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if a.page != b.page:
                continue
            if b.y0 >= a.y1 - 0.5 or a.y0 >= b.y1 - 0.5:
                continue
            inter = (pymupdf.Rect(a.x0, a.y0, a.x1, a.y1) &
                     pymupdf.Rect(b.x0, b.y0, b.x1, b.y1))
            if inter.get_area() > 1.0:
                bad.append(("overlap", a.label, b.label, round(inter.get_area(), 1)))
    return bad


def leftover_by_slot(placed, slots):
    """(slot index, unused points) for every slot that held something."""
    used = {}
    for b in placed:
        used[b.slot] = max(used.get(b.slot, 0.0), b.y1)
    out = []
    for i, s in enumerate(slots):
        if i in used:
            out.append((i, round(s.y1 - used[i], 1)))
    return out
