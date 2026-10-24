"""Typeset the displayed formulas instead of keeping the scanned bitmaps.

`eq_map.json` is  {"<page>": [{"rect": [x0,y0,x1,y1], "latex": "..."}]}.
Each entry is cleared of the scan and set as vector maths, centred in its box,
so every formula in the paper looks the same.
"""
import json
import os

import pymupdf
import render
import texmath

HERE = os.path.dirname(os.path.abspath(__file__))

# eq_map.json was typed by hand for ONE paper; applying it to another paper
# stamps that paper's formulas (and their boxes) onto the wrong pages, so the
# map is only used for the paper it was made for.
EQ_PAPER = "01_mandelbrot1963_original"

# space between the lines of one display, as a fraction of its own size.
# Measured off the source: pitch 19.5 pt for 9.8 pt maths, i.e. 1.99 em of
# baseline-to-baseline.  Our line image is about 1.25 em tall, so the blank
# between two of them is ~0.75 em.
LINE_GAP = 0.75


def _gap(size):
    return LINE_GAP * size


def load(name=None):
    # a map built for THIS paper (eq_vlm.py writes eq_map_<paper>.json) wins;
    # the hand-typed eq_map.json belongs to paper 01 alone
    if name:
        own = os.path.join(HERE, "eq_map_%s.json" % name)
        if os.path.exists(own):
            with open(own, encoding="utf-8") as f:
                rows = json.load(f)
            # an inline formula is left alone: its box is too small for a
            # display repaint, and the text flows around it as it always did
            out = {}
            for pno, items in rows.items():
                keep = [r for r in items
                        if r["rect"][3] - r["rect"][1] >= 18
                        and r["rect"][2] - r["rect"][0] >= 80]
                if keep:
                    out[pno] = keep
            return out
    if name is not None and name != EQ_PAPER:
        return {}
    path = os.path.join(HERE, "eq_map.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rows_for(page_no, name=None):
    return load(name).get(str(page_no)) or []


def split_ops(latex, soft=0):
    r"""Cut a formula before its top-level + - = operators.

    Only depth 0 counts: the minus inside (1 + 1/xi) or inside a fraction is
    part of the term, not a place where a line may break.

    The inside of ( ) and [ ] is closed to a break too, unless `soft` says how
    deep a parenthesis the cut may go: soft=1 breaks at the top-level + of a
    \left( ... \right) sum but never inside a term like (a - b_j v), which
    opens a second level.  LaTeX prints an unbalanced parenthesis without
    complaint, so those cuts need no repair, but \left/\right pairs do --
    _balance() closes and reopens them line by line.  Braces stay hard at any
    depth: { } is grouping, not type.
    """
    out, cur = [], ""
    brace, paren = 0, 0
    i, n = 0, len(latex)
    while i < n:
        c = latex[i]
        if c == "\\":
            j = i + 1
            while j < n and latex[j].isalpha():
                j += 1
            cur += latex[i:j]
            i = j
            continue
        if c == "{":
            brace += 1
        elif c == "}":
            brace = max(0, brace - 1)
        elif c in "([":
            paren += 1
        elif c in ")]":
            paren = max(0, paren - 1)
        # a break is allowed where a value has just ended -- "x - y", "]−[",
        # ") + (" -- and never where the sign is unary
        if (brace == 0 and paren <= soft and c in "+-=" and cur.strip()
                and (latex[i - 1] in " )]}" or latex[i - 1] == " "
                     or latex[i - 1].isalnum())):
            out.append(cur.strip())
            cur = ""
        cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out or [latex]


def _fit(latex, size, width):
    """Render one line at `size`; shrink only if a single term still too wide."""
    d = texmath.render(latex)
    sc = size / 12.0
    w = d["w"] * sc
    if w > width and d["w"] > 0:
        sc *= width / w
        floor = size / 12.0 * 0.85
        if sc < floor:
            sc = floor
    return d, sc, d["w"] * sc


def _count(ln, cmd):
    k, pos = 0, ln.find(cmd)
    while pos >= 0:
        k += 1
        pos = ln.find(cmd, pos + len(cmd))
    return k


def _balance(lines):
    r"""Close and reopen \left ... \right across the line breaks.

    A display broken inside \left( ... \right) does not compile line by
    line: the line that ends inside the group needs a \right. and the line
    that starts inside it needs a \left.  The pairs are counted, not matched
    to their delimiter -- \left. closes on whatever \right comes.
    """
    out, bal = [], 0
    for ln in lines:
        pre = "\\left. " * bal
        tail = bal + _count(ln, "\\left") - _count(ln, "\\right")
        post = " \\right." * max(0, tail)
        out.append(pre + ln + post)
        bal = max(0, tail)
    return out


def _eq_cols(latex, d):
    """Where the printed '=' and the character after it start, in points.

    Measured off the compiled PDF rather than guessed from advance widths:
    TeX's thick spaces around a relation are not part of any substring's
    width, and a two- or three-point error shows up as a crooked column.
    """
    try:
        doc = pymupdf.open(d["pdf"])
        chars = []
        for blk in doc[0].get_text("rawdict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln["spans"]:
                    for ch in sp["chars"]:
                        if ch["c"].strip():
                            chars.append((ch["c"], ch["bbox"][0] - d["x0"]))
        doc.close()
        for i, (c, x) in enumerate(chars):
            if c == "=":
                nxt = chars[i + 1][1] if i + 1 < len(chars) else x + 10.0
                return x, nxt
    except Exception:                     # noqa: BLE001
        pass
    return None


def _eq_pos(s):
    """Index of the first top-level '=' in a printed line, or -1."""
    depth, i = 0, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        elif c == "=" and depth == 0:
            return i
        i += 1
    return -1


def layout_formula(latex, size, width, number="", hint=None):
    """Lines of a display formula that fit `width`, and their total height.

    A formula too wide for the column is broken before an operator rather
    than squeezed: the column has the room, the line does not.  When `latex`
    is already a list, those are the printed lines and nothing is re-broken.

    `hint` is the printed journal's own line geometry for this display -- how
    many lines, how wide each one is, where each starts (geom.display_hint).
    It only moves the break points: the widths are scaled to this formula's own
    total width, so a reconstruction that sets wider than the original still
    breaks in the same proportion.  hint=None is exactly the old behaviour.
    """
    if isinstance(latex, list):
        # the number does not take width from the formula: it is placed
        # after the formula, wherever the formula ends (see draw())
        out, scs, h = [], [], 0.0
        for ln in latex:
            dd, ss, _ww = _fit(ln, size, width)
            out.append(ln)
            scs.append(ss)
            h += dd["h"] * ss
        return out, scs, h + _gap(size) * (len(out) - 1)
    d, sc, w = _fit(latex, size, width)
    if w <= width * 0.995:
        return [latex], [sc], d["h"] * sc
    pieces = split_ops(latex)
    if len(pieces) < 2:
        return [latex], [sc], d["h"] * sc
    # The hint's cuts are fractions of the display's own ink, not widths in
    # points: our LaTeX is set at 11.5 pt where the journal used 10, so the
    # same *width* would push the break one operator early.  The fraction says
    # "break 68% of the way through" and survives the difference in size.
    cuts = (hint or {}).get("cuts") or []
    hratio = 1.0

    def measure(txt, pre, wmax):
        # a line that starts or ends inside \left( ... cannot be measured,
        # let alone drawn, until the delimiter is closed the way it will be
        # closed on the page
        try:
            s = "\\left. " * max(0, pre) + txt
            return _fit(_balance([s])[0], size, wmax)[2]
        except Exception:                     # noqa: BLE001
            return 1e9

    def build(pcs, hang=0.0, tgts=None):
        # tgts is unused now; kept so old call sites do not have to change
        # a continuation line is set at the '=' of the first line, so it has
        # `hang` less room than the first one
        # balance of \left/\right before each piece: a line that starts at
        # piece i has to reopen the delimiters that are still open there
        pre, bal = [0] * len(pcs), 0
        for i, p in enumerate(pcs):
            pre[i] = bal
            bal = max(0, bal + _count(p, "\\left") - _count(p, "\\right"))

        def line_of(ix):
            # the \left. that reopens the group is added by _balance(), which
            # walks the lines in order and knows what is still open at each
            return " ".join(pcs[k] for k in ix)

        def room(idx):
            return width if idx == 0 else max(20.0, width - hang)

        def width_of(ix, wmax):
            return measure(" ".join(pcs[k] for k in ix), pre[ix[0]], wmax)

        groups, cur = [], [0]
        for i in range(1, len(pcs)):
            wmax = room(len(groups))
            if width_of(cur + [i], wmax) <= wmax * 0.995:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)

        def group_w(g):
            return width_of(g, room(groups.index(g)))

        # Orphans.  A line holding nothing but the head or nothing but the
        # tail reads as a typesetting error, so the short line borrows from
        # its neighbour.
        if len(groups) > 2 and group_w(groups[0]) < 0.25 * width:
            groups[0] = groups[0] + groups[1]
            del groups[1]
        while (len(groups) >= 2 and len(groups[-2]) > 1
               and group_w(groups[-1]) < 0.25 * width):
            groups[-1] = [groups[-2].pop()] + groups[-1]
        return _balance([line_of(g) for g in groups])

    def cut_groups(pcs):
        """Groups of pieces whose ends fall on the journal's own line ends."""
        if not cuts:
            return None
        pre = [0] * len(pcs)
        bal = 0
        for k, p in enumerate(pcs):
            pre[k] = bal
            bal = max(0, bal + _count(p, "\\left") - _count(p, "\\right"))

        def gw(g):
            return measure(" ".join(pcs[k] for k in g), pre[g[0]], width)

        cum, total = [], 0.0
        for k in range(len(pcs)):
            total = measure(" ".join(pcs[:k + 1]), 0, width)
            cum.append(total)
        if total <= 0:
            return None
        bounds, prev = [], 0
        for c in cuts:
            t = c * total
            idx = min(range(len(cum)), key=lambda k: abs(cum[k] - t))
            idx = max(idx, prev)
            if idx >= len(pcs) - 1:
                break
            bounds.append(idx)
            prev = idx + 1
        groups, start = [], 0
        for b in bounds:
            groups.append(list(range(start, b + 1)))
            start = b + 1
        groups.append(list(range(start, len(pcs))))
        groups = [g for g in groups if g]

        # A line the journal could fit but we cannot (our type is larger) must
        # not be left alone: "= b_0" on a line by itself reads as a mistake.
        # Borrow from a neighbour whenever the result still fits the column.
        k = 0
        while k < len(groups) and len(groups) > 1:
            if gw(groups[k]) < 0.25 * width:
                if k > 0 and gw(groups[k - 1] + groups[k]) <= width * 0.995:
                    groups[k - 1] = groups[k - 1] + groups[k]
                    del groups[k]
                    continue
                if k + 1 < len(groups) and gw(groups[k] + groups[k + 1]) <= width * 0.995:
                    groups[k] = groups[k] + groups[k + 1]
                    del groups[k + 1]
                    continue
            k += 1
        return groups

    def all_fit(ls, hang):
        for k, ln in enumerate(ls):
            wmax = width if k == 0 else max(20.0, width - hang)
            if _fit(ln, size, wmax)[2] > wmax * 0.995:
                return False
        return True

    cut_lines = None
    if cuts:
        g = cut_groups(pieces)
        if g:
            cut_lines = _balance([" ".join(pieces[k] for k in grp) for grp in g])
    lines = cut_lines or build(pieces)
    # the continuation lines hang at the '=' of the first line: lay them out
    # again with that indent taken off the room they have
    hang = 0.0
    if hint and len(hint.get("starts") or []) > 1:
        # where the printed continuation actually starts, relative to line 1
        h = (hint["starts"][1] - hint["starts"][0]) * hratio
        if 4.0 < h < 0.5 * width:
            hang = h
    if len(lines) > 1:
        i = _eq_pos(lines[0])
        if i > 0:
            try:
                sc0 = size / 12.0
                cols = _eq_cols(lines[0], texmath.render(lines[0]))
                if cols is not None:
                    # continuations hang under the character after the '='
                    hang = cols[1] * sc0
                else:
                    head = lines[0][:i].rstrip()
                    w_lhs = texmath.render(head)["w"] * sc0
                    w_eq = texmath.render("=")["w"] * sc0
                    w_bef = texmath.render(head + " =")["w"] * sc0
                    hang = w_bef + (w_bef - w_lhs - w_eq)
            except Exception:                 # noqa: BLE001
                hang = 0.0
    if hang > 0 and hang < 0.5 * width and not all_fit(lines, hang):
        alt = build(pieces, hang)
        if all_fit(alt, hang):
            lines = alt
    if not all_fit(lines, hang):
        alt = build(pieces, hang)
        if all_fit(alt, hang):
            lines = alt
    if not all_fit(lines, hang):
        # There are + and - inside the parentheses that could take the break:
        # use them before shrinking anything (07 p15 (A2) was squeezed to 82%
        # because the only cuts allowed were outside the brackets).
        for depth in (1, 2, 3):
            alt = build(split_ops(latex, soft=depth), hang)
            if all_fit(alt, hang):
                lines = alt
                break

    out, scs = [], []
    for k, ln in enumerate(lines):
        wmax = width if k == 0 else max(20.0, width - hang)
        _dd, ss, _ww = _fit(ln, size, wmax)
        out.append(ln)
        scs.append(ss)
    # one size for the whole display: a line set visibly smaller than the
    # lines above and below it is read as a mistake
    if len(scs) > 1:
        scs = [min(scs)] * len(scs)
    h = sum(texmath.render(ln)["h"] * ss for ln, ss in zip(out, scs))
    h += _gap(size) * (len(out) - 1)
    return out, scs, h


def height(latex, size, width, number="", hint=None):
    try:
        return layout_formula(latex, size, width, number, hint)[2]
    except Exception:                         # noqa: BLE001
        return size


def number(page, rect, val, size):
    """Set a printed equation number at the right edge of rect."""
    txt = "(%s)" % str(val).strip().strip("()")
    w = render.FONT["math"].text_length(txt, size)
    base = rect.y1 - size * 0.25
    try:
        render._tw_append(page, rect.x1 - w, base, txt, "math", size)
    except Exception:                          # noqa: BLE001
        page.insert_font(fontname="math", fontfile=render.FONT_M)
        page.insert_text((rect.x1 - w, base), txt, fontname="math",
                         fontsize=size, color=(0, 0, 0))


def draw(page, rect, latex, size, number="", hint=None):
    """Set a display formula in `rect`, broken at operators if too wide."""
    full = pymupdf.Rect(rect)
    numw = 0.0
    if number:
        numw = render.FONT["math"].text_length(
            "(%s)" % str(number).strip().strip("()"), size) + 6
    lines, scs, _total = layout_formula(latex, size, rect.width, number, hint)
    avail = rect.width
    indent = 1.2 * size if len(lines) > 1 else 0.0

    def _eqpos(s):
        """Index of the first top-level '=' in a printed line, or -1."""
        depth, i = 0, 0
        while i < len(s):
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c in "{([":
                depth += 1
            elif c in "})]":
                depth -= 1
            elif c == "=" and depth == 0:
                return i
            i += 1
        return -1

    # The lines of one display share an equals sign.  Two columns come out of
    # that line: where the '=' itself starts (eqx), and where the first
    # character after it starts (rhsx).  A line that begins with '=' goes to
    # eqx, a line that begins with '+' or '-' goes to rhsx -- i.e. under the
    # character following the '=', not under the '=' (08 p7 (5): under the E).
    ref, eqx, rhsx = None, 0.0, 0.0
    for k, ln in enumerate(lines):
        i = _eqpos(ln)
        if i > 0:
            try:
                cols = _eq_cols(ln, texmath.render(ln))
                if cols is not None:
                    eqx, rhsx = cols[0] * scs[k], cols[1] * scs[k]
                else:
                    head = ln[:i].rstrip()
                    w_lhs = texmath.render(head)["w"] * scs[k]
                    w_eq = texmath.render("=")["w"] * scs[k]
                    w_bef = texmath.render(head + " =")["w"] * scs[k]
                    eqx = w_bef - w_eq
                    rhsx = eqx + w_eq + (w_bef - w_lhs - w_eq)
                ref = k
                if k:
                    # that line is itself indented as a continuation, so its
                    # columns do not start at the margin
                    eqx += indent
                    rhsx += indent
                break
            except Exception:                 # noqa: BLE001
                pass
    y = rect.y0
    box = None
    base = y
    for k, ln in enumerate(lines):
        d = texmath.render(ln)
        sc = scs[k]
        w, h = d["w"] * sc, d["h"] * sc
        depth = d["depth"] * sc
        if len(lines) == 1:
            # centred in *its column*, not on the page: rect is the column's
            # rect, and the number lane only pushes the formula left when the
            # two would otherwise touch
            x0 = rect.x0 + max(0.0, (rect.width - w)) / 2.0
            x0 = max(x0, rect.x0)
        else:
            s = ln.lstrip()
            # the \left. that reopens a group is invisible but still text:
            # look past it to see which operator starts the line
            op = s
            while op.startswith("\\left."):
                op = op[len("\\left."):].lstrip()
            if ref is not None and k > ref and op[:1] == "=":
                x0 = rect.x0 + eqx          # under the '=' of the line above
            elif ref is not None and k > ref and op[:1] in "+-":
                x0 = rect.x0 + rhsx         # under the character after it
            elif ref is not None and k > ref and s.startswith("\\quad"):
                x0 = rect.x0 + eqx + 1.0 * size
            else:
                x0 = rect.x0 + (indent if k else 0.0)
        # the room a line actually has is from where it starts to the column
        # edge: an indented line has less, and is not pushed left to make up
        # for it (the number moves instead, see below)
        room = rect.x1 - x0
        if w > room and w > 0:
            sc *= room / w
            w, h = d["w"] * sc, d["h"] * sc
            depth = d["depth"] * sc
        if __import__("os").environ.get("EQ_DEBUG") and len(lines) > 1:
            print("  eq%d/%d ref=%s eqx=%.1f x0=%.1f w=%.1f | %s"
                  % (k, len(lines), ref, eqx, x0, w, ln[:34]))
        base = y + h - depth
        try:
            doc = render._doc_for(d["pdf"])
        except Exception:                     # noqa: BLE001
            # unreadable cache entry: set the formula once more, never print
            # its LaTeX source
            import texmath as _tm
            d = _tm.render(ln)
            doc = render._doc_for(d["pdf"])
        clip = pymupdf.Rect(d["x0"], d["y0"], d["x0"] + d["w"], d["y0"] + d["h"])
        b = pymupdf.Rect(x0, base - (h - depth), x0 + w, base + depth)
        page.show_pdf_page(b, doc, 0, clip=clip)
        if box is None:
            box = pymupdf.Rect(b)
        else:
            box.include_rect(b)
        y = base + depth + _gap(size)
    if box is None:
        box = pymupdf.Rect(rect)
    if number:
        # The number belongs to the formula: it sits at the right edge of the
        # column when the formula ends before that edge, and follows the
        # formula into the margin when the formula is wider.  It is never the
        # formula that gives way -- shrinking or sliding a display to make room
        # for a number reads as a botched setting (10 p11 (3.24)).
        txt = "(%s)" % str(number).strip().strip("()")
        w = render.FONT["math"].text_length(txt, size)
        nx = max(full.x1 - w, box.x1 + 6.0)
        nx = min(nx, page.rect.x1 - w - 2.0)
        nx = max(nx, full.x0)
        try:
            render._tw_append(page, nx, base, txt, "math", size)
        except Exception:                          # noqa: BLE001
            page.insert_font(fontname="math", fontfile=render.FONT_M)
            page.insert_text((nx, base), txt, fontname="math",
                             fontsize=size, color=(0, 0, 0))
        box.include_rect(pymupdf.Rect(nx, base - size, nx + w, base + size * 0.3))
    return box
