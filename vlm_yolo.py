"""VLM text on YOLO geometry.

The layout model says where the blocks are and what kind they are; the vision
model says what they say.  Nothing here reads the OCR text layer.
"""
from __future__ import annotations

import os
import re
import sys
import time

import pymupdf

import cache
import core
import geom
import equations
import layout
import layout_yolo
import pipeline
import render
import trans
import vlm_page
import quality
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
# each version writes into its own staging directory: a bad build must never
# overwrite the PDF that is currently published
OUT = cache.OUT

TEXT_ROLES = {"body", "abstract", "heading", "title", "reference", "authors",
              "affiliation"}
TEXT_CLS = {"plain text", "title"}
CAP_CLS = {"figure_caption", "table_caption"}
FIG_CLS = {"figure", "table"}
SUP = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
       "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079"}


def _bold_tags(zh):
    r"""**emphasised** runs become a pair of zero-width toggle tokens.

    A star inside $...$ is maths, not emphasis: \kappa^* and \theta^* are
    superscripts.  Eating the pair turned the formula into \kappa^[\theta^ ,
    which no longer compiles, and 07 p11 printed the raw LaTeX in a caption.
    """
    if "*" not in (zh or ""):
        return zh
    out, pos = [], 0
    for m in MATH_SPAN.finditer(zh):
        out.append(_bold_tags_plain(zh[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_bold_tags_plain(zh[pos:]))
    return "".join(out)


MATH_SPAN = re.compile(r"\$[^$]*\$")


def _bold_tags_plain(zh):
    if "*" not in zh:
        return zh
    zh = re.sub(r"\*\*(.+?)\*\*", lambda m: "\u0001" + m.group(1) + "\u0001", zh)
    # the translator also marks emphasis with a single pair; a lone star (a
    # footnote dagger at the head of a note) has no partner and is left alone
    return re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)",
                  lambda m: "\u0001" + m.group(1) + "\u0001", zh)


def _strip_overlap(prev, z):
    """Drop the words the model repeated when it split a sentence at a page.

    The repeat is rarely a clean prefix: "…以强调收益依赖于" is followed by
    "（强烈地）依赖于Δt。", so the tail of the previous block is searched in
    the head of this one and everything up to the end of that repeat drops.
    """
    z = (z or "").lstrip()
    tail = (prev or "").strip()
    if not tail or not z:
        return z
    # The duplicated fragment, when there is one, is at the very start: the
    # translator repeated the tail of the previous block before continuing.
    # A short match anywhere in the first sixteen characters is not evidence --
    # 19 lost a whole sentence because "可写为" (three characters) also ends
    # the block above it.
    for k in range(min(10, len(tail)), 4, -1):
        frag = tail[-k:]
        pos = z[:3].find(frag)
        if pos >= 0:
            cut = z[pos + k:].lstrip("，。、；： ）\u00a0")
            if cut:                  # never delete the whole block
                return cut
    return z
def _ends_sentence(s):
    t = (s or "").rstrip()
    if not t:
        return True
    if re.search(r"\u27e6M\d+\u27e7$", t):
        return False          # a formula hides the real ending
    return t[-1] in ".!?"


def _joinable(ra, a_en, rb, b_en, skipped=False):
    """May b continue the sentence a left unfinished?"""
    if ra not in ("body", "abstract") or rb not in ("body", "abstract"):
        return False
    if _ends_sentence(a_en):
        return False
    # a table or a picture may sit between the two halves: printing the table
    # at the top of the next page splits the paragraph without ending it.  That
    # is only allowed while the sentence is mid-clause, never after a lead-in
    # like "... are defined as:", where the next text follows the display.
    if skipped and (a_en or "").rstrip()[-1:] in ":;,、(（":
        return False
    n = (b_en or "").lstrip()
    if not n:
        return False
    return n[0].islower() or n.startswith("\u27e6M") or n[0] in ",;)"


def _merge_protected(a_en, a_keep, b_en, b_keep):
    """Join two protected texts, renumbering b's formula placeholders."""
    off = len(a_keep)
    b2 = re.sub(r"\u27e6M(\d+)\u27e7",
                lambda m: "\u27e6M%d\u27e7" % (int(m.group(1)) + off), b_en)
    a = (a_en or "").rstrip()
    if a.endswith("-"):
        return a[:-1] + b2.lstrip(), list(a_keep) + list(b_keep)
    return a + " " + b2.lstrip(), list(a_keep) + list(b_keep)


def _continues(a, b):
    """One sentence cut by a column or page break, written as two blocks."""
    if a["b"]["role"] not in ("body", "abstract"):
        return False
    if b["b"]["role"] not in ("body", "abstract"):
        return False
    t = a["en"].rstrip()
    if not t or t[-1] in ".!?:;":
        return False
    n = b["en"].lstrip()
    return bool(n) and (n[0].islower() or n[0] in ",;)")


def _split_zh(zh, ratio):
    """Cut a translation back where the English was cut -- at a comma."""
    n = len(zh)
    if n < 4:
        return zh, ""
    target = int(n * ratio)
    best = None
    lo = max(2, target - 40)
    hi = min(n - 2, target + 40)
    for i in range(lo, hi):
        if zh[i - 1] in "，。；：、！？ ":
            d = abs(i - target)
            if best is None or d < best[0]:
                best = (d, i)
    cut = best[1] if best else max(1, min(n - 1, target))
    if zh[:cut].count("$") % 2:
        j = zh.find("$", cut)
        if j > 0:
            cut = j + 1
    return zh[:cut].strip(), zh[cut:].strip()


def _latex(text):
    parts = [l.strip().strip("$").strip() for l in (text or "").split("\n") if l.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "\\begin{aligned}" + "\\\\".join(parts) + "\\end{aligned}"


def _norm(t):
    return re.sub(r"\s+", "", t or "").lower()


def _dedupe(rows):
    """Drop a block whose text the block above already carries.

    When a paragraph runs over the page break the model sometimes writes it
    twice: merged into the block above and again as the next page block.
    The second copy adds nothing and prints the paragraph twice, so it goes.
    """
    out = []
    for b in rows:
        t = _norm(b.get("text"))
        if len(t) > 40 and any(t in _norm(a.get("text")) for a in out[-3:]):
            continue
        out.append(b)
    return out


def _split_display(rows):
    """Break blocks at the lines that are nothing but a formula.

    The model already writes a display formula on a line of its own, even when
    it sits inside a note; the line break is the structural signal, so a note
    holding two inequalities becomes five blocks (text, formula, text, formula,
    text) instead of one paragraph with the maths buried in it.
    """
    out = []
    for b in rows:
        if b.get("role") == "caption":
            # a caption's line breaks are paragraph breaks, not separate
            # blocks.  Split, each piece is pinned to the same caption box and
            # the second and third land on their own guessed bboxes -- on top
            # of the picture and of each other (07 p11, Figure 2).
            out.append(b)
            continue
        text = (b.get("text") or "").strip()
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if len(parts) < 2:
            out.append(b)
            continue
        x0, y0, x1, y1 = b["bbox"]
        step = (float(y1) - float(y0)) / len(parts)
        # Symbol-definition lists are independent lines, not one paragraph.
        # Sending them through normal sentence joining destroys that structure.
        definitions = (settings.get("TR_DEFINITION_LINES", "1") != "0"
                       and all(re.match(r"^\$[^$]+\$\s*=\s*[A-Za-z]", p) for p in parts))
        # the printed number sits on one of the split lines (the first one,
        # as a rule): it must not be dropped with the split, or the line that
        # follows then borrows its neighbour number
        pending = (b.get("number") or "").strip()
        for k, p in enumerate(parts):
            nb = dict(b)
            nb["text"] = p
            nb["bbox"] = [x0, round(y0 + step * k, 1), x1,
                          round(y0 + step * (k + 1), 1)]
            if definitions:
                nb.update(role="body", align="left", _definition=True, _mid=True)
            if re.fullmatch(r"\$[^$]+\$", p) or re.fullmatch(r"\$\$[\s\S]+\$\$", p):
                nb["role"] = "formula"
                nb["number"] = pending
                pending = ""
                # the parts of one display share a group id so the setter can
                # stack them under a common equals sign instead of centring
                # every line on its own
                nb["_grp"] = (b["page"], round(float(y0), 1))
            out.append(nb)
    return out


def _ink_y(page, x0, y0, x1, y1, dpi=200):
    """Top of the actual ink inside a guessed band (page points)."""
    clip = pymupdf.Rect(x0, y0, x1, y1) & page.rect
    if clip.is_empty or clip.height < 2:
        return None
    pix = page.get_pixmap(clip=clip, dpi=dpi, colorspace=pymupdf.csGRAY)
    w, h = pix.width, pix.height
    if not w or not h:
        return None
    data = pix.samples
    for y in range(h):
        row = data[y * w:(y + 1) * w]
        if min(row) < 170:
            return clip.y0 + y * clip.height / h - 1
    return None


def _rect(b):
    return pymupdf.Rect(*b["rect"])


EQ_SCALE = 1.12


def _body_size(doc, boxes):
    """The size the paper actually sets its body text in (points).

    The 10.3 default fitted one journal; another sets 9.8 and came out five
    per cent too large everywhere.  Read it off the source's own text layer
    when there is one (scans have none, and fall back to the default).
    """
    import collections
    c = collections.Counter()
    for k, items in boxes.items():
        try:
            page = doc[int(k) - 1]
        except Exception:                        # noqa: BLE001
            continue
        rects = [pymupdf.Rect(r["rect"]) for r in items
                 if r["cls"] == "plain text"]
        if not rects:
            continue
        try:
            d = page.get_text("dict")
        except Exception:                        # noqa: BLE001
            continue
        for b in d["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    t = (sp.get("text") or "").strip()
                    if len(t) < 20:
                        continue
                    r = pymupdf.Rect(sp["bbox"])
                    if any(r.intersects(q) for q in rects):
                        c[round(sp["size"], 1)] += len(t)
    return c.most_common(1)[0][0] if c else None


def _fit_k(vlm, boxes):
    """No correction: measured against the layout boxes, the model's
    normalised coordinates land within a few points.  The 0.82 factor this
    function used to return came from fitting against OCR paragraphs, whose
    tops are unreliable -- it compressed correct coordinates and pushed every
    block up the page."""
    return 1.0


def _fit_k_unused(vlm, boxes):
    if not vlm or not boxes:
        return 1.0
    vb = [b for b in vlm if b["role"] in TEXT_ROLES]
    bb = [r for r in boxes if r["cls"] in TEXT_CLS]
    if len(vb) < 2 or len(bb) < 2:
        return 1.0
    best, bk = 1e18, 1.0
    for step in range(24):
        k = 0.72 + step * 0.015
        tot = 0.0
        for b in vb:
            x0, y0, x1, y1 = b["bbox"]
            cy = (y0 + y1) / 2.0 * 1000.0 and (y0 + y1) / 2.0
            cy = cy / 1000.0 * 663.0 * k
            d = min(abs((r["rect"][1] + r["rect"][3]) / 2.0 - cy) for r in bb)
            tot += min(d, 40.0)
        if tot < best:
            best, bk = tot, k
    return bk


def _number_for(page_src, capbox, cache, key):
    if key in cache:
        return cache[key]
    import subprocess
    import tempfile
    pix = page_src.get_pixmap(clip=pymupdf.Rect(*capbox["rect"]), dpi=400)
    tmp = os.path.join(tempfile.gettempdir(), "num_%s.png" % abs(hash(key)))
    pix.save(tmp)
    try:
        out = subprocess.run(["tesseract", tmp, "stdout", "--psm", "7", "-l", "eng"],
                             capture_output=True, text=True, timeout=30,
                             encoding="utf-8", errors="replace").stdout
    except Exception:                            # noqa: BLE001
        out = ""
    m = re.search(r"\(?\s*(\d{1,2})\s*\)?", out or "")
    val = m.group(1) if m else ""
    cache[key] = val
    return val


BOX_CLS = {"plain text", "title", "isolate_formula", "figure_caption",
           "table_caption", "formula_caption", "figure", "table",
           "page_header", "page_footer"}


def _boxes_by_page(boxes):
    """One deduped, top-down list of boxes per page, for the numbered image."""
    out = {}
    for k, items in boxes.items():
        chosen = []
        order = sorted(items, key=lambda r: -((r["rect"][2] - r["rect"][0])
                                             * (r["rect"][3] - r["rect"][1])))
        for r in order:
            if r["cls"] not in BOX_CLS:
                continue
            a = pymupdf.Rect(r["rect"])
            if any((a & pymupdf.Rect(c)).get_area() > 0.6 * a.get_area()
                   for c in chosen):
                continue
            chosen.append(r["rect"])
        chosen.sort(key=lambda q: (round(q[1] / 8.0), q[0]))
        out[int(k)] = chosen
    return out


def _layer_kind(src):
    """What the text layer is made of -- true text, an OCR layer, or a picture.

    Printed with the geometry line so a build that quietly behaves like v2 (no
    usable layer) cannot be mistaken for one where P2/P3 were in force.  See
    textlayer.py for the rules; this must never fail the build.
    """
    try:
        import textlayer
        return textlayer.analyze(src)["kind"]
    except Exception:                            # noqa: BLE001
        return "unknown"


def build(tag, suffix="_vy", size=10.3, lead=1.35, gap=0.35, direct=False,
          ids=False):
    import papers
    src = papers.find_paper(tag)
    name = os.path.basename(src)[:-4]
    doc = pymupdf.open(src)
    # the text layer's line boxes, if this paper has a usable one (see geom.py);
    # None means every rule below behaves exactly like v2
    G = None
    if settings.get("TR_GEOMETRY", "1") != "0":
        G = geom.load(name)
        if G is None and not os.path.exists(os.path.join(cache.GEOM, name + ".json")):
            extracted = geom.build(src, verbose=False)
            G = extracted if extracted.get("usable") else None
    kind = _layer_kind(src)
    if G:
        print("  geometry: %s, %d/%d pages, layer=%s"
              % (G.get("geom_source"), G.get("pages_with_layer"),
                 G.get("pages_total"), kind), flush=True)
    else:
        print("  geometry: none -> behaves like v2, layer=%s" % kind, flush=True)
    print("  [检测] 版面框…", flush=True)
    boxes = layout_yolo.boxes_of(name, src)
    print("  [检测] 完成：%d 页" % len(boxes), flush=True)


    # A page whose only detected text is a sliver (the rest is one big table)
    # would give a 60 pt "column", and every note on it would be set one
    # character per line.  Such a page borrows the document's own text width.
    _spans = []
    for _items in boxes.values():
        _xs = [r["rect"] for r in _items if r["cls"] in TEXT_CLS]
        if len(_xs) >= 3:
            _spans.append((min(r[0] for r in _xs), max(r[2] for r in _xs)))
    _spans.sort()
    DOC_X = _spans[len(_spans) // 2] if _spans else None
    if ids:
        # the model never makes a box: it reads the numbered detector boxes,
        # so every block's geometry is the detector's, to the point
        rows = vlm_page.blocks_of_ids(name, src, _boxes_by_page(boxes))
    else:
        rows = _dedupe(_split_display(vlm_page.blocks_of(name, src, direct=direct)))
    for warning in quality.transcription_warnings(rows, len(doc)):
        print("  " + warning, flush=True)

    # A block the model called a formula whose text is really prose with an
    # inline formula in it ("buy if $Z_t > A$,") does not compile and would be
    # dropped without a trace -- 11 lost four of them that way.  If it has a
    # $...$ span it is a text block: demote it before anything is translated.
    demoted = 0
    for r in rows:
        if r["role"] != "formula":
            continue
        tex = _latex(r.get("text"))
        if not tex or "$" not in (r.get("text") or ""):
            continue
        try:
            import texmath as _tm
            _tm.render(tex)
        except Exception:                     # noqa: BLE001
            r["role"] = "body"
            demoted += 1
    if demoted:
        print("  %d formula block(s) are prose with inline maths: set as text"
              % demoted)

    pages = {}
    for r in rows:
        pages.setdefault(int(r["page"]), []).append(r)

    # a figure/table block is pasted as a picture; any prose the model left in
    # its text is never translated and never drawn, so it would vanish
    lost = [(b["page"], b["role"], len(b.get("text") or ""))
            for b in rows
            if b["role"] in ("figure", "table")
            and len(b.get("text") or "") > 200]
    if lost:
        print("  ! picture block carries prose (lost): %s"
              % ", ".join("p%d %s %dch" % t for t in lost[:6]))

    if direct:
        # one call: the page image already carries the sentence, so the
        # model may as well write the Chinese.  No placeholders, no
        # second request, no split sentences to stitch back together.
        zmap = {}
        for pg in sorted(pages):
            for b in pages[pg]:
                t = (b.get("text") or "").strip()
                if t and b["role"] in TEXT_ROLES | {"footnote", "caption"}:
                    zmap[id(b)] = t
        print("  direct: %d blocks written by the vision model" % len(zmap))
    else:
        seq = []
        for pg in sorted(pages):
            for b in pages[pg]:
                if b["role"] not in TEXT_ROLES | {"footnote", "caption"}:
                    continue
                en = (b.get("text") or "").strip()
                if len(en) < 2:
                    continue
                en = re.sub(r"\$\^\{?(\d+)\}?\$", lambda m: SUP.get(m.group(1), m.group(0)), en)
                # Bold spans travel as placeholders: the translator is told to keep
                # those verbatim, while a literal ** is markdown and gets stripped.
                cnt = [0]

                def _bm(m):
                    cnt[0] += 1
                    return "\u27e6B%d\u27e7%s\u27e6B%d\u27e7" % (cnt[0], m.group(1), cnt[0])

                en = re.sub(r"\*\*(.+?)\*\*", _bm, en)
                t, keep = trans.protect_math(en)
                seq_bold = cnt[0]
                seq.append({"b": b, "en": t, "keep": keep, "join": 0,
                            "bold_n": cnt[0]})

        # A sentence cut by a column or a page break is one sentence.  The two
        # halves are joined *before* the translator sees them, and the joined
        # text stays one block for the typesetter: nothing has to be cut back,
        # and the reader never gets "T 的日内值…" starting a column with no
        # subject.  Only text blocks that touch each other may join -- a
        # formula or a picture between them is a wall.
        by_id = dict((id(s["b"]), s) for s in seq)
        order = [b for pg in sorted(pages) for b in pages[pg]]

        def next_text(k):
            """The next *text* block, and whether anything sat in between.

            Running heads, page numbers, captions, notes, pictures and tables
            are furniture: a paragraph that breaks across a page keeps going
            after them.  They are reported so a lead-in like "… are defined
            as:" does not swallow the text that belongs after the display."""
            skipped = False
            for j in range(k + 1, len(order)):
                r = order[j]["role"]
                if r == "formula":
                    # a display formula is a wall, always: "Defining ... as"
                    # + (6) + "these estimators can be expressed ..." is one
                    # sentence cut in two, and joining it would move the
                    # formula to the end of the sentence
                    return None, skipped
                if r in TEXT_ROLES:
                    s = by_id.get(id(order[j]))
                    if s is not None and s.get("_merged"):
                        continue          # already folded into an earlier block
                    return j, skipped
                skipped = True
            return None, skipped

        joins = 0
        k = 0
        while k < len(order) - 1:
            j, skipped = next_text(k)
            if j is None:
                break
            a, b = order[k], order[j]
            sa, sb = by_id.get(id(a)), by_id.get(id(b))
            # a title the model cut into one block per line is still one title:
            # it is set as one paragraph, not as four
            same_title = a["role"] == "title" and b["role"] == "title"
            if (sa is not None and sb is not None
                    and not sa.get("_merged") and not sb.get("_merged")
                    and not a.get("_definition") and not b.get("_definition")
                    and (same_title or
                         _joinable(a["role"], sa["en"], b["role"], sb["en"],
                                   skipped))):
                sa["en"], sa["keep"] = _merge_protected(
                    sa["en"], sa["keep"], sb["en"], sb["keep"])
                sa["bold_n"] = sa.get("bold_n", 0) + sb.get("bold_n", 0)
                sb["_merged"] = True
                b["_skip"] = True
                joins += 1
                continue                  # the head keeps growing
            k += 1
        if joins:
            print("  joined %d sentence(s) across a block break" % joins)

        payload, owners = [], []
        for it in seq:
            if it.get("_merged"):
                continue
            payload.append({"en": it["en"], "keep": list(it["keep"]),
                            "section": ""})
            owners.append((it, None))
        zh = []
        if payload:
            # pass the payload as built: the "keep" list must travel with the text
            zh = trans.Translator().translate(payload, name, name)
            print("  translated %d blocks" % len(seq))
        def _unbold(z, n):
            z = re.sub(r"\u27e6B\d+\u27e7", "**", z or "")
            if n and z.count("**") < 2 * n:
                # the translator rewrites the marker as a single star: take those
                # stars back as the bold delimiters they were standing in for
                seen = [0]

                def rep(m):
                    seen[0] += 1
                    return "**" if seen[0] <= 2 * n else ""

                z = re.sub(r"(?<!\*)\*(?!\*)", rep, z)
            return z

        zmap = {}
        if os.environ.get("VLM_NOTDRAWN_DEBUG"):
            print("  payload %d, seq %d, zh %d" % (len(payload), len(seq), len(zh)),
                  file=sys.stderr)
            for (h, _t), z in zip(owners, zh):
                if (h.get("b", {}).get("text") or "").startswith("Thus, equations"):
                    print("  target id=%s zh=%r" % (id(h["b"]), z),
                          file=sys.stderr)
        for (head, tail), z in zip(owners, zh):
            for issue in quality.translation_warnings(head["en"], z, head["keep"]):
                print("  ! %s p%s: %s" % (issue, head["b"].get("page"),
                                          head["en"][:65]), flush=True)
            z = _unbold(z, head.get("bold_n") or 0)
            if not z:
                continue
            if tail is None:
                zmap[id(head["b"])] = z
                continue
            ratio = len(head["en"]) / float(len(head["en"]) + len(tail["en"]))
            a, b2 = _split_zh(z, ratio)
            if a:
                zmap[id(head["b"])] = a
            if b2:
                zmap[id(tail["b"])] = b2

    # A sentence the model flags as continuing into the next block becomes one
    # flow item, so the typesetter never sees the break.  Text may not cross a
    # formula: "…被定义为：" + formula + "其中…" is two independent pieces even
    # though the colon says the sentence runs on.
    joined = 0
    for pg in sorted(pages):
        blocks = pages[pg]
        for k, b in enumerate(blocks):
            if not b.get("join_next") or id(b) not in zmap:
                continue
            nxt = blocks[k + 1] if k + 1 < len(blocks) else None
            if nxt is None or nxt["role"] not in TEXT_ROLES:
                continue                      # a formula/figure is a wall
            a = (zmap.get(id(b)) or "").rstrip()
            c = (zmap.get(id(nxt)) or "").lstrip()
            if not a or not c:
                continue
            zmap[id(b)] = a + _strip_overlap(a, c)
            zmap[id(nxt)] = ""
            # its text now lives in the block above: say so, or the audit counts
            # it as a body block that was never painted (19: "136/137")
            nxt["_skip"] = True
            b["join_next"] = False
            joined += 1
            if os.environ.get("VLM_NOTDRAWN_DEBUG"):
                print("  join-blank p%d %r" % (nxt.get("page"),
                      (nxt.get("text") or "")[:50]), file=sys.stderr)
    if joined:
        print("  joined %d sentence(s) across a block break" % joined)

    # a block that still starts mid-sentence gets no first-line indent
    prev_tail = None
    for pg in sorted(pages):
        for b in pages[pg]:
            if b["role"] not in TEXT_ROLES or id(b) not in zmap:
                continue
            z = (zmap[id(b)] or "").strip()
            if not z:
                continue
            if prev_tail is not None and prev_tail[-1:] not in "。！？；：.!?;:":
                b["_mid"] = True
                z = _strip_overlap(prev_tail, z)
                zmap[id(b)] = z
                if (os.environ.get("VLM_NOTDRAWN_DEBUG")
                        and (b.get("text") or "").startswith("Thus, equations")):
                    print("  mid-strip id=%s prev=%r new=%r"
                          % (id(b), prev_tail[-30:], z), file=sys.stderr)
            prev_tail = z

    num_cache = {}
    drawn = {}
    carry_by_page = {}
    carry_in = []
    SEEN = {}
    out = pymupdf.open()
    _t_layout = time.time()
    for i in range(doc.page_count):
        if i % 5 == 0 or i == doc.page_count - 1:
            print("  [排版] %d/%d 页  %.0fs"
                  % (i + 1, doc.page_count, time.time() - _t_layout), flush=True)
        W, H = doc[i].rect.width, doc[i].rect.height
        page = out.new_page(width=W, height=H)
        pipeline._fonts(page)
        render.MATH_MAX_W = W
        pboxes = boxes.get(str(i + 1), [])
        vlm = pages.get(i + 1, [])

        def cbox(b, k):
            x0, y0, x1, y1 = b["bbox"]
            return pymupdf.Rect(x0 / 1000.0 * W, y0 / 1000.0 * H * k,
                                x1 / 1000.0 * W, y1 / 1000.0 * H * k)

        k = 1.0 if ids else _fit_k(vlm, pboxes)
        texts = [r for r in pboxes if r["cls"] in TEXT_CLS]
        formulas = [r for r in pboxes if r["cls"] == "isolate_formula"]
        caps = [r for r in pboxes if r["cls"] in CAP_CLS]
        capnums = [r for r in pboxes if r["cls"] == "formula_caption"]
        figs = [r for r in pboxes if r["cls"] in FIG_CLS]
        # a figure box is the detector's guess at the bitmap; the bitmap's own
        # rectangle is exact, so take it whenever one sits inside the guess
        figs = geom.snap_figs(figs, geom.images(G, i + 1, doc[i].rect))
        foots = [r for r in pboxes if r["cls"] == "page_footer"]

        # Anything with a fixed position -- figure, running head, foot, caption
        # -- is claimed before a single line is set.  Flowing text is then laid
        # out by free_y(), which pushes it past whatever it would have covered;
        # a block can never land on top of a picture, because the picture was
        # already there when the block asked for room.
        claims = []

        def claim(r):
            claims.append(pymupdf.Rect(r))

        def _ov(a, b):
            return (a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5
                    and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5)

        def free_y(y, h, ax0, ax1):
            r = pymupdf.Rect(ax0, y, ax1, y + h)
            for _ in range(len(claims) + 2):
                hit = [c for c in claims if _ov(r, c)]
                if not hit:
                    return y
                y = max(c.y1 for c in hit) + 2.0
                r = pymupdf.Rect(ax0, y, ax1, y + h)
            return y

        # The notes close the page.  Whatever the model puts after the first
        # note belongs to the note strip -- including a display formula that
        # sits inside a note -- and is set there in order, never in the body.
        # Notes are chosen by role, not by "everything after the first note":
        # on a two-column page the left column's note is often read before the
        # right column's text, and a positional cut would drop that text.
        notes = [b for b in vlm if b['role'] == 'footnote']
        bodyv = [b for b in vlm if b['role'] in TEXT_ROLES]
        notev = list(notes)

        # where the note strip begins: everything printed above has to stop
        # there, or the notes and the prose end up sharing the same lines
        note_ys = [cbox(notes[0], k).y0] if notes else []
        note_ys += [r['rect'][1] for r in pboxes if r['cls'] == 'page_footer']
        # the strip also shows up as an "abandon" box low on the page, and that
        # box is exact while the model's note box can be eighty points out
        note_ys += [r['rect'][1] for r in pboxes
                    if r['cls'] == 'abandon' and r['rect'][1] > 0.55 * H]
        if note_ys:
            note_top = min(note_ys) - 4
            # ...and it can be wrong upwards too.  A cap above the last body
            # element would swallow that element into the notes, so the cap may
            # only move down, never up.
            last = max([r['rect'][3] for r in pboxes
                        if r['rect'][3] <= note_top + 96
                        and (r['cls'] in TEXT_CLS or r['cls'] == 'isolate_formula')]
                       + [0.0])
            note_top = max(note_top, last + 6)
        else:
            # no notes and no printed foot: the usable area runs to the paper
            # edge, and "-60" would carry the page's last formula to the next
            # page for no reason
            note_top = H - 8

        # a display formula inside the note band belongs to the notes, not to
        # the body: it is set there, in order, at note size
        if notes:
            # only a page that *has* notes has a note strip: without this guard
            # a formula printed at the foot of a note-less page is pulled into
            # a strip that is not there and gets centred across the page
            notev += [b for b in vlm
                      if b['role'] == 'formula' and cbox(b, k).y0 >= note_top - 2]
        # the body stream carries the text blocks *and* the display formulas
        # that are not part of the notes
        bodyv = [b for b in vlm
                 if (b['role'] in TEXT_ROLES or b['role'] == 'formula')
                 and b not in notev]

        x0, x1 = (min([r["rect"][0] for r in texts] or [40]),
                  max([r["rect"][2] for r in texts] or [W - 40]))
        if DOC_X and (x1 - x0) < 0.6 * (DOC_X[1] - DOC_X[0]):
            x0, x1 = DOC_X

        # A page may be set in two or three vertical strips.  The detector
        # boxes every line, so the gutter between them is measurable, and the
        # prose is then set inside its own column instead of being stretched
        # across both.  Single-column pages come back as one range and nothing
        # below changes.
        class _Ln(object):
            __slots__ = ("rect",)

            def __init__(self, r):
                self.rect = pymupdf.Rect(r)

        try:
            # the gutter is measured on the lines that live inside a column:
            # a title or an abstract spans the page, and its box fills the gap
            # that the column detector is looking for
            span0 = max(x1 - x0, 1.0)
            narrow = [_Ln(r["rect"]) for r in texts
                      if r["rect"][2] - r["rect"][0] < 0.55 * span0]
            colr = core.find_columns(narrow or [_Ln(r["rect"]) for r in texts], W, H)
        except Exception:                        # noqa: BLE001
            colr = [(x0, x1)]
        if len(colr) < 2:
            colr = [(x0, x1)]

        def col_of(cb):
            if len(colr) < 2:
                return 0
            span0 = x1 - x0
            mid = (cb.x0 + cb.x1) / 2.0
            # wide *and* centred: only then does a block belong to both columns.
            # A table that is merely wide but sits in the left column keeps the
            # right column's cursor where it was.
            if cb.width > 0.72 * span0 and abs(mid - (x0 + x1) / 2.0) < 0.22 * span0:
                return -1
            # a block that straddles the gutter is not inside either column: the
            # abstract of a two-column paper sits in the middle of the page and
            # must keep its own width, not be squeezed into the left column
            for i in range(len(colr) - 1):
                if cb.x0 < colr[i][1] - 2 and cb.x1 > colr[i + 1][0] + 2:
                    return -1
            cx = (cb.x0 + cb.x1) / 2.0
            best, dist = 0, 1e9
            for n, (a, b) in enumerate(colr):
                d = 0.0 if a - 10 <= cx <= b + 10 else min(abs(cx - a), abs(cx - b))
                if d < dist:
                    best, dist = n, d
            return best

        # The strip must also fit above whatever closes the page -- the printed
        # foot or the bottom margin.  Its height is known before a line of the
        # body is set, so the cap moves up by the whole strip, not by "top -4".
        def strip_height():
            tot = 0.0
            for b in notev:
                if b['role'] == 'formula':
                    tex = _latex(b.get('text'))
                    if not tex:
                        continue
                    try:
                        import texmath
                        tot += texmath.render(tex)['h'] * size * 0.8 * EQ_SCALE / 12.0
                    except Exception:            # noqa: BLE001
                        continue
                    tot += 1.5
                    continue
                z = zmap.get(id(b))
                if not z:
                    continue
                ln = layout.wrap_zh(z, size * 0.8, x1 - x0, False, 0.0)
                tot += len(ln) * size * 0.8 * lead + 1.5
            return tot

        if notev:
            # the printed foot is pasted from either class, so the note strip
            # has to keep clear of both -- otherwise the note is set *below*
            # the copyright line and the page number
            foot_ys = [r['rect'][1] for r in pboxes
                       if r['cls'] == 'page_footer'
                       or (r['cls'] == 'abandon' and r['rect'][1] > 0.85 * H)]
            bottom = min(foot_ys + [H - 24.0])
            note_top = min(note_top, bottom - strip_height() - 4)
        # Pictures are pasted exactly where the detector put them.  Carried text
        # is ordinary flow: if it would land on a picture, free_y moves the text,
        # never the picture.
        shift = 0.0


        # pictures first: the text may not be pushed around by them, but they
        # must be on the page before anything is measured against it
        for r in figs:
            rr = pymupdf.Rect(r["rect"][0], r["rect"][1] + shift,
                              r["rect"][2], r["rect"][3] + shift)
            page.insert_image(rr,
                              pixmap=doc[i].get_pixmap(clip=pymupdf.Rect(*r["rect"]), dpi=220))
            claim(rr)
            drawn["figure"] = drawn.get("figure", 0) + 1

        cap_boxes = [r for r in pboxes if r["cls"] in CAP_CLS]
        cap_blocks = [b for b in vlm if b["role"] == "caption" and zmap.get(id(b))]
        taken = set()
        for b in cap_blocks:
            z = zmap.get(id(b))
            cb = geom.snap_top(cbox(b, k), G, i + 1,
                               avoid=[pymupdf.Rect(*q["rect"]) for q in figs])
            # a caption is pinned to the box nearest its own top edge, not to
            # the n-th caption box: on a page with two pictures the two lists
            # are not in the same order, and zipping them swaps the captions
            near = None
            best = 60.0
            for q in cap_boxes:
                qk = tuple(round(v) for v in q["rect"])
                if qk in taken:
                    continue
                d = abs(q["rect"][1] + shift - cb.y0)
                if d < best:
                    near, best = q, d
            if near is not None:
                taken.add(tuple(round(v) for v in near["rect"]))
                rr = pymupdf.Rect(near["rect"][0], near["rect"][1] + shift,
                                  near["rect"][2], near["rect"][3] + shift)
            else:
                rr = pymupdf.Rect(cb)
            # a caption belongs to its own column, however wide its box is
            ci = col_of(cb)
            if ci >= 0:
                rr.x0 = max(rr.x0, colr[ci][0])
                rr.x1 = min(rr.x1, colr[ci][1])
            if rr.width < 40:
                rr = pymupdf.Rect(cb)
            run_ = layout.Run("text", zh=_bold_tags(z), ratio=0.85)
            sz = size * 0.85
            # the block is one claim, but its paragraphs keep their breaks:
            # label / sentence / note are separate lines in the printed caption,
            # and wrap_zh would join them with a space into "Figure 2随机波动率…"
            lines = []
            for para in (run_.zh or "").split("\n"):
                para = para.strip()
                if para:
                    lines.extend(layout.wrap_zh(para, sz, rr.width, False, 0.0))
            h = len(lines) * sz * lead
            # A translated caption is often longer than the printed one.  It
            # may not run into the picture or table it belongs to, so when the
            # lines do not fit below the caption box they grow *upwards* into
            # the blank that the short original caption left.
            fixed = [pymupdf.Rect(q["rect"][0], q["rect"][1] + shift,
                                  q["rect"][2], q["rect"][3] + shift)
                     for q in figs]
            fixed += [pymupdf.Rect(q["rect"][0], q["rect"][1] + shift,
                                   q["rect"][2], q["rect"][3] + shift)
                      for q in cap_boxes if q is not near]
            y0 = rr.y0
            box_rect = pymupdf.Rect(rr.x0, y0, rr.x1, y0 + h)
            if any(_ov(box_rect, q) for q in fixed):
                up = rr.y1 - h
                up_rect = pymupdf.Rect(rr.x0, up, rr.x1, up + h)
                if not any(_ov(up_rect, q) for q in fixed):
                    y0, box_rect = up, up_rect
                else:
                    y0 = free_y(y0, h, rr.x0, rr.x1)
                    box_rect = pymupdf.Rect(rr.x0, y0, rr.x1, y0 + h)
            blk = layout.Block(run_, lines=lines, page=i, x0=rr.x0, x1=rr.x1,
                               y0=y0, y1=y0 + h)
            claim(box_rect)
            drawn["caption"] = drawn.get("caption", 0) + 1
            pipeline._draw_block(page, blk, size, lead)

        def nearest(pool, y):
            best, br = 1e9, None
            for r in pool:
                c = (r["rect"][1] + r["rect"][3]) / 2.0
                if abs(c - y) < best:
                    best, br = abs(c - y), r
            return br if best < 45 else None

        # ---- anchors: top edge first, interpolation for the rest ----------
        items = []
        fig_rects = [pymupdf.Rect(*r["rect"]) for r in figs]
        for b in bodyv:
            if b['role'] not in TEXT_ROLES:
                continue
            z = zmap.get(id(b))
            if not z:
                if os.environ.get("VLM_NOTDRAWN_DEBUG"):
                    print("  skip (no translation) id=%s p%d %r"
                          % (id(b), i + 1, (b.get("text") or "")[:50]),
                          file=sys.stderr)
                continue
            cb0 = cbox(b, k)
            # a heading printed *inside* the picture ("APPENDIX A" above a big
            # table) is already in the pasted crop: drawing it again prints it
            # twice, once of them pushed below the picture
            if any((cb0 & fr).get_area() > 0.8 * cb0.get_area() for fr in fig_rects):
                b["_skip"] = True
                continue
            items.append({'b': b, 'z': z, 'cb': cb0, 'box': None, 'y': None})

        for it in items:                      # match by top edge, not centre
            best = min(texts, key=lambda r: abs(r['rect'][1] - it['cb'].y0), default=None)
            if best is not None and abs(best['rect'][1] - it['cb'].y0) <= 22:
                it['box'] = best
                it['y'] = best['rect'][1]

        stream = []
        # blocks with no box: place them between their matched neighbours,
        # using only the model's order -- its absolute y may be off by 60pt
        for ii, it in enumerate(items):
            if it['y'] is not None:
                continue
            prev = next(((items[j]) for j in range(ii - 1, -1, -1)
                         if items[j]['y'] is not None), None)
            nxt = next(((items[j]) for j in range(ii + 1, len(items))
                        if items[j]['y'] is not None), None)
            if prev and nxt:
                span = nxt['cb'].y0 - prev['cb'].y0
                f = (it['cb'].y0 - prev['cb'].y0) / span if span > 1 else 0.5
                it['y'] = prev['y'] + max(0.0, min(1.0, f)) * (nxt['y'] - prev['y'])
            elif prev:
                it['y'] = prev['y'] + (it['cb'].y0 - prev['cb'].y0)
            elif nxt:
                it['y'] = max(40.0, nxt['y'] - (nxt['cb'].y0 - it['cb'].y0))
            else:
                it['y'] = it['cb'].y0
            # then snap to the real ink so a small guess error disappears
            ink = _ink_y(doc[i], 40, it['y'] - 6, W - 40,
                         it['y'] + max(10.0, it['cb'].height) + 4)
            if ink is not None:
                it['y'] = ink

        # The model already returns the blocks in reading order.  Sorting by
        # the anchor overrules that order whenever two anchors are close, and a
        # formula then lands after the paragraph that introduces it.  Order
        # comes from the model; the anchor only says where on the page.
        idx_of = {id(b): n for n, b in enumerate(vlm)}
        # Formulas stand in the same order on both sides, so when the counts
        # agree the k-th model line *is* the k-th detected line.  Nearest-centre
        # matching sends a line 200 pt away to a box that merely happens to be
        # close, and the gap it leaves then swallows the note strip.
        bformulas = [r for r in formulas if r['rect'][3] <= note_top + 2]
        eqs = []
        prev_grp, prev_key = None, None
        for b in bodyv:
            if b['role'] != 'formula':
                continue
            tex = _latex(b.get('text'))
            if not tex:
                continue
            try:
                import texmath
                texmath.render(tex)
            except Exception:                # noqa: BLE001
                continue
            if G:
                # how the journal itself broke this display: line widths and
                # where each line starts, straight from the text layer
                b['_hint'] = geom.display_hint(G, i + 1, cbox(b, k))
            g = b.get('_grp')
            if g is not None and g == prev_grp and eqs:
                # a second, third ... line of the same display
                hb, htex, hnum = eqs[-1]
                hb['bbox'] = [min(hb['bbox'][0], b['bbox'][0]),
                              min(hb['bbox'][1], b['bbox'][1]),
                              max(hb['bbox'][2], b['bbox'][2]),
                              max(hb['bbox'][3], b['bbox'][3])]
                if isinstance(htex, str):
                    htex = [htex]
                eqs[-1] = (hb, htex + [tex], hnum)
                b['_skip'] = True          # drawn as part of the first line
                prev_key = id(b)
                continue
            # ONE line, not a list: a list means "these are the printed lines,
            # do not re-break", and a formula wrapped as [tex] can then only be
            # squeezed to fit the column -- 07 p3 (6) came out at 8.4 pt that way.
            # As a string it goes through split_ops and breaks at its operators.
            eqs.append((b, tex, (b.get('number') or '').strip()))
            prev_grp, prev_key = g, id(b)
        # every display is a list of lines: one line for an ordinary formula
        if len(eqs) == len(bformulas):
            pair = {id(b): r for (b, _t, _n), r in zip(eqs, bformulas)}
        else:
            pair = {}
            for b, _t, _n in eqs:
                pair[id(b)] = nearest(bformulas,
                                      (cbox(b, k).y0 + cbox(b, k).y1) / 2.0)

        # The model drops the printed number more often than it keeps it, but
        # the detector boxes that number on its own.  A number belongs to the
        # one line of the display whose band holds it, and to no other.
        used_num = set()
        page_nums = set()          # numbers already printed on this page
        last_num = [0]             # the last one printed, top to bottom

        def number_of(b):
            num = (b.get('number') or '').strip()
            if num:
                # a number the model printed counts as printed: the next line
                # of the same display must not borrow it from the caption box
                v = num.strip("()")
                if v in page_nums:
                    # the model repeated a number it already used on this page:
                    # printing it twice is worse than printing none
                    return ""
                page_nums.add(v)
                try:
                    last_num[0] = max(last_num[0], int(v))
                except ValueError:
                    pass
                return num
            if not capnums:
                return ""
            cb = cbox(b, k)
            box = pair.get(id(b))
            band = pymupdf.Rect(cb)
            if box:
                band.include_rect(pymupdf.Rect(*box['rect']))
            cand = []
            for r in capnums:
                key = tuple(round(v) for v in r['rect'])
                if key in used_num:
                    continue
                yc = (r['rect'][1] + r['rect'][3]) / 2.0
                gap = 0.0 if band.y0 - 1 <= yc <= band.y1 + 1 else \
                    min(abs(yc - band.y0), abs(yc - band.y1))
                cand.append((gap, abs(yc - cb.y1), len(cand), key, r))
            if not cand:
                return ""
            cand.sort(key=lambda t: t[:3])
            # Numbers run strictly downwards on a page and never repeat: if
            # the nearest box would print one we already have, the next box
            # along is the one that belongs to this line.
            for gap, _d, _n, key, r in cand:
                if os.environ.get("VLM_NUM_DEBUG"):
                    print("  num p%d y=%.0f gap=%.1f key=%s page_nums=%s last=%d"
                          % (i + 1, cb.y0, gap, key,
                             sorted(page_nums), last_num[0]), file=sys.stderr)
                if gap > 14:
                    break
                v = _number_for(doc[i], r, num_cache, (i, key))
                used_num.add(key)
                if not v or v in page_nums:
                    continue
                try:
                    iv = int(v)
                except ValueError:
                    iv = None
                if iv is not None and iv <= last_num[0]:
                    continue
                page_nums.add(v)
                if iv is not None:
                    last_num[0] = iv
                return "(%s)" % v
            return ""

        ybox = dict((id(it['b']), it['box']) for it in items)
        stream = []
        for it in items:
            stream.append((1, idx_of[id(it['b'])], it['y'], 'txt',
                           (it['b'], it['z']), col_of(it['cb'])))
        def claim_num(num):
            # a number the model printed is printed: it has to block the
            # detector fallback too, or the next line borrows it and the page
            # shows the same number twice
            v = num.strip("()")
            page_nums.add(v)
            try:
                last_num[0] = max(last_num[0], int(v))
            except ValueError:
                pass

        for b, tex, num0 in eqs:
            box = pair.get(id(b))
            ay = box['rect'][1] if box else cbox(b, k).y0
            if num0:
                claim_num(num0)
                num = num0
            else:
                num = number_of(b)
            stream.append((1, idx_of[id(b)], ay, 'eq',
                           (tex, num, b.get('_hint')),
                           col_of(cbox(b, k))))
        stream.sort(key=lambda t: (t[0], t[1]))
        # running head (with the printed page number) and footer: kept as
        # pictures, since neither is prose and neither may be translated
        done_boxes = set()

        def paste_box(q, tag):
            key = tuple(round(v) for v in q["rect"])
            if key in done_boxes:
                return
            done_boxes.add(key)
            # the detector boxes a running head to the nearest glyph, and the
            # first or last letters then fall outside the crop: take six points
            # of the blank margin either side, it can only hold blank
            rr = pymupdf.Rect(max(0.0, q["rect"][0] - 6), q["rect"][1],
                              min(W, q["rect"][2] + 6), q["rect"][3])
            page.insert_image(rr,
                              pixmap=doc[i].get_pixmap(clip=rr, dpi=300))
            claim(rr)
            drawn[tag] = drawn.get(tag, 0) + 1

        heads = [q for q in pboxes
                 if q["cls"] == "page_header"
                 or (q["cls"] == "abandon" and q["rect"][3] < 0.12 * H)]
        heads.sort(key=lambda q: -(q["rect"][2] - q["rect"][0]))
        # a section heading ("APPENDIX A") can be boxed as an abandon at the top
        # of the page; if the model transcribed it as a heading it is set in
        # Chinese, and pasting it as well would print it twice
        vhead = [cbox(b, k) for b in vlm
                 if b["role"] in ("heading", "title")]
        kept = []
        for q in heads:                           # the detector often boxes the
            r = q["rect"]                         # head twice, one inside the other
            if any(not (r[3] <= k[1] + 1 or r[1] >= k[3] - 1) for k in kept):
                continue
            rr = pymupdf.Rect(*r)
            if any((rr & vh).get_area() > 0.5 * rr.get_area() for vh in vhead):
                continue
            kept.append(r)
            paste_box(q, "header")
        # a copyright / permission paragraph at the foot of the first page is
        # boxed as a footer *and* transcribed by the model as a note.  Pasting
        # the picture as well prints the same paragraph twice, once in English.
        # A page number is a short box and is kept.
        vnotes = [cbox(b, k) for b in vlm if b["role"] == "footnote"]
        for q in pboxes:
            # the printed foot sits at the very bottom margin; a fixed fraction
            # is more reliable than "below the notes", because the model's note
            # box can start below the copyright line
            if q["cls"] == "page_footer" or (q["cls"] == "abandon"
                                             and q["rect"][1] > 0.85 * H):
                fr = pymupdf.Rect(*q["rect"])
                if fr.height > 16 and any((fr & vn).get_area() > 0
                                          for vn in vnotes):
                    continue
                paste_box(q, "footer")
        # nothing may start above the running head
        top_margin = max([q["rect"][3] for q in pboxes
                          if q["cls"] == "page_header"
                          or (q["cls"] == "abandon" and q["rect"][3] < 0.12 * H)]
                         + [24.0]) + 8
        # what the previous page could not hold comes first, from the top:
        # dumping it on an extra page at the end would break the reading order
        incoming = carry_in
        carry_in = []
        # captions are fixed too: they travel with their picture, so the body
        # has to keep clear of them just as it keeps clear of the picture

        stream = [(0, t[1], 0.0, t[3], t[4], 0) for t in incoming] + stream
        col_y = dict((n, top_margin) for n in range(len(colr)))
        # a column's first block is a continuation of the page before it: only a
        # block that follows another block *in the same column* may start a new
        # paragraph and take the first-line indent
        seen_col = dict((n, False) for n in range(len(colr)))
        carry = carry_by_page.setdefault(i, [])
        for item in stream:
            _rank, _idx, _ay, kind, obj, _ci = item
            # a block that spans the page (a wide figure, a banner heading)
            # waits for both columns and then frees both of them again
            if _ci < 0:
                bx0, bx1 = colr[0][0], colr[-1][1]
                y = max(col_y.values())
            else:
                bx0, bx1 = colr[_ci]
                y = col_y[_ci]
            if kind == 'eq':
                tex, num, hint = obj
                # the height of the laid-out formula, not of one unbroken line:
                # a long formula is broken at its operators, so it may need two
                # or three lines and the page must know that before placing it
                h = equations.height(tex, size * EQ_SCALE, bx1 - bx0, num,
                                     hint)
                yy = free_y(y, h, bx0, bx1)
                if yy + h > note_top:
                    carry.append(item)
                    continue
                claim((bx0, yy, bx1, yy + h + 2))
                box = pymupdf.Rect(bx0, yy, bx1, yy + h + 2)
                try:
                    d = equations.draw(page, box, tex, size * EQ_SCALE, num,
                                       hint)
                except Exception as e:            # noqa: BLE001
                    print("  !! eq %s p%d: %s" % (num or "?", i + 1,
                                                  str(e)[:90]), file=sys.stderr)
                    continue
                drawn['formula'] = drawn.get('formula', 0) + 1
                newy = d.y1 + max(gap, 0.6) * size
                if _ci < 0:
                    for n in col_y:
                        col_y[n] = newy
                else:
                    col_y[_ci] = newy
                continue
            b, z = obj
            z = _bold_tags(z)
            al = b.get('align') or 'left'
            if b['role'] in ('heading', 'title'):
                # the model calls a centred section heading "left" often enough
                # that the detector's own box settles it: a tight box whose
                # centre sits on the column's centre is a centred heading
                yb = ybox.get(id(b))
                if yb is not None and yb['cls'] != 'title':
                    # the anchor may have snapped to the paragraph under the
                    # heading; that box spans the column and says nothing
                    yb = None
                if yb is None:
                    # the model's box for a heading can be fifty points out; the
                    # detector's own heading box is the better witness
                    near = [r for r in texts
                            if abs((r['rect'][1] + r['rect'][3]) / 2.0 - _ay) < 150]
                    if _ci >= 0:
                        # the witness has to stand in the same column as the
                        # heading, or the box next door decides the alignment
                        c0, c1 = colr[_ci]
                        near = [r for r in near
                                if c0 - 8 <= (r['rect'][0] + r['rect'][2]) / 2.0
                                <= c1 + 8]
                    # a heading is a *title* box: the paragraph beside it spans
                    # the whole column and would always look left-aligned
                    cand = [r for r in near if r['cls'] == 'title'] or near
                    cand.sort(key=lambda r: abs((r['rect'][1] + r['rect'][3])
                                                / 2.0 - _ay))
                    yb = cand[0] if cand else None
                cx = ((yb['rect'][0] + yb['rect'][2]) / 2.0) if yb else None
                ww = (yb['rect'][2] - yb['rect'][0]) if yb else None
                colw = max(bx1 - bx0, 1.0)
                if (cx is not None and ww is not None and ww < 0.8 * colw
                        and abs(cx - (bx0 + bx1) / 2.0) < 0.06 * colw):
                    al = 'center'
                if os.environ.get("VLM_LAYOUT_DEBUG"):
                    print("  head p%d ay=%.0f yb=%s cx=%s ww=%s colw=%.0f colmid=%.0f -> %s"
                          % (i + 1, _ay, [round(v) for v in yb['rect']] if yb else None,
                             round(cx, 1) if cx else None, round(ww, 1) if ww else None,
                             colw, (bx0 + bx1) / 2.0, al), file=sys.stderr)
            if _ci < 0:
                # a block that spans the page keeps its own width: the abstract
                # is centred on the page, not stretched over both columns
                cb = cbox(b, k)
                bx0 = max(x0, min(cb.x0, x1 - 80))
                bx1 = min(x1, max(cb.x1, bx0 + 80))
            first_here = (not seen_col.get(_ci, False)) if _ci >= 0 else True
            run_ = layout.Run(
                'heading' if b['role'] == 'heading' else
                ('title' if b['role'] == 'title' else 'text'),
                zh=z,
                ratio=max(0.6, min(2.2, (b.get('size_pct') or 100) / 100.0)),
                bold=bool(b.get('bold')),
                align=al,
                indent=0.0 if (b.get('_cont') or b.get('_mid') or first_here) else
                (2.0 if b['role'] in ('body', 'abstract') else 0.0))
            if _ci >= 0 and b['role'] in ('body', 'abstract'):
                seen_col[_ci] = True
            sz = size * run_.ratio
            lines = layout.wrap_zh(run_.zh, sz, bx1 - bx0, run_.bold,
                                   run_.indent * sz)
            lh = sz * lead
            h = len(lines) * lh
            # the anchor says where the block starts; when the Chinese above is
            # shorter than the English, honouring it to the point leaves a hole
            # of a paragraph.  Relative spacing, not absolute position: never
            # more than two ems of source gap.
            if b['role'] in ('heading', 'title') and _ay < y:
                # a section heading keeps its own place even when the reading
                # order puts it after a full-page table: it is not flow text
                anchor = _ay
            else:
                anchor = _ay if _ay <= y + 2.0 * size else y + 2.0 * size
            yy = free_y(y if b.get('_definition') else
                        max(0.0, anchor) if b['role'] in ('heading', 'title')
                        else max(y, anchor), h, bx0, bx1)
            if yy + h > note_top:
                # fill the page to its last line and carry the rest: moving a
                # whole paragraph down leaves a hole the size of the paragraph
                nfit = int((note_top - yy) // lh)
                if nfit >= 2:
                    head = lines[:nfit]
                    rest_zh = "".join(layout.line_text(l) for l in lines[nfit:])
                    blk = layout.Block(run_, lines=head, page=i, x0=bx0, x1=bx1,
                                       y0=yy, y1=yy + len(head) * lh)
                    claim((bx0, yy, bx1, yy + len(head) * lh + 0.25 * sz))
                    drawn[b['role']] = drawn.get(b['role'], 0) + 1
                    pipeline._draw_block(page, blk, size, lead)
                    cont = dict(b)
                    cont['_cont'] = True
                    carry.append((1, idx_of.get(id(b), 0), 0.0, 'txt',
                                  (cont, rest_zh), 0))
                    col_y[_ci if _ci >= 0 else 0] = note_top
                    continue
                carry.append(item)
                continue
            block = layout.Block(run_, lines=lines, page=i, x0=bx0, x1=bx1,
                                 y0=yy, y1=yy + h)
            # the drawn ink runs a descender past the measured line box, so the
            # booked rect is a quarter em taller than the layout height
            claim((bx0, yy, bx1, yy + h + 0.25 * sz))
            drawn[b['role']] = drawn.get(b['role'], 0) + 1
            SEEN.setdefault(id(b), []).append(i + 1)
            pipeline._draw_block(page, block, size, lead)
            newy = yy + h + (0.12 if b.get('_definition') else gap) * size
            if b['role'] in ('heading', 'title'):
                newy = max(newy, y)      # a heading placed above must not rewind
            if os.environ.get("VLM_LAYOUT_DEBUG"):
                print("  lay p%d %s col=%d y=%.0f h=%.0f ay=%.0f %r"
                      % (i + 1, b["role"], _ci, yy, h, _ay, z[:16]), file=sys.stderr)
            if _ci < 0:
                for n in col_y:
                    col_y[n] = newy
            else:
                col_y[_ci] = newy
        carry_in = carry
        # Equation numbers come from the model only.  The detector's
        # formula_caption boxes were used as a fallback, but an OCR'd number
        # that belongs to some other line prints a stray "(18)" under a
        # formula -- a missing number is the lesser fault.
        # The note strip, in reading order: a display formula inside a note is
        # a line of its own there, not a fragment of the surrounding sentence.
        ynote = note_top
        for b in notev:
            if b["role"] == "formula":
                tex = _latex(b.get("text"))
                if not tex:
                    continue
                nhint = (geom.display_hint(G, i + 1, cbox(b, k))
                         if G else None)
                try:
                    h = equations.height(tex, size * 0.8 * 1.12, x1 - x0,
                                         (b.get("number") or "").strip(),
                                         nhint)
                except Exception:                # noqa: BLE001
                    continue
                ynote = free_y(ynote, h + 2, x0, x1)
                box = pymupdf.Rect(x0, ynote, x1, ynote + h + 2)
                claim((x0, ynote, x1, ynote + h + 2))
                try:
                    d = equations.draw(page, box, tex, size * 0.8 * 1.12,
                                       (b.get("number") or "").strip(), nhint)
                except Exception:                # noqa: BLE001
                    continue
                drawn["formula"] = drawn.get("formula", 0) + 1
                ynote = d.y1 + 1.5
                continue
            z = zmap.get(id(b))
            if not z:
                continue
            # the same boundary the body was capped at: one number, one rule
            run_ = layout.Run("text", zh=_bold_tags(z), ratio=0.8)
            sz = size * 0.8
            lines = layout.wrap_zh(run_.zh, sz, x1 - x0, False, 0.0)
            ynote = free_y(ynote, len(lines) * sz * lead, x0, x1)
            blk = layout.Block(run_, lines=lines, page=i, x0=x0, x1=x1,
                               y0=ynote, y1=ynote + len(lines) * sz * lead)
            claim((x0, ynote, x1, ynote + len(lines) * sz * lead))
            drawn["footnote"] = drawn.get("footnote", 0) + 1
            pipeline._draw_block(page, blk, size, lead)
            ynote += len(lines) * sz * lead + 1.5
        # the strip starts at the note top even on a page without notes, so
        # only a page that actually set a note can overflow
        if notev and ynote > H - 4:
            print("  ! notes overflow page %d by %.0fpt" % (i + 1, ynote - (H - 4)))

    # only the last page's leftovers need new sheets: everything else already
    # continued on the following printed page
    extra_pages = 0
    while carry_in:
        extra_pages += 1
        page = out.new_page(width=doc[-1].rect.width, height=doc[-1].rect.height)
        pipeline._fonts(page)
        render.MATH_MAX_W = page.rect.width
        x0, x1 = 52, page.rect.width - 58
        y = 60.0
        rest = []
        for item in carry_in:
            _rank, _idx, _ay, kind, obj = item[:5]
            if kind == 'eq':
                tex, num = obj
                h = equations.height(tex, size * EQ_SCALE, x1 - x0, num)
                if y + h > page.rect.height - 60:
                    rest.append(item)
                    continue
                try:
                    d = equations.draw(page, pymupdf.Rect(x0, y, x1, y + h + 2),
                                       tex, size * EQ_SCALE, num)
                except Exception:            # noqa: BLE001
                    continue
                y = d.y1 + gap * size
            else:
                b, z = obj
                run_ = layout.Run('text', zh=z,
                                  ratio=max(0.6, min(2.2, (b.get('size_pct') or 100) / 100.0)),
                                  bold=bool(b.get('bold')),
                                  align='center' if b.get('align') == 'center' else 'left',
                                  indent=0.0 if (b.get('_cont') or b.get('_mid')) else
                (2.0 if b['role'] in ('body', 'abstract') else 0.0))
                sz = size * run_.ratio
                lines = layout.wrap_zh(run_.zh, sz, x1 - x0, run_.bold, run_.indent * sz)
                h = len(lines) * sz * lead
                if y + h > page.rect.height - 60:
                    rest.append(item)
                    continue
                blk = layout.Block(run_, lines=lines, page=out.page_count - 1,
                                   x0=x0, x1=x1, y0=y, y1=y + h)
                drawn[b['role']] = drawn.get(b['role'], 0) + 1
                pipeline._draw_block(page, blk, size, lead)
                y = y + h + gap * size
        carry_in = rest
    dups = {k: v for k, v in SEEN.items() if len(v) > 1}
    if dups:
        print("  ! drawn twice: %d block(s), pages %s" % (len(dups), list(dups.values())[:4]))
    want = {}
    for b in rows:
        if b.get("_skip"):              # merged into the block before it
            continue
        if (os.environ.get("VLM_NOTDRAWN_DEBUG") and b["role"] == "body"
                and id(b) not in SEEN):
            print("  unpainted p%s bbox=%s %r"
                  % (b.get("page"), [round(v) for v in b["bbox"]],
                     (b.get("text") or "")[:50]), file=sys.stderr)
        want[b["role"]] = want.get(b["role"], 0) + 1
    short = ["%s %d/%d" % (r, drawn.get(r, 0), n) for r, n in sorted(want.items())
             if r not in ("header", "footer", "table") and drawn.get(r, 0) < n]
    if short:
        print("  ! not drawn: %s" % ", ".join(short))
    print("  drawn: %s%s" % (", ".join("%s %d" % (a, b) for a, b in sorted(drawn.items())),
                             ", extra pages %d" % extra_pages if extra_pages else ""))
    out.subset_fonts()
    out.set_metadata({"producer": "translated pipeline %s" % cache.VERSION,
                      "title": name})
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "【翻译】%s%s.pdf" % (name, suffix))
    out.save(dst, garbage=4, deflate=True)
    print("  saved %s" % dst)
