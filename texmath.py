"""Compile a LaTeX formula with pdflatex and measure the resulting box.

The output stays a PDF page; it is placed with ``show_pdf_page`` so the formula
remains vector content (crisp at any zoom, selectable).
"""
import hashlib
import os
import re
import subprocess
import tempfile

import pymupdf

import settings
BIN = settings.get("TR_PDFLATEX")  # resolve PATH lazily; cached maths need no compiler
import cache

CACHE = cache.TEX

TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,mathrsfs}
\usepackage{mathptmx}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\newbox\fb
\begin{document}
\setbox\fb=\hbox{$%s$}%%
\typeout{FORMULABOX \the\ht\fb\space\the\dp\fb}%%
\box\fb
\end{document}
"""


def _key(latex):
    return hashlib.sha1(latex.encode("utf-8")).hexdigest()[:16]


def _measure(path, tex_depth=None):
    doc = pymupdf.open(path)
    page = doc[0]
    pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    import numpy as np
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ys, xs = np.nonzero(a < 128)
    if len(xs) == 0:          # nothing typeset: a broken formula
        return None
    D = 300 / 72.0
    rect = pymupdf.Rect(xs.min() / D, ys.min() / D, (xs.max() + 1) / D, (ys.max() + 1) / D)
    base = _baseline(doc, page)
    depth = tex_depth if tex_depth is not None else max(0.0, rect.y1 - base)
    return {"w": round(rect.width, 2), "h": round(rect.height, 2),
            "depth": round(depth, 2),
            "x0": round(rect.x0, 2), "y0": round(rect.y0, 2),
            "base": round(base, 2)}


def _balanced(latex):
    """Braces must balance, or TeX swallows the rest of the batch."""
    B = chr(92)
    depth = 0
    i = 0
    while i < len(latex):
        ch = latex[i]
        if ch == B and i + 1 < len(latex):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def render_many(latexes, workers=8):
    """Compile the missing formulas in parallel.

    One pdflatex process per formula is the slow path (0.3 s startup each), but
    it is the only one that stays correct when a formula has unbalanced braces:
    a single TeX run for a whole paper breaks page-to-formula alignment the
    moment one of them fails.  A pool of 8 processes brings 168 formulas down
    from ~95 s to ~30 s.
    """
    from concurrent.futures import ThreadPoolExecutor
    todo, seen = [], set()
    for latex in latexes:
        latex = sanitize(_clean(latex))
        if not latex.strip() or latex in seen:
            continue
        seen.add(latex)
        if not (os.path.exists(os.path.join(CACHE, _key(latex) + ".pdf"))
                and os.path.exists(os.path.join(CACHE, _key(latex) + ".json"))):
            todo.append(latex)
    if not todo:
        return 0
    os.makedirs(CACHE, exist_ok=True)

    def one(latex):
        try:
            render(latex)
            return 1
        except Exception:                  # noqa: BLE001
            return 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(one, todo))


def _clean(latex):
    latex = latex.replace(chr(0x2019), chr(39)).replace(chr(0x201d), chr(39))
    return latex.replace(chr(0x2018), chr(39)).replace(chr(0x201c), chr(39))


UNI = {
    "ϵ": r"\epsilon", "ϑ": r"\vartheta", "ϕ": r"\phi",
    "ς": r"\varsigma", "ℓ": r"\ell", "×": r"\times",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta",
    "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega",
    "−": "-", "×": r"\times", "·": r"\cdot", "≤": r"\le", "≥": r"\ge", "≠": r"\neq",
    "≈": r"\approx", "∞": r"\infty", "∂": r"\partial", "∑": r"\sum", "∫": r"\int",
    "√": r"\sqrt", "∈": r"\in", "→": r"\to", "⟨": r"\langle", "⟩": r"\rangle",
}


_FAILED = set()


def sanitize(latex):
    """Make a model-written formula digestible for plain pdflatex."""
    B = chr(92)
    for mark in (0x2019, 0x201d, 0x2018, 0x201c):
        latex = latex.replace(chr(mark), chr(39))
    for env in ("equation*", "equation", "displaymath"):
        latex = latex.replace(B + 'begin{' + env + '}', '').replace(B + 'end{' + env + '}', '')
    while B + 'tag{' in latex:
        i = latex.index(B + 'tag{')
        j = latex.index('}', i)
        num = latex[i + 5:j]
        latex = latex[:i] + B + 'qquad (' + num + ')' + latex[j + 1:]
    out = []
    k = 0
    while k < len(latex):
        if latex[k:k + 2] == B + 'u' and len(latex) >= k + 6:
            hx = latex[k + 2:k + 6]
            if all(c in '0123456789abcdefABCDEF' for c in hx):
                out.append(chr(int(hx, 16)))
                k += 6
                continue
        out.append(latex[k])
        k += 1
    latex = ''.join(out)
    import unicodedata
    res = []
    for ch in latex:
        if ord(ch) < 128:
            res.append(ch)
        elif ch in UNI:
            res.append(UNI[ch] + ' ')
        else:
            res.append(unicodedata.normalize('NFKD', ch).encode('ascii', 'ignore').decode())
    return ''.join(res)

def render(latex):
    latex = sanitize(latex)
    """Return dict(pdf, w, h, depth) in points, baseline-relative.

    `depth` is how far the ink reaches below the baseline; the caller needs it
    to sit the formula on the text baseline instead of its bounding box.
    """
    if latex in _FAILED:
        raise RuntimeError("latex failed earlier: %s" % latex[:40])
    os.makedirs(CACHE, exist_ok=True)
    k = _key(latex)
    out = os.path.join(CACHE, k + ".pdf")
    meta = os.path.join(CACHE, k + ".json")
    if os.path.exists(out) and os.path.exists(meta) and os.path.getsize(out) > 0:
        import json
        try:
            with open(meta, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:                     # noqa: BLE001
            d = None
        if d is not None:
            d["pdf"] = out
            d["key"] = k
            return d
        for p in (out, meta):
            try:
                os.remove(p)
            except OSError:
                pass

    with tempfile.TemporaryDirectory() as td:
        tex = os.path.join(td, "f.tex")
        with open(tex, "w", encoding="utf-8") as f:
            f.write(TEMPLATE % latex)
        compiler = BIN or settings.pdflatex()
        env = dict(os.environ, PATH=os.path.dirname(compiler) + os.pathsep + os.environ.get("PATH", ""))
        r = subprocess.run([compiler, "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory", td, tex],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=120)
        src = os.path.join(td, "f.pdf")
        if not os.path.exists(src):
            raise RuntimeError("latex failed: " + (r.stdout or "")[-400:])
        import shutil
        # write it atomically: a formula is compiled on demand, and a second
        # process (or thread) asking for the same one must never read a
        # half-copied file -- 11 died on exactly that, a truncated .pdf in _tex
        tmp = out + ".%d.tmp" % os.getpid()
        shutil.copyfile(src, tmp)
        os.replace(tmp, out)

    doc = pymupdf.open(out)
    page = doc[0]
    import re as _re
    m = _re.search(r"FORMULABOX ([0-9.]+)pt\s+([0-9.]+)pt", r.stdout or "")
    tex_h = float(m.group(1)) + float(m.group(2)) if m else None
    tex_d = float(m.group(2)) if m else None
    # find the ink box: this gives width/height, and the baseline via the text
    pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    import numpy as np
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ys, xs = np.nonzero(a < 128)
    D = 300 / 72.0
    rect = pymupdf.Rect(xs.min() / D, ys.min() / D, (xs.max() + 1) / D, (ys.max() + 1) / D)
    base = _baseline(doc, page)
    # TeX's own \dp is exact; the span-based guess is only a fallback
    depth = tex_d if tex_d is not None else max(0.0, rect.y1 - base)
    d = {"w": round(rect.width, 2), "h": round(rect.height, 2),
         "depth": round(depth, 2),
         "x0": round(rect.x0, 2), "y0": round(rect.y0, 2),
         "base": round(base, 2)}
    import json
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(d, f)
    # one document per formula, and a 30-page paper compiles a few thousand of
    # them: keep none of them open or MuPDF runs out of file handles (11 died
    # with "Failed to open file" on a file that was there and 4.5 kB long)
    try:
        doc.close()
    except Exception:                         # noqa: BLE001
        pass
    d["pdf"] = out
    return d


def _baseline(doc, page):
    """Baseline of the formula on the TeX page.

    The compiled page still has a text layer, and every span records the
    baseline it was set on (`origin`).  Superscripts sit above it, so the
    largest origin is the main line.
    """
    best = None
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                # the main line is the one set in the largest type: a
                # subscript or superscript would otherwise define the baseline
                # and shift the whole formula (eps_t h_t^{-1/2} was 5 pt low)
                if best is None or sp["size"] > best[0]:
                    best = (sp["size"], sp["origin"][1])
    return best[1] if best else 72.0


if __name__ == "__main__":
    for f in [r"s^{\alpha}=s^{\prime\alpha}+s^{\prime\prime\alpha}",
              r"(A)\quad s^{\alpha}=s^{\prime\alpha}+s^{\prime\prime\alpha}",
              r"U_n-\delta\ \text{ and }\ N^{-1/\alpha}\sum_{n=1}^{N}(U_n-\delta)"]:
        d = render(f)
        print("%-56s w=%.1f h=%.1f depth=%.1f" % (f[:56], d["w"], d["h"], d["depth"]))
