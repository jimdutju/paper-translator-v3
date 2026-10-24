"""Conservative, machine-checkable quality gates; never a semantic score."""
import collections
import hashlib
import json
import os
import re
import time

import pymupdf

PLACEHOLDER = re.compile(r"(?:⟦[MB]\d+⟧|\ue000\d+\ue001)")
RAW_TEX = re.compile(r"\\(?:frac|sqrt|begin|mathrm|mathbf|sigma|alpha|beta|lambda)\b")


def transcription_warnings(rows, page_count):
    warnings = []
    seen = {int(b.get("page") or 0) for b in rows}
    missing = sorted(set(range(1, page_count + 1)) - seen)
    if missing:
        warnings.append("! transcription missing pages: " + ",".join(map(str, missing)))
    for b in rows:
        box = b.get("bbox") or []
        if (len(box) != 4 or not all(isinstance(v, (int, float)) for v in box)
                or not 0 <= box[0] < box[2] <= 1000
                or not 0 <= box[1] < box[3] <= 1000):
            warnings.append("! invalid transcription bbox p%s" % b.get("page"))
        text = b.get("text") or ""
        if b.get("role") == "formula" and re.search(r"(?<!\\)[_^]\{", text) and "$" not in text and "\\" not in text:
            warnings.append("! formula may contain unparsed plain-text maths p%s" % b.get("page"))
    return list(dict.fromkeys(warnings))


def translation_warnings(source, translated, keep=()):
    """Check omissions and mathematical identity, without judging prose style."""
    if not (translated or "").strip():
        return ["missing translation"]
    issues = []
    if PLACEHOLDER.search(translated):
        # Bold placeholders are restored later; math placeholders must be gone.
        if re.search(r"⟦M\d+⟧", translated):
            issues.append("unresolved math placeholder")
    actual = collections.Counter(re.findall(r"\$([^$]+)\$", translated))
    expected = collections.Counter(keep or [])
    if expected != actual:
        issues.append("inline math changed or missing")
    return issues


def audit_pdf(pdf):
    """Return counts plus exact pages requiring inspection."""
    overlaps, blank, placeholders, raw_tex = 0, [], [], []
    with pymupdf.open(pdf) as doc:
        pages = len(doc)
        for n, page in enumerate(doc, 1):
            text = page.get_text()
            if PLACEHOLDER.search(text):
                placeholders.append(n)
            if RAW_TEX.search(text):
                raw_tex.append(n)
            if len(text.strip()) < 12:
                pix = page.get_pixmap(dpi=24, colorspace=pymupdf.csGRAY, alpha=False)
                ink = sum(v < 220 for v in pix.samples) / max(1, len(pix.samples))
                if ink < 0.001:
                    blank.append(n)
            imgs = [pymupdf.Rect(r) for im in page.get_images(full=True)
                    for r in page.get_image_rects(im[0])]
            for b in page.get_text("blocks"):
                r = pymupdf.Rect(b[:4])
                if any((r & q).width > 1 and (r & q).height > 1 for q in imgs):
                    overlaps += 1
    warnings = []
    for label, found in [("blank pages", blank), ("unresolved placeholders", placeholders),
                         ("raw LaTeX in output", raw_tex)]:
        if found:
            warnings.append("! %s: %s" % (label, ",".join(map(str, found))))
    return {"pages": pages, "text_on_image": overlaps, "blank_pages": blank,
            "placeholder_pages": placeholders, "raw_latex_pages": raw_tex,
            "warnings": warnings}


def write_record(tag, pdf, lines, seconds, version="v3"):
    audit = audit_pdf(pdf)
    warnings = list(dict.fromkeys([s.strip() for s in lines if s.strip().startswith("!")]
                                  + audit["warnings"]))
    geometry = next((s.strip() for s in lines if s.strip().startswith("geometry:")), None)
    with open(pdf, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    record = dict(audit, tag=tag, version=version, pdf=os.path.basename(pdf),
                  sha256=digest, geometry=geometry, warnings=warnings,
                  seconds=round(seconds), at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  verdict="needs_review" if warnings or audit["text_on_image"] else "checks_passed",
                  scope="Structural checks only; transcription and translation accuracy require review.")
    target = os.path.join(os.path.dirname(pdf), tag + ".json")
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(temporary, target)
    return record


def record_is_clean(pdf, tag):
    target = os.path.join(os.path.dirname(pdf), tag + ".json")
    try:
        with open(target, encoding="utf-8") as f:
            record = json.load(f)
        with open(pdf, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        return (record.get("verdict") == "checks_passed" and record.get("sha256") == digest
                and not record.get("warnings") and not record.get("text_on_image"))
    except (OSError, ValueError):
        return False
