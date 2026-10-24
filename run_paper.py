"""Build one paper into _out/v3 and say whether it is actually finished.

usage:  python run_paper.py 06_engle2002 [more tags ...]

One process per paper: a batch that builds several in one interpreter runs
out of file handles around the second paper (fitz raises "Too many open
files"), so each tag gets its own child process.

Besides the console verdict, a <tag>.json is left next to the PDF: pages,
text-on-image, every "!" warning, seconds, sha256.  publish.py refuses to make
a build public unless that record says the build is clean.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import pymupdf

import cache
import quality
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cache.OUT


def audit(pdf):
    """Pages, and how many text blocks sit on top of a picture."""
    doc = pymupdf.open(pdf)
    bad = 0
    for page in doc:
        imgs = [pymupdf.Rect(r) for im in page.get_images(full=True)
                for r in page.get_image_rects(im[0])]
        for b in page.get_text("blocks"):
            r = pymupdf.Rect(b[:4])
            if any(min(r.x1, q.x1) - max(r.x0, q.x0) > 1
                   and min(r.y1, q.y1) - max(r.y0, q.y0) > 1 for q in imgs):
                bad += 1
    return len(doc), bad


def build(tag):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    code = "import vlm_yolo; vlm_yolo.build(%r)" % tag
    r = subprocess.run([sys.executable, "-u", "-c", code], cwd=HERE,
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log = (r.stdout or "") + (r.stderr or "")
    return r.returncode, log


def record(tag, pdf, log, pages, on_image, seconds, geo=None):
    h = hashlib.sha256(open(pdf, "rb").read()).hexdigest()
    rec = {
        "tag": tag, "version": cache.VERSION,
        "pdf": settings.display_path(pdf, cache.ROOT).replace("\\", "/"),
        "sha256": h, "pages": pages, "text_on_image": on_image,
        "geometry": geo,
        "warnings": [l.strip() for l in log.splitlines()
                     if l.strip().startswith("!")],
        "seconds": round(seconds), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    return rec


def main(tags):
    os.chdir(HERE)
    ok = True
    for tag in tags:
        print("\n=== %s" % tag, flush=True)
        t0 = time.time()
        rc, log = build(tag)
        tail = [l for l in log.strip().splitlines() if l.strip()][-14:]
        print("\n".join("  " + l.strip() for l in tail))
        warn = [l.strip() for l in log.splitlines() if l.strip().startswith("!")]
        pdf = os.path.join(OUT, "\u3010\u7ffb\u8bd1\u3011%s_vy.pdf" % tag)
        if rc or not os.path.exists(pdf):
            print("  !! build failed (rc=%s), no PDF" % rc)
            ok = False
            continue
        pages, on_image = audit(pdf)
        geo = next((l.strip() for l in log.splitlines()
                    if l.strip().startswith("geometry:")), None)
        print("  pages %d   text-on-image %d   %.0fs"
              % (pages, on_image, time.time() - t0))
        # printed on its own line: the tail above only keeps the last 14 lines,
        # and this one is written at the very start of the build
        print("  %s" % (geo or "geometry: (not reported)"))
        for line in warn:
            print("  warning: %s" % line)
        rec = record(tag, pdf, log, pages, on_image, time.time() - t0, geo)
        checked = quality.write_record(tag, pdf, log.splitlines(), time.time() - t0)
        # Keep the workspace publication contract (relative to translated/).
        checked["pdf"] = rec["pdf"]
        with open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8") as f:
            json.dump(checked, f, ensure_ascii=False, indent=2)
        warn = checked["warnings"]
        print("  record %s" % rec["pdf"])
        if on_image or warn:
            ok = False
    print("\n%s" % ("ALL CLEAN (structural checks only; semantic review still required)" if ok else "NEEDS A LOOK"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["06_engle2002"]))
