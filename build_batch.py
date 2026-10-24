"""Rebuild papers one per process.

The TeX/PDF temp files the renderer opens are not all closed, and a single
process runs out of handles after one big paper.  A process per paper gives
every paper a clean handle table.
"""
import subprocess
import sys

TAGS = sys.argv[1:] or ["02_cont2001", "03_engle1982", "05_gjr1993", "06_engle2002"]

for tag in TAGS:
    print("=== %s" % tag, flush=True)
    rc = subprocess.call([sys.executable, "-u", "-c",
                          "import vlm_yolo; vlm_yolo.build(%r)" % tag])
    if rc:
        print("!!! %s failed (rc=%d)" % (tag, rc), flush=True)
print("=== done", flush=True)
