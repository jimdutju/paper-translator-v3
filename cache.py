"""v3-owned cache paths, usable in the Quant workspace or a standalone repo.

Older v2 caches can be explicitly copied into these paths for a warm start.
No v3 build writes into the frozen v2 directory.
"""
import os
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..")) if os.path.isdir(
    os.path.join(HERE, "..", "_v2")) else HERE
V2 = os.path.join(ROOT, "_v2")
PAPERS = settings.path("TR_PAPERS", os.path.join(ROOT, "..", "papers")
                       if ROOT != HERE else os.path.join(HERE, "papers"))

VERSION = "v3"

# v3 owns its caches: a rebuild must not change the frozen v2 baseline.
BASE = settings.path("TR_CACHE_DIR", os.path.join(HERE, ".cache"))
VLM = os.path.join(BASE, "vlm")
YOLO = os.path.join(BASE, "yolo")
YOLO_CFG = os.path.join(BASE, "yolo_config")
TEX = os.path.join(BASE, "tex")
ZH = os.path.join(BASE, "translations.json")
GLOSS = os.path.join(BASE, "glossary.json")

# ours alone
GEOM = os.path.join(HERE, "_geom")
# where builds land; translate.py points this at --out for each child process
OUT = os.path.abspath(os.environ.get("TR_OUT")
                      or os.path.join(ROOT, "_out", VERSION))


def out_pdf(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, "\u3010\u7ffb\u8bd1\u3011%s_vy.pdf" % name)
