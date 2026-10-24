"""Layout boxes from DocLayout-YOLO.

The model answers one question: where are the text blocks, formulas, figures,
captions, headers and footers on this page.  It is not asked what the text says
-- that is the vision model's job.
"""
from __future__ import annotations

import json
import os

import pymupdf

import cache
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = cache.YOLO
WEIGHTS = settings.path("TR_YOLO_WEIGHTS", "models/doclayout_yolo_docstructbench_imgsz1024.pt")
DPI = 144
CONF = 0.2

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        if not os.path.isfile(WEIGHTS):
            raise RuntimeError("DocLayout-YOLO weights not found. Set TR_YOLO_WEIGHTS.")
        os.environ.setdefault("YOLO_CONFIG_DIR", cache.YOLO_CFG)
        os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
        from doclayout_yolo import YOLOv10
        _MODEL = YOLOv10(WEIGHTS)
    return _MODEL


def _detect(doc):
    import numpy as np
    m = _model()
    out = {}
    sc = 72.0 / DPI
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        res = m.predict(img, imgsz=1024, conf=CONF, device="cpu", verbose=False)[0]
        rows = []
        for b in res.boxes:
            x0, y0, x1, y1 = [float(v) * sc for v in b.xyxy[0]]
            rows.append({"cls": res.names[int(b.cls)],
                         "rect": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                         "conf": round(float(b.conf), 3)})
        rows.sort(key=lambda r: (r["rect"][1], r["rect"][0]))
        out[str(i + 1)] = rows
    return out


def boxes_of(name, src, force=False):
    """Cached layout boxes, keyed by page number (1-based)."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "%s.json" % name)
    doc = pymupdf.open(src)
    if not force and os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if len(d) >= doc.page_count:
                return d
        except Exception:                       # noqa: BLE001
            pass
    d = _detect(doc)
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return d
