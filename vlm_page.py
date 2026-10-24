"""Page structure from the vision model: roles, boxes, alignment, size.

One request carries the whole paper, so the model can join sentences that run
across a page break.  The result is cached per paper: the call is the expensive
part, the JSON is the artefact everything downstream reads.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request

import pymupdf
import yaml

import cache
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
# the transcription is shared with the old pipeline: it costs minutes and the
# prompt is still the same one
CACHE_DIR = cache.VLM
API = settings.get("TR_VISION_URL", "https://api.deepseek.com/chat/completions")
MODEL = settings.get("TR_VISION_MODEL", "deepseek-flash")
DPI = 200                 # above this the API discards the extra pixels

PROMPT = """这是同一篇学术论文的全部 %d 页。请拆成排版块，按阅读顺序输出 JSON：

{"blocks":[{"page":1,"role":"body","bbox":[x0,y0,x1,y1],"align":"justify","size_pct":100,"bold":false,"number":"","text":"..."}]}

bbox 是该块在页面上的位置，用 0-1000 的归一化坐标，原点左上。
role 取：header, title, authors, affiliation, abstract, heading, body, caption, figure, table, formula, footnote, footer, reference。
align 取 left / center / justify。
size_pct 是字号相对正文的百分比。
原文里用粗体强调的词（例如列表项开头的术语）在原样抄写时用 **双星号** 包住，不要整段加粗。
number 只对 formula 块有意义：填公式行右端的编号，形如 "(8)"；没有编号填空串。
figure 和 table 块的 text 一律填空串：图里、表里的文字不抄（靠贴图保留）；表格上方的引导句、表格下方的 NOTE 说明各自单独成块（body 或 footnote）。
text 照抄原文，行内公式用 $...$，上下标写清楚。
页尾的词被连字符断开时（页尾写 "condi-"，下页开头写 "tional"），去掉连字符合成一个词即可。除此之外不要跨页合并段落：一段话跨了两页就分成两个块，后一页从这一页自己的第一句开始，不要把上一页已经有的句子再写一遍。
脚注里独占一行的公式同样要单独成块（role 用 formula，bbox 取那一行），不要混进 footnote 的文字里；公式之前的文字一块、之后的文字一块，按阅读顺序排开。
"""


DIRECT_PROMPT = """这是同一篇学术论文的全部 %d 页。请拆成排版块，按阅读顺序输出 JSON，text 直接用简体中文：

{"blocks":[{"page":1,"role":"body","bbox":[x0,y0,x1,y1],"align":"justify","size_pct":100,"bold":false,"number":"","prev_joined":false,"text":"..."}]}

bbox 是该块在页面上的位置，用 0-1000 的归一化坐标，原点左上。
role 取：header, title, authors, affiliation, abstract, heading, body, caption, figure, table, formula, footnote, footer, reference。
align 取 left / center / justify。
size_pct 是字号相对正文的百分比。
number 只对 formula 块有意义：填公式行右端的编号，形如 "(8)"；没有编号填空串。
figure 和 table 块的 text 一律填空串：图里、表里的文字不抄，靠贴图保留；表格上方的引导句、表格下方的 NOTE 说明各自单独成块（body 或 footnote）。

text 的写法：
- 用简体中文（大陆学术书面语）逐句翻译这一块的内容，不要概括、不要合并句子、信息不得增删。
- 行间公式只写一次：写进 formula 块之后，**正文块的 text 里不要再重复这段公式**（正文直接跳过它继续写后面的句子），否则同一行公式会被排两遍。
- 行间公式（单独占一行的公式）**必须单独输出成 role=formula 的块**，bbox 取那一行，text 返回 LaTeX 原文、不要译成中文、不要并进正文段落；行内公式才用 $...$ 留在正文里，上下标写清楚。**每个 $ 必须成对出现**，不许出现落单的 $。
- 数字、年份、变量名、引文编号 [12]、(3)、DOI 原样保留；"Figure 1""Table 2""Eq. (5)" 这类标签保留英文，不要译成"图 1"。
- 人名、机构名、期刊名、文献题名保留英文。
- 原文用粗体强调的片段（例如列表项开头的术语）在译文里用 **双星号** 包住，位置和范围与原文一致，不要整段加粗。
- 页尾的词被连字符断开时（页尾写 "condi-"，下页开头写 "tional"），去掉连字符合成一个词即可。除此之外不要跨页合并段落：一段话跨了两页就分成两个块，后一页从这一页自己的第一句开始，不要把上一页已经有的句子再写一遍。
- 脚注里独占一行的公式同样要单独成块（role 用 formula，bbox 取那一行），不要混进 footnote 的文字里。
- **跨块续写**：一句话如果被栏或页截断（左栏末尾是半句，右栏/下一页开头接着说），把这两个块当作一个句子来译：前一块译到断点为止，后一块从断点接着写，**不要把前一块已经写过的词在第二块里再写一遍**，也不要在第二块开头重新起一个完整句子。
- **公式是一道墙**：如果前一块和后一块之间隔着一个 formula、figure 或 table 块，哪怕前一块没写完（例如“…被定义为：”），也**不算**跨块续写，两块各自独立译。

prev_joined 的填法（回头核对，不是预判）：
从第 3 个块开始，每写完一块，回头看一眼已经写好的前两块——**上上个块**（第 k-2 块）的句子，有没有延续到**上一块**（第 k-1 块）？也就是上一块是不是接着上上块没写完的那句话写的。
- 接上了：本块（第 k 块）的 prev_joined 填 true。
- 上一块是独立起头的：填 false。
- 第 1、2 个块填 false。
这一步是核对已经写出来的两块文字，不要预判本块之后会怎样。
"""

ID_PROMPT = """这是同一篇学术论文的全部 %d 页。每页上已经用红框标出了**全部**待识别的区域，每个框有一个全局唯一的编号（跨页连续）。
请对每个编号输出一条记录：
{"id": 12, "role": "body", "number": "", "text": "..."}

role 取：header, title, authors, affiliation, abstract, heading, body, caption, figure, table, formula, footnote, footer, reference。
number 只对 formula 有意义：填公式行右端的编号，形如 "(8)"；没有编号填空串。
figure 和 table 框的 text 一律填空串（图里、表里的文字不抄，靠贴图保留）。
text 照抄原文，行内公式用 $...$，上下标写清楚。原文里用粗体强调的片段用 **双星号** 包住。
页尾的词被连字符断开时（页尾写 "condi-"，下页开头写 "tional"），去掉连字符合成一个词；除此之外不要跨块合并段落。
脚注里独占一行的公式同样单独输出一条（role 用 formula），不要混进 footnote 的文字里。
**每个编号都必须有一条记录，一个也不能漏**；按阅读顺序输出 JSON 数组，不要输出别的文字。
"""

def _key():
    key = settings.api_key()
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY before requesting vision transcription.")
    return key


def _path(name, direct=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "%s%s.json" % (name, ".direct" if direct else ""))


def _ask(pdf_path, n, direct=False, start=0):
    doc = pymupdf.open(pdf_path)
    prompt = (DIRECT_PROMPT if direct else PROMPT) % n
    if start:
        # a page range, not the whole paper: the pages are labelled with their
        # number in the whole paper so the model answers in absolute pages
        prompt = prompt.replace(
            "这是同一篇学术论文的全部 %d 页。" % n,
            "这是同一篇学术论文的第 %d–%d 页（全篇共 %d 页）；"
            "page 一律填全篇页码，也就是下面标注的页号不要改成 1 开始。"
            % (start + 1, start + n, doc.page_count))
    # per-run amendment, for a paper whose transcription needs a firmer line
    # than the shared prompt takes (VLM_EXTRA="..." python run_paper.py tag)
    extra = os.environ.get("VLM_EXTRA") or ""
    if extra:
        prompt += "\n" + extra
    content = [{"type": "text", "text": prompt}]
    for i in range(n):
        png = doc[start + i].get_pixmap(dpi=DPI).tobytes("png")
        content.append({"type": "text", "text": "第 %d 页：" % (start + i + 1)})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode(),
            "detail": "original"}})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0.0, "max_tokens": 64000,
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    txt = d["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        raise RuntimeError("vision model returned no JSON")
    raw = m.group(0)
    try:
        d = json.loads(raw)
    except ValueError:
        # LaTeX arrives with unescaped backslashes (\lambda) often enough to be
        # worth repairing: double the ones that are not JSON escapes and retry
        # \lambda and \frac: a backslash before a letter is a TeX command, not
        # a JSON escape -- even for \f and \t, which would otherwise turn into
        # a form feed.  Pairs that are already escaped are left alone.
        fixed = re.sub(r'(?<!\\)\\(?=[A-Za-z])', r"\\\\", raw)
        d = json.loads(fixed)
    blocks = d["blocks"]
    # a sentence that runs across two blocks is reported once, at the top, as a
    # pair of block numbers; per-block booleans were never filled in
    for pair in d.get("joins") or []:
        try:
            i, j = int(pair[0]) - 1, int(pair[1]) - 1
        except Exception:                     # noqa: BLE001
            continue
        if 0 <= i < len(blocks) and j == i + 1:
            blocks[i]["join_next"] = True
    # block k carrying prev_joined means the pair (k-2, k-1) is one sentence:
    # the model reports it while looking back at text it has already written
    for k, b in enumerate(blocks):
        if b.get("prev_joined") is True and k >= 2:
            blocks[k - 2]["join_next"] = True
    return blocks


def _numbered(src, boxes_by_page):
    """A copy of the PDF with every box outlined in red and numbered."""
    doc = pymupdf.open(src)
    ids = {}
    n = 0
    for pno in range(doc.page_count):
        page = doc[pno]
        try:
            page.clean_contents()
        except Exception:                     # noqa: BLE001
            pass
        W, H = page.rect.width, page.rect.height
        for rect in boxes_by_page.get(pno + 1, []):
            n += 1
            r = pymupdf.Rect(rect)
            page.draw_rect(r, color=(1, 0, 0), width=1.2)
            page.insert_text((r.x0 + 1.5, max(r.y0 - 1.5, 6.0)), str(n),
                             fontsize=7, color=(1, 0, 0))
            ids[n] = (pno + 1, r, W, H)
    return doc, ids


def _ask_ids(src, boxes_by_page):
    doc, ids = _numbered(src, boxes_by_page)
    content = [{"type": "text", "text": ID_PROMPT % doc.page_count}]
    for i in range(doc.page_count):
        png = doc[i].get_pixmap(dpi=DPI).tobytes("png")
        content.append({"type": "text", "text": "第 %d 页：" % (i + 1)})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode(),
            "detail": "original"}})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0.0, "max_tokens": 64000,
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    txt = d["choices"][0]["message"]["content"] or ""
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", txt)
    if not m:
        raise RuntimeError("vision model returned no JSON")
    raw = m.group(0)
    try:
        data = json.loads(raw)
    except ValueError:
        fixed = re.sub(r"(?<!\\)\\(?=[A-Za-z])", r"\\\\", raw)
        data = json.loads(fixed)
    items = data if isinstance(data, list) else data.get("blocks", [])
    rows = []
    for it in items:
        try:
            k = int(it.get("id"))
        except Exception:                     # noqa: BLE001
            continue
        if k not in ids:
            continue
        page, rect, W, H = ids[k]
        rows.append({
            "page": page,
            "role": it.get("role") or "body",
            "bbox": [rect.x0 / W * 1000.0, rect.y0 / H * 1000.0,
                     rect.x1 / W * 1000.0, rect.y1 / H * 1000.0],
            "number": it.get("number") or "",
            "text": it.get("text") or "",
        })
    missing = [k for k in ids if k not in set(int(x.get("id")) for x in items
                                              if str(x.get("id", "")).isdigit())]
    if missing:
        print("  ! %d of %d boxes got no record" % (len(missing), len(ids)))
    return rows


def blocks_of_ids(name, pdf_path, boxes_by_page, force=False):
    """Structure read off numbered detector boxes (no model-made boxes)."""
    p = os.path.join(CACHE_DIR, "%s.ids.json" % name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force and os.path.exists(p):
        try:
            rows = json.load(open(p, encoding="utf-8"))
            if rows:
                return rows
        except Exception:                     # noqa: BLE001
            pass
    rows = _ask_ids(pdf_path, boxes_by_page)
    json.dump(rows, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


SPLIT_AT = 48      # a whole-paper request is reliable to about this size
FLOOR = 10         # never split a range below this many pages


def _chunk_path(name, start, end):
    return os.path.join(CACHE_DIR, "%s.p%04d-%04d.json" % (name, start + 1, end))


def _shift_pages(rows, start, span):
    """Make chunk page numbers absolute (the model may answer 1..span)."""
    if not start or not rows:
        return rows
    if max(int(r.get("page") or 1) for r in rows) <= span:
        for r in rows:
            r["page"] = int(r.get("page") or 1) + start
    return rows


def _range_rows(name, pdf_path, start, end, direct=False):
    """One page range, halved again if the model's answer comes back broken.

    The whole-paper call is the cheapest when it works (one prompt, the model
    sees the whole article) and impossible when the JSON runs past what a
    reply can hold -- 20_gukellyxiu2020, 80 pages, died at 250k characters.
    So: try the range; on failure split it in half and recurse.  Each range
    that succeeds is cached on its own, so a retry never pays twice.
    """
    cp = _chunk_path(name, start, end)
    if os.path.exists(cp):
        try:
            rows = json.load(open(cp, encoding="utf-8"))
            if rows:
                return rows
        except Exception:                     # noqa: BLE001
            pass
    span = end - start
    print("  [转写] 第 %d–%d 页（%d 页）…" % (start + 1, end, span), flush=True)
    try:
        rows = _ask(pdf_path, span, direct, start=start)
        rows = _shift_pages(rows, start, span)
        json.dump(rows, open(cp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("  [转写] 第 %d–%d 页：%d 块" % (start + 1, end, len(rows)),
              flush=True)
        return rows
    except Exception as e:                    # noqa: BLE001
        if span <= FLOOR:
            raise
        mid = start + span // 2
        print("  ! 第 %d–%d 页失败（%s），拆成 %d + %d"
              % (start + 1, end, str(e)[:60], mid - start, end - mid),
              flush=True)
        rows = (_range_rows(name, pdf_path, start, mid, direct)
                + _range_rows(name, pdf_path, mid, end, direct))
        json.dump(rows, open(cp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return rows


def blocks_of(name, pdf_path, force=False, direct=False):
    """Cached structure of one paper, ordered by (page, reading order)."""
    p = _path(name, direct)
    n = pymupdf.open(pdf_path).page_count
    if not force and os.path.exists(p):
        try:
            rows = json.load(open(p, encoding="utf-8"))
            if rows and max(r.get("page", 0) for r in rows) >= n:
                return _ordered(rows)
        except Exception:                       # noqa: BLE001
            pass
    t0 = time.time()
    if n > SPLIT_AT:
        print("  [转写] %d 页：超过 %d 页整篇调用会超出回复长度，按页段递归转写"
              % (n, SPLIT_AT), flush=True)
        half = (n + 1) // 2
        rows = (_range_rows(name, pdf_path, 0, half, direct)
                + _range_rows(name, pdf_path, half, n, direct))
    else:
        print("  [转写] 把 %d 页交给视觉模型（一次调用，通常 1–5 分钟）…" % n,
              flush=True)
        rows = _ask(pdf_path, n, direct)
    json.dump(rows, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  [转写] 完成：%d 块，%.0f 秒" % (len(rows), time.time() - t0),
          flush=True)
    return _ordered(rows)


def _ordered(rows):
    def key(r):
        pg = int(r.get("page") or 1)
        return (pg, int(r.get("order") or 0))
    return sorted(rows, key=key)
