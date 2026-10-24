r"""Translate one PDF, or every PDF in a folder -- in parallel, with live progress.

    python translate.py 12_famafrench1993            # a tag from papers/
    python translate.py D:\some\paper.pdf            # a single file
    python translate.py D:\some\folder -o D:\out     # every *.pdf under it
    python translate.py D:\some\folder --jobs 4      # four papers at once
    python translate.py D:\some\folder --dry          # just list what would run

Parallel and safe: papers share nothing that is written, except the translation
cache, which is merged under a file lock (trans._save) instead of overwritten.
Each paper still runs in its own process -- the renderer runs out of PDF handles
otherwise.

Output is streamed line by line, prefixed with the paper, so a console with four
papers on it still says which one is talking.  A long silent stage (the one
vision call takes minutes) gets a heartbeat, so silence never looks like a hang.

At the end: a table of what was built, skipped and failed; a failure prints the
last lines of its log and the path of the full one (<out>/logs/<name>.log).
Exit code is non-zero if anything failed.  Nothing is published.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf
import quality
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
TRANS = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_OUT = os.path.join(TRANS, "_out", "v3")

_PRINT = threading.Lock()


def say(msg):
    with _PRINT:
        print(msg, flush=True)


def sources(arg):
    if os.path.isdir(arg):
        out = []
        for dp, _dirs, files in os.walk(arg):
            for fn in sorted(files):
                if fn.lower().endswith(".pdf") and "_ocr" not in fn.lower():
                    out.append(os.path.join(dp, fn))
        return sorted(out)
    return [arg]


def audit(pdf):
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


def build(src, out_dir, log_path, tag, domain=None, system_file=None):
    """Run one paper in its own process, streaming its output here."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TR_OUT=out_dir)
    if domain:
        env["TR_DOMAIN"] = domain
    if system_file:
        env["TR_SYSTEM_FILE"] = system_file
    cmd = [sys.executable, "-u", "-c",
           "import vlm_yolo; vlm_yolo.build(%r)" % src]
    t0 = time.time()
    lines = []
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=HERE, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        # read on a thread: a blocking readline never lets the heartbeat fire
        q = queue.Queue()

        def pump():
            for line in proc.stdout:
                q.put(line)
            q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        last = time.time()
        while True:
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None and q.empty():
                    break
                if time.time() - last > 20:
                    say("  %-10s … %ds 无新输出（转写阶段通常 1–5 分钟）"
                        % (tag, int(time.time() - t0)))
                    last = time.time()
                continue
            if line is None:
                break
            line = line.rstrip()
            lines.append(line)
            log.write(line + "\n")
            log.flush()
            say("  %-10s %s" % (tag, line.strip()))
            last = time.time()
    return proc.wait(), lines, time.time() - t0


def run_one(src, name, out_dir, logs, force, tag, domain=None, system_file=None):
    pdf = os.path.join(out_dir, "【翻译】%s_vy.pdf" % name)
    if os.path.exists(pdf) and not force and quality.record_is_clean(pdf, name):
        return ("skip", name, pdf, None)
    log_path = os.path.join(logs, name + ".log")
    rc, lines, secs = build(src, out_dir, log_path, tag, domain, system_file)
    if rc or not os.path.exists(pdf):
        return ("fail", name, pdf, (rc, lines[-14:], log_path))
    record = quality.write_record(name, pdf, lines, secs)
    pages, on_image, warn = record["pages"], record["text_on_image"], record["warnings"]
    return ("ok", name, pdf, (pages, on_image, warn, secs))


def main():
    ap = argparse.ArgumentParser(
        description="把 PDF（或一个文件夹里的 PDF）翻成中文 PDF，进度实时显示。")
    ap.add_argument("input", nargs="+", help="PDF 路径 / 目录 / papers 里的 tag")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="产物目录")
    ap.add_argument("-j", "--jobs", type=int, default=3,
                    help="同时跑几篇（默认 3；每篇一个进程）")
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    ap.add_argument("--domain", help="这篇论文的领域（写进提示词第一行），覆盖 domains.json")
    ap.add_argument("--system-file", help="整段替换系统提示词的文件（.txt）")
    ap.add_argument("--dry", action="store_true", help="只列出要做什么")
    a = ap.parse_args()

    out_dir = os.path.abspath(a.out)
    logs = os.path.join(out_dir, "logs")
    os.makedirs(logs, exist_ok=True)

    jobs = []
    for arg in a.input:
        jobs.extend(sources(arg))
    seen, uniq = set(), []
    for s in jobs:
        k = os.path.abspath(s) if os.path.exists(s) else s
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    jobs = uniq

    def tag_of(src):
        base = os.path.basename(src)[:-4] if src.lower().endswith(".pdf") else src
        return base.split("_")[0]

    def name_of(src):
        return os.path.basename(src)[:-4] if src.lower().endswith(".pdf") else src

    # per-paper field: --domain wins, then domains.json by file name or tag
    dm = {}
    for p in (os.path.join(HERE, "domains.json"),
              os.path.join(os.path.dirname(a.out), "domains.json")):
        if os.path.exists(p):
            try:
                import json as _json
                with open(p, encoding="utf-8") as f:
                    dm.update(_json.load(f))
            except Exception:                     # noqa: BLE001
                pass

    def domain_of(src):
        if a.domain:
            return a.domain
        return dm.get(name_of(src)) or dm.get(tag_of(src))

    if a.dry:
        say("将处理 %d 篇：" % len(jobs))
        for s in jobs:
            say("   %-28s %s" % (tag_of(s), s))
        return 0

    say("产出目录 %s" % out_dir)
    say("共 %d 篇，同时 %d 篇（每篇一个进程）\n" % (len(jobs), max(1, a.jobs)))
    t_all = time.time()
    done, skipped, failed = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {}
        for s in jobs:
            base = os.path.basename(s)[:-4] if s.lower().endswith(".pdf") else s
            futs[ex.submit(run_one, s, base, out_dir, logs, a.force,
                           tag_of(s), domain_of(s), a.system_file)] = (s, base)
        for fut in as_completed(futs):
            src, base = futs[fut]
            try:
                kind, name, pdf, extra = fut.result()
            except Exception as e:                    # noqa: BLE001
                failed.append((base, "?", [str(e)], ""))
                say("  !! %s 崩溃: %s" % (base, e))
                continue
            if kind == "skip":
                skipped.append((name, pdf))
                say("\n== %s 已有产物，跳过（--force 可重建）" % name)
            elif kind == "fail":
                rc, tail, log_path = extra
                failed.append((name, rc, tail, log_path))
                say("\n== %s 失败（退出码 %s），日志 %s" % (name, rc, log_path))
                for line in tail:
                    say("     | " + line)
            else:
                pages, on_image, warn, secs = extra
                done.append((name, pages, on_image, len(warn), secs, pdf))
                say("\n== %s：%d 页，压图 %d，告警 %d，%.0f 秒"
                    % (name, pages, on_image, len(warn), secs))

    say("\n" + "=" * 72)
    say("完成 %d 篇，跳过 %d 篇，失败 %d 篇，总用时 %.0f 秒"
        % (len(done), len(skipped), len(failed), time.time() - t_all))
    for name, pages, on_image, nw, secs, pdf in sorted(done):
        say("  %-6s %-26s %3d 页  压图 %d  告警 %d  %4.0fs  %s"
            % ("REVIEW" if on_image or nw else "OK", name, pages, on_image,
               nw, secs, settings.display_path(pdf, TRANS)))
    for name, pdf in sorted(skipped):
        say("  SKIP  %-26s %s" % (name, settings.display_path(pdf, TRANS)))
    for name, rc, tail, log_path in failed:
        say("  FAIL  %-26s rc=%s  日志 %s" % (name, rc, log_path))
    if failed:
        say("\n有失败。可以把上面 FAIL 的日志发给我，或者单独重跑：")
        say("  python translate.py <那个 PDF 或 tag> --force")
    return 1 if failed or any(row[2] or row[3] for row in done) else 0


if __name__ == "__main__":
    sys.exit(main())
