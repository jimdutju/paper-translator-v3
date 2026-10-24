"""Translation layer: DeepSeek chat API, glossary per paper, disk cache.

One API call carries several consecutive paragraphs plus the paper title, the
current section heading and a glossary, so pronouns, connectives and technical
terms stay consistent inside a page.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import cache
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
# entries are keyed by PROMPT_VERSION + the English, so both versions can share
# one file without ever reading each other's answers
CACHE = cache.ZH
GLOSS = cache.GLOSS
PROMPT_VERSION = "v2.6"   # v2.6: bold placeholders are kept verbatim

API = settings.get("TR_TRANSLATION_URL", "https://api.deepseek.com/v1").rstrip("/")
KEY = settings.api_key()

BASE_GLOSSARY = """volatility 波动率 | returns 收益率 | stylized facts 典型事实 |
heteroskedasticity 异方差 | homoskedasticity 同方差 | conditional variance 条件方差 |
covariance 协方差 | correlation 相关系数 | distribution 分布 | density 密度 |
estimator 估计量 | consistent 一致的 | asymptotically normal 渐近正态 |
stationarity 平稳性 | unit root 单位根 | martingale 鞅 | random walk 随机游走 |
diffusion 扩散 | Brownian motion 布朗运动 | stochastic volatility 随机波动率 |
Cauchy 柯西 | Gauss 高斯 | Levy 莱维 | Pareto 帕累托 | mean reversion 均值回复 | autocorrelation 自相关 | kurtosis 峰度 | skewness 偏度 |
tail index 尾指数 | stable distribution 稳定分布 | Pareto 帕累托 |
ARCH 自回归条件异方差 | GARCH 广义自回归条件异方差 | log-likelihood 对数似然 |
maximum likelihood 极大似然 | Lagrange multiplier 拉格朗日乘子 |
speculative prices 投机性价格 | price changes 价格变动 | stock returns 股票收益率 |
portfolio 投资组合 | factor model 因子模型 | risk premium 风险溢价 |
market microstructure 市场微观结构 | bid 买价 | ask 卖价 | spread 价差 |
insider trading 内幕交易 | informed trader 知情交易者 | liquidity 流动性 |
order flow 订单流 | adverse selection 逆向选择 | inventory 存货 |
realized volatility 已实现波动率 | high-frequency 高频 | semi-variance 半方差 |
eigenvalue 特征值 | shrinkage 收缩 | sample covariance matrix 样本协方差矩阵 |
out-of-sample 样本外 | in-sample 样本内 | standard error 标准误 |
null hypothesis 原假设 | test statistic 检验统计量 | p-value p 值 |
chaos 混沌 | attractor 吸引子 | nonlinearity 非线性 | Lyapunov exponent 李雅普诺夫指数 |
machine learning 机器学习 | neural network 神经网络 | cross-validation 交叉验证 |
asset pricing 资产定价 | expected return 期望收益率 | book-to-market 账面市值比 |
cross-section 横截面 | time series 时间序列 | regression 回归 |
local linearization 局部线性化 | likelihood ratio 似然比 |
"""

DEFAULT_DOMAIN = "量化金融与计量经济学"

SYSTEM_BODY = """你是{domain}的专业学术翻译，把英文论文译成简体中文（大陆学术书面语）。

硬性要求：
1. 只翻译"正文"。不要添加解释、不要概括、不要合并句子。逐句对应，信息不得增删。
2. 数字、年份、变量名、公式符号原样保留。**原文里 `$...$` 包起来的 LaTeX 公式必须原样照抄**（不改写、不翻译、不删 `$`、不在内部加空格）；没有 `$...$` 的行内数学用 Unicode 表示（如 σ²、xₜ、N^(-1/2)、√T），不要自己新增 `$`。禁止 markdown。
3. 引文编号 [12]、(3)、页码、DOI 原样保留；"Figure 1""Table 2""Eq. (5)" 这类标签也原样保留英文，不要改成"图 1"。
4. 人名、机构名、期刊名、文献题名、专有名词保留英文原文；书名/论文题名保持英文并加中文引号「」不必，直接用英文引号。
5. 脚注上标记号形如 ⟦1⟧、⟦2⟧ 必须原样保留在句中对应的位置。加粗标记形如 ⟦B1⟧（成对出现，包住要加粗的片段）同样是占位符：**必须原样照抄，不要换成星号、不要删掉**，只翻译两个相同 ⟦B1⟧ 之间的文字。公式占位符形如 ⟦M1⟧、⟦M2⟧，必须原样保留、不得改写、不得删除；但它是普通成分，周围的英文（如 and、or、where）要照常译成中文，占位符前后要有中文标点或空格。
6. 章节编号原样保留：I.、II.、A.、B.、1.、2.1 等不要改写、不要换成阿拉伯数字。
7. 术语按给定术语表统一；没有给出的专业术语用学界通行译法。
8. 中文里不要出现无意义的空格；不要输出繁体字；不要输出"以下是译文"之类的话。
9. 遇到被版面截断的残句（结尾没有句号），照原样译出，不要在末尾补句号。
10. 原文里被 **双星号** 包住的片段是加粗强调（常见于列表项的术语），译文里要把对应片段同样用 ** 包住，位置和范围与原文一致；没有 ** 的地方不要自己加。

输出严格的 JSON 对象：{"t":[{"i":1,"zh":"…"},{"i":2,"zh":"…"}]}，i 与输入编号一一对应。"""


def _domain():
    """The field named in the first line of the prompt.

    The pipeline is no longer used for one field only.  TR_DOMAIN names the
    field for this paper (translate.py --domain, or domains.json by tag);
    TR_SYSTEM_FILE replaces the whole prompt when a paper needs more than that.
    """
    return (os.environ.get("TR_DOMAIN") or "").strip() or DEFAULT_DOMAIN


def _system_prompt():
    path = os.environ.get("TR_SYSTEM_FILE") or ""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return SYSTEM_BODY.replace("{domain}", _domain())


SYSTEM = _system_prompt()

# a custom prompt must not read the default prompt's cache: the same English
# translated under a different field would silently reuse the old answer
SYSTEM_TAG = "" if SYSTEM == SYSTEM_BODY.replace("{domain}", DEFAULT_DOMAIN) else (
    "sys:" + hashlib.sha1(SYSTEM.encode("utf-8")).hexdigest()[:8])
_model = settings.get("TR_TRANSLATION_MODEL", "deepseek-chat")
if _model != "deepseek-chat" or API != "https://api.deepseek.com/v1":
    SYSTEM_TAG += "|backend:" + hashlib.sha1((_model + "|" + API).encode()).hexdigest()[:8]


MATH_RE = re.compile(r"\$([^$]+)\$")
PH = "\u27e6M%d\u27e7"


def protect_math(text):
    """Hide $latex$ behind ⟦M1⟧ so the model cannot reformat it."""
    keep = []

    def repl(m):
        keep.append(m.group(1).strip())
        return PH % len(keep)

    return MATH_RE.sub(repl, text), keep


GLUE = re.compile(r"(?<=[\u0370-\u03ff])\s*(and|or)\s*(?=[\u0370-\u03ff])")


TERM_FIX = [
    (re.compile(r"Cauchy\s*律"), "柯西律"),
    (re.compile(r"Gauss\s*律"), "高斯律"),
    (re.compile(r"Cauchy\s*分布"), "柯西分布"),
    (re.compile(r"Gauss\s*分布"), "高斯分布"),
]


def unify_terms(zh):
    """Both halves of a pair must be in the same language."""
    for pat, rep in TERM_FIX:
        zh = pat.sub(rep, zh)
    return zh


def normalize_punct(zh):
    """Make the two sides of a bracket agree, and use full-width brackets when
    they wrap Chinese."""
    out = []
    i = 0
    while i < len(zh):
        if zh[i] in "(（":
            j = zh.find(")", i + 1)
            k = zh.find("）", i + 1)
            if k != -1 and (j == -1 or k < j):
                end, close = k, "）"
            else:
                end, close = j, ")"
            if end == -1:
                out.append(zh[i])
                i += 1
                continue
            inner = zh[i + 1:end]
            cjk = any("\u4e00" <= c <= "\u9fff" for c in inner)
            op = "（" if cjk else "("
            cl = "）" if cjk else ")"
            out.append(op + inner + cl)
            i = end + 1
            continue
        out.append(zh[i])
        i += 1
    return "".join(out)


def fix_glue(zh):
    """τandβ -> τ 和 β: English connectives stranded between two symbols."""
    return GLUE.sub(lambda m: " 和 " if m.group(1) == "and" else " 或 ", zh)


def restore_math(zh, keep):
    def back(m):
        i = int(m.group(1))
        return "$" + keep[i - 1] + "$" if 0 < i <= len(keep) else m.group(0)
    out = re.sub(r"\u27e6M(\d+)\u27e7", back, zh)
    if keep and "\u27e6M" not in zh and "\u27e6" not in out:
        # the model sometimes drops the bracket glyphs and leaves a bare "M1";
        # the number still marks the slot, so the formula goes back there
        out = re.sub(r"M(\d{1,2})(?!\d)", back, out)
    return out


def _post(prompt, max_tokens=8000, timeout=300, temp=0.2):
    if not KEY:
        raise RuntimeError("no DEEPSEEK_API_KEY")
    body = {
        "model": settings.get("TR_TRANSLATION_MODEL", "deepseek-chat"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        API + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def _json_of(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_repair_json(raw))


def _repair_json(raw):
    r"""Fix the two ways a JSON reply from the model usually breaks.

    A translation that mentions LaTeX comes back with a bare backslash
    (\sigma), which is not a legal JSON escape -- 21 lost a whole batch to
    "Invalid \escape".  A trailing comma before } or ] is the other one.
    """
    # \frac, \nu, \beta: JSON reads \f as formfeed, \n as newline -- a
    # letter right after the control escape means it was LaTeX
    s = re.sub(r'\\([bfnrtu])(?=[A-Za-z])', r"\\\\\1", raw)
    s = re.sub(r'\\(?!["\\\\/bfnrtu])', r"\\\\", s)
    s = re.sub(r'\\(u)(?![0-9a-fA-F]{4})', r"\\\\u", s)
    s = re.sub(r",\s*([}\]])$", r"\1", s)
    s = re.sub(r",\s*([}\]])$", r"\1", s, flags=re.M)
    return s


class Translator:
    def __init__(self, workers=4, verbose=True):
        self.cache = _load(CACHE)
        self.gloss = _load(GLOSS)
        self.lock = threading.Lock()
        self.workers = workers
        self.verbose = verbose
        self.calls = 0
        self.chars_in = 0
        self.chars_out = 0

    # -- glossary -----------------------------------------------------------
    def paper_glossary(self, key, sample, title):
        if key in self.gloss:
            return self.gloss[key]
        prompt = (
            "下面是一篇量化金融/计量经济学论文的开头。请列出 25 个最关键的学术术语"
            "（英文术语 → 简体中文译名），只输出 JSON：{\"terms\":[[\"en\",\"zh\"],…]}。\n"
            "术语必须是论文中反复出现、且中文译名有讲究的词（如 heteroskedasticity、"
            "martingale、realized volatility），不要列 the/and 这类词。\n\n"
            "标题：" + title + "\n\n正文片段：\n" + sample[:3000])
        try:
            d = _json_of(_post(prompt, max_tokens=1500, temp=0.1))
            terms = [(a, b) for a, b in d.get("terms", []) if a and b][:40]
        except Exception as e:
            if self.verbose:
                print("  glossary failed:", e, file=sys.stderr)
            terms = []
        self.gloss[key] = terms
        with self.lock:
            _save(GLOSS, self.gloss)
        return terms

    # -- translation --------------------------------------------------------
    def translate(self, groups, title, key):
        """groups: list of dict(en=str, section=str). Returns list of zh strings."""
        todo = [g for g in groups if _hash(g["en"]) not in self.cache
                or not _math_slots_valid(self.cache[_hash(g["en"])], g["en"])]
        # A formerly cached answer may have silently dropped a variable. Retry
        # just those paragraphs, preserving all other expensive translations.
        for g in todo:
            self.cache.pop(_hash(g["en"]), None)
        if not todo:
            # everything cached: still unhide the formulas, or the page shows
            # "M1" wherever a symbol belongs
            return [restore_math(self.cache.get(_hash(g["en"]), ""),
                                 g.get("keep") or [])
                    for g in groups]
        # the glossary is now assembled per request: only the terms that the
        # batch actually contains are sent (see glossary_for)
        terms = self.paper_glossary(key, "\n".join(g["en"] for g in groups[:6]), title)

        chunks = _chunk(todo, max_items=6, max_chars=3600)
        if self.verbose:
            print("  [翻译] %d 组 → %d 个请求" % (len(todo), len(chunks)),
                  flush=True)
        with self.lock:
            self.calls += len(chunks)
        results = {}
        threads = []
        jobs = [(ch, terms, title) for ch in chunks]

        def work(job):
            ch, tm, ti = job
            try:
                got = self._batch(ch, tm, ti)
            except Exception as e:                # noqa: BLE001
                if self.verbose:
                    import traceback
                    print("  batch failed (%d): %s\n%s"
                          % (len(ch), e, traceback.format_exc()[-900:]),
                          file=sys.stderr, flush=True)
                got = [None] * len(ch)
            with self.lock:
                for g, zh in zip(ch, got):
                    if zh:
                        results[g["en"]] = zh

        for job in jobs:
            t = threading.Thread(target=work, args=(job,))
            t.start()
            threads.append(t)
            while sum(1 for x in threads if x.is_alive()) >= self.workers:
                time.sleep(0.05)
        last, shown = 0.0, -1
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            now = time.time()
            if self.verbose and now - last > 5.0 and len(results) != shown:
                last, shown = now, len(results)
                print("  [翻译] %d/%d 组已回" % (len(results), len(todo)),
                      flush=True)
        for t in threads:
            t.join()
        # anything still missing gets one more, unbatched attempt
        for rnd in range(3):
            # missing *or* still bad (English left behind, no Chinese): the
            # second kind is what the batch could not fix in place
            missing = [g for g in todo
                       if not results.get(g["en"])
                       or _bad(results[g["en"]], g["en"])]
            if not missing:
                break
            # longest first: that is where an untranslated sentence costs the
            # most, and the cap keeps a whole-paper retry from running for hours
            missing.sort(key=lambda g: -len(g["en"]))
            missing = missing[:24]
            if self.verbose:
                print("  retry round %d: %d groups" % (rnd + 1, len(missing)),
                      flush=True)
            for g in missing:
                zh = self._retry_one(g, terms, title)
                if zh:
                    results[g["en"]] = zh
            time.sleep(1.0)
        with self.lock:
            for en, zh in results.items():
                if zh and _math_slots_valid(zh, en):
                    self.cache[_hash(en)] = zh
            _save(CACHE, self.cache)
        # Put the hidden formulas back here, not in every caller: the cache
        # stores the protected text, and whoever reads it must get the real
        # thing.  A caller that forgets prints "M6" where the page had a chi.
        out = []
        for g in groups:
            zh = self.cache.get(_hash(g["en"]), "")
            keep = g.get("keep")
            if keep is None and "⟦M" in (g.get("en") or ""):
                # a caller passed the protected text without its keep list:
                # the placeholders can never be restored, and the page will
                # show "M1" where a formula belongs
                print("  ! group without keep list: %r" % g["en"][:40], file=sys.stderr)
                keep = []
            out.append(restore_math(zh, keep or []) if zh else "")
        return out

    def _batch(self, chunk, terms, title):
        parts = []
        section = ""
        for i, g in enumerate(chunk, 1):
            if g.get("section") and not section:
                section = g["section"]
            parts.append("%d. %s" % (i, g["en"]))
        gloss = glossary_for("\n".join(g["en"] for g in chunk), terms)
        ctx = ("\n本批段落所在章节（仅供理解上下文，不要翻译）：" + section) if section else ""
        head = "\n术语表：" + gloss if gloss else ""
        prompt = (SYSTEM + "\n\n论文标题：" + title + head + ctx +
                  "\n\n待译段落（同一篇论文的连续段落，注意前后衔接）：\n\n" +
                  "\n\n".join(parts))
        txt = _post(prompt, max_tokens=8000)
        d = _json_of(txt)
        got = {}
        for it in d.get("t", []):
            try:
                got[int(it["i"])] = str(it["zh"]).strip()
            except Exception:
                pass
        res = []
        for i, g in enumerate(chunk, 1):
            zh = got.get(i)
            if not zh:
                # a *missing* paragraph is worth one immediate retry; a bad
                # but present one is left to the rounds below, because doing
                # both here nested four attempts inside every worker and 21
                # spent minutes per batch on it
                zh = self._retry_one(g, terms, title)
            res.append(zh)
        return res

    def _retry_one(self, g, terms, title):
        gloss = glossary_for(g["en"], terms)
        head = "\n术语表：" + gloss if gloss else ""
        prompt = (SYSTEM + "\n\n论文标题：" + title + head +
                  "\n\n下面这段英文没有译好（有整段英文残留或漏译），请完整、逐句地译成中文，"
                  "只输出 JSON {\"zh\":\"…\"}：\n\n" + g["en"])
        for attempt in range(3):
            try:
                d = _json_of(_post(prompt, max_tokens=3000))
                zh = str(d.get("zh", "")).strip()
                if zh and not _bad(zh, g["en"]):
                    return zh
            except Exception:
                pass
            time.sleep(1.0 * (attempt + 1))
        # still failing: split a long paragraph in two and try the halves
        en = g["en"]
        if len(en) > 400:
            parts = re.split(r"(?<=[.;:])\s+", en)
            if len(parts) >= 2:
                mid = len(parts) // 2
                out = []
                for half in (" ".join(parts[:mid]), " ".join(parts[mid:])):
                    if not half.strip():
                        continue
                    try:
                        d = _json_of(_post(SYSTEM + "\n\n论文标题：" + title +
                                           "\n术语表：" + gloss +
                                           "\n\n待译段落：\n\n1. " + half, max_tokens=3000))
                        zh = str(d["t"][0]["zh"]).strip() if d.get("t") else ""
                    except Exception:
                        zh = ""
                    if not zh:
                        return None
                    out.append(zh)
                return "".join(out)
        return None


# --- glossary ------------------------------------------------------------- #
# Only the terms a request actually contains are sent with it: a long list
# costs nothing then, and every entry that is sent is one the model can use.

def _gloss_pairs(text):
    """'volatility 波动率 | returns 收益率' -> [(en, zh)]"""
    out = []
    for chunk in re.split(r"[|\n]", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        i = next((k for k, c in enumerate(chunk)
                  if "\u4e00" <= c <= "\u9fff"), -1)
        if i <= 0:
            continue
        out.append((chunk[:i].strip(), chunk[i:].strip()))
    return out


_RE_CACHE = {}


def _term_re(en):
    """Word-boundary regex for one term, tolerant of plurals and hyphens."""
    key = en.lower()
    if key in _RE_CACHE:
        return _RE_CACHE[key]
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", en) if p]
    rx = None
    if parts:
        body = r"[\s\-]*".join(parts)
        rx = re.compile(r"(?<![A-Za-z0-9])" + body +
                        r"(?:s|es|ed|ing)?(?![A-Za-z0-9])", re.I)
    _RE_CACHE[key] = rx
    return rx


_BASE_PAIRS = None


def _base_pairs():
    """The hand-kept list plus the terms the translated papers agree on."""
    global _BASE_PAIRS
    if _BASE_PAIRS is not None:
        return _BASE_PAIRS
    pairs = _gloss_pairs(BASE_GLOSSARY)
    try:
        seen = {}
        for _k, rows in _load(GLOSS).items():
            for en, zh in rows:
                seen.setdefault(en.strip().lower(), []).append(zh.strip())
        for en, zhs in seen.items():
            if len(zhs) < 3:
                continue
            best = max(set(zhs), key=zhs.count)
            if zhs.count(best) >= 0.6 * len(zhs):
                pairs.append((en, best))
    except Exception:                     # noqa: BLE001
        pass
    out, have = [], set()
    for en, zh in pairs:
        k = en.lower()
        if not en or k in have:
            continue
        if len(en.replace(" ", "")) < 3 and not en.isupper():
            continue
        have.add(k)
        out.append((en, zh))
    _BASE_PAIRS = out
    return out


def glossary_for(text, extra=()):
    """The glossary lines for one request: only what the text contains."""
    pairs = list(_base_pairs())
    have = {en.lower() for en, _zh in pairs}
    for en, zh in extra or ():
        k = (en or "").strip().lower()
        if k and k not in have:
            have.add(k)
            pairs.append((en.strip(), zh.strip()))
    hits = []
    for en, zh in pairs:
        rx = _term_re(en)
        if rx is not None and rx.search(text):
            hits.append("%s %s" % (en, zh))
    return " | ".join(hits)


def _hash(s):
    # the field named in the prompt is part of the key: the same English set
    # under another field must not silently reuse this answer.  The default
    # field keeps the old key format, or every existing cache entry would be
    # orphaned and the whole corpus re-translated.
    if SYSTEM_TAG:
        return hashlib.sha1(("%s|%s|%s" % (PROMPT_VERSION, SYSTEM_TAG, s))
                            .encode("utf-8")).hexdigest()
    return hashlib.sha1(("%s|%s" % (PROMPT_VERSION, s))
                        .encode("utf-8")).hexdigest()


def _load(p):
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(p, d):
    """Merge this paper's new entries into the shared cache, under a lock.

    Several papers can be translating at once; each process only holds the
    entries it made, and a plain overwrite would silently drop the others'
    work.  The keys are English hashes, so merging is a dict update.  The lock
    is a file, because that is the only mutex both processes share; a lock left
    behind by a crash is broken after two minutes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    lock, mine = p + ".lock", False
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            mine = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 120:
                    os.remove(lock)
                    continue
            except OSError:
                pass
            time.sleep(0.1)
    if not mine:
        raise TimeoutError("Could not acquire translation cache lock: " + lock)
    try:
        cur = _load(p)
        cur.update(d)
        tmp = "%s.%d.tmp" % (p, os.getpid())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False)
        os.replace(tmp, p)
        return cur
    finally:
        if mine:
            try:
                os.remove(lock)
            except OSError:
                pass


def _chunk(items, max_items, max_chars):
    out, cur, n = [], [], 0
    for it in items:
        if cur and (len(cur) >= max_items or n + len(it["en"]) > max_chars):
            out.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += len(it["en"])
    if cur:
        out.append(cur)
    return out


def _math_slots_valid(zh, en):
    expected = Counter(re.findall(r"⟦M(\d+)⟧", en or ""))
    actual = Counter(re.findall(r"⟦M(\d+)⟧", zh or ""))
    # Existing caches sometimes use the documented bare-M restoration form.
    for n in re.findall(r"(?<![A-Za-z0-9⟦])M(\d{1,2})(?!\d|⟧)", zh or ""):
        actual[n] += 1
    return expected == actual


def _bad(zh, en):
    """Detect a translation that left English behind or dropped the content."""
    if not zh:
        return True
    if not _math_slots_valid(zh, en):
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff]", zh))
    if cjk < 0.15 * len(re.sub(r"\s", "", zh)):
        return True
    # A short block may be a header, a URL, a copyright line or a list of
    # author names -- all of which stay in English by rule 4.  Only a real
    # paragraph can hide an untranslated sentence.
    if len(en) < 400:
        return False
    return re.search(r"(?:[A-Za-z][A-Za-z'\-]+[ ,]+){8,}", zh) is not None
