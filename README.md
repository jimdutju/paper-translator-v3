# Paper Translator v3

面向扫描版与原生文字版学术论文的英文 → 中文 PDF 实验管线。当前主要样本来自量化金融和计量经济学。

视觉模型负责读原文、公式与阅读顺序，DocLayout-YOLO 负责版面框；PDF 的文字层只用于几何。中文正文重新排版，公式经 LaTeX 编译为矢量内容，图和表保留原图。输出页数可以改变。

这是可运行的实验项目。**结构检查通过不代表没有漏译、转写错误或数学错误。** 图表内部保留原图；图注/标签、页眉页脚及其他指定保留区的英文应按用户的翻译边界评价，不能仅因出现英文就判为漏译。

## 本地比较

本次对照对象是本机安装的 **PDFMathTranslate/pdf2zh 1.9.11**（附带 BabelDOC 0.2.33），不是最新 pdf2zh-next。两边的文字翻译后端均为 `deepseek-chat`；v3 另外使用视觉转写模型，输入、提示词、计算量均不相同。部分缓存被复用，运行时间不作为冷启动性能结论。

结果和边界见 [对照记录](docs/COMPARISON.md)。未做完整语义评测，未声称整体优于 pdf2zh。

## 安装

已在 Windows 10 / Python 3.13 验证现有环境（PyMuPDF 1.28.2、NumPy 2.5.0、PyYAML 6.0.3、doclayout-yolo 0.0.4）；建议 Python 3.11–3.13。需要能访问翻译和视觉 API。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item settings.example.json settings.local.json
```

另需自行准备：

- DocLayout-YOLO 的 `doclayout_yolo_docstructbench_imgsz1024.pt`，按其上游说明下载，放入 `models/` 或设置 `TR_YOLO_WEIGHTS`。见 [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)。
- MiKTeX 或 TeX Live 的 `pdflatex`，以及 `amsmath`、`amssymb`、`mathrsfs`、`mathptmx`。让 `pdflatex` 位于 PATH，或设置 `TR_PDFLATEX`。
- 中文常规/粗体和西文数学字体。Windows 默认使用本机宋体、黑体和 Times；Linux 尝试 Noto Serif CJK 与 DejaVu Serif。未验证 Linux 端到端运行。字体文件不随项目分发。
- 一个支持图像输入、能够返回该管线 JSON 格式的视觉模型，以及文字翻译模型。示例保留开发时的模型名；若你的服务没有该模型，必须修改模型和端点。

API key 通过环境变量设置，不要写进源码：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:DEEPSEEK_API_KEY='<你的 API key>'
.venv\Scripts\python translate.py D:\papers\sample.pdf -o output
```

也可以翻译目录、控制并行数和领域：

```powershell
python translate.py D:\papers -j 2 -o output --domain '量化金融与计量经济学'
python translate.py D:\papers\sample.pdf -o output --force
python translate.py D:\papers --dry
```

脚本可接收任意 PDF 路径。tag 简写仅搜索 `papers/`，可用 `TR_PAPERS` 修改。并发会增加 API 调用量；整篇 PDF 页面图像会发给所配置的视觉服务，文字发给翻译服务。

## 配置

环境变量优先于 `settings.local.json`。本地设置文件和缓存不入库。

| 变量 | 用途 |
|---|---|
| `DEEPSEEK_API_KEY` | API 凭证 |
| `TR_VISION_URL` / `TR_VISION_MODEL` | 视觉模型的完整 chat/completions URL 与模型名 |
| `TR_TRANSLATION_URL` / `TR_TRANSLATION_MODEL` | 文字模型的 API 根地址（通常以 `/v1` 结束）与模型名 |
| `TR_YOLO_WEIGHTS` | 检测器权重文件 |
| `TR_PDFLATEX` | pdflatex 可执行文件 |
| `TR_CACHE_DIR` | 私有缓存目录，默认 `.cache/` |
| `TR_FONT_REGULAR` / `TR_FONT_BOLD` | 中文字体文件 |
| `TR_FONT_MATH` / `TR_FONT_MATH_BOLD` / `TR_FONT_MATH_ITALIC` / `TR_FONT_MATH_BOLD_ITALIC` | 数学/西文字体文件 |
| `TR_GEOMETRY=0` | 关闭文字层几何辅助，回退到原有几何路径 |
| `TR_DEFINITION_LINES=0` | 关闭逐行定义列表修复，用于回归对照 |
| `TR_DOMAIN` / `--domain` | 翻译领域 |
| `TR_SYSTEM_FILE` / `--system-file` | 自定义翻译提示词文件 |

模型后端或翻译提示词改变时，文字翻译缓存使用独立键。**视觉模型或视觉提示词改变时，需换用新的 `TR_CACHE_DIR` 或移走对应 `vlm/` 缓存。不同内容的同名 PDF 也应使用独立缓存目录。**

## 质量门禁

每次批量构建生成 `<文件名>.json`，记录 SHA-256、页数、告警、文字压图、空白页、公式源码和占位符残留。出现告警或压图时，CLI 返回非零退出码。

- 只有文件与干净构建记录的 SHA-256 一致时，才自动跳过。
- 翻译缓存中的数学占位符必须完整保留；缺变量的旧译文会按段重译，不必整篇重译。
- 输出公式逐项检查行内 LaTeX 是否与输入一致。这不检验视觉模型有没有先抄错原式。
- 未解决：视觉模型漏掉页面中的某段文字、读序错误、语义误译、复杂跨栏版面、正文页过多留白。图表内部文字的中文化是额外能力，按要求保留英文不属于质量缺陷。

```powershell
python -m unittest discover -s tests -v
```

测试包含丢失数学占位符的旧缓存恢复、定义列表拆行、空白页、构建记录失效和纯公式保护。完整运行还需要上面的字体、LaTeX、检测器和 API。

## 上游与分发范围

本仓库代码采用 [MIT License](LICENSE)。pdf2zh 作为外部比较对象，不随本仓库分发。管线依赖 PyMuPDF、DocLayout-YOLO、NumPy、PyYAML 和本机 LaTeX；使用与分发这些依赖、模型或字体时，应遵守各自许可。

仓库只收录代码、配置示例、测试与对照记录；不收录论文全文、译文全集、API 凭证、模型权重或私人缓存。
