"""Portable configuration. Environment variables override settings.local.json.

The local settings file is deliberately excluded from source control. API keys
belong in the environment or an explicitly selected credentials file.
"""
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
_path = Path(os.environ.get("TR_SETTINGS", HERE / "settings.local.json"))
LOCAL = json.loads(_path.read_text(encoding="utf-8-sig")) if _path.is_file() else {}


def get(name, default=None):
    return os.environ.get(name, LOCAL.get(name, default))


def path(name, default):
    value = Path(str(get(name, default))).expanduser()
    return str((HERE / value).resolve() if not value.is_absolute() else value)


def display_path(value, base):
    """A relative display path when possible, including Windows drive changes."""
    try:
        return os.path.relpath(value, base)
    except ValueError:
        return os.path.abspath(value)


def api_key():
    key = get("DEEPSEEK_API_KEY")
    if key:
        return key
    credentials = get("TR_CREDENTIALS_FILE")
    if credentials:
        import yaml
        with open(credentials, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return (data.get("refs") or {}).get("DEEPSEEK_API_KEY", "")
    return ""


def font_file(name, *candidates):
    configured = get(name)
    choices = [configured] if configured else candidates
    for candidate in choices:
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser())
    raise RuntimeError("Font not found. Set %s to an installed font file." % name)


def pdflatex():
    value = get("TR_PDFLATEX") or shutil.which("pdflatex")
    if not value:
        raise RuntimeError("pdflatex not found. Install MiKTeX/TeX Live or set TR_PDFLATEX.")
    return value
