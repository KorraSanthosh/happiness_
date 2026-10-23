#!/usr/bin/env python3
"""
Render docs/GlucoTwin_20_Minute_Speaking_Script.md to docs/GlucoTwin_20_Minute_Speaking_Script.pdf.

A small Markdown subset (headings, paragraphs, bold/italic/code, tables, horizontal rules) is converted to
print-styled HTML, then printed to PDF with a headless Chromium browser (Chrome, Brave or Chromium).

Run from the repo root:  python3 docs/presentation/build_script_pdf.py [--browser "/path/to/browser"]
"""
import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "GlucoTwin_20_Minute_Speaking_Script.md"
OUT = ROOT / "docs" / "GlucoTwin_20_Minute_Speaking_Script.pdf"
BROWSERS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Chromium.app/Contents/MacOS/Chromium", "google-chrome", "chromium", "chromium-browser"]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; line-height: 1.55; color: #0f2a35; }
h1 { font-size: 20pt; margin: 0 0 6pt; color: #0c4c58; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding: 6pt 10pt; background: #e6f1f4; border-left: 4pt solid #0f8f8a;
     border-radius: 4pt; break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
p.say { margin-left: 10pt; }
p.cue { background: #f6fafb; border: 1px solid #d6e6ea; border-radius: 4pt; padding: 4pt 8pt; font-size: 10pt; break-inside: avoid; }
p.transition { font-style: italic; color: #46616d; }
code { font-family: Menlo, Consolas, monospace; font-size: 9pt; background: #eef4f6; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 10pt; break-inside: avoid; }
th, td { border: 1px solid #c9dade; padding: 3pt 6pt; text-align: left; }
th { background: #0c4c58; color: #fff; }
hr { border: 0; border-top: 1px solid #c9dade; margin: 12pt 0; }
"""


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<i>\1</i>", t)
    return t


def to_html(md: str) -> str:
    out, para, table, mode = [], [], [], "plain"

    def flush():
        nonlocal para
        if para:
            text = " ".join(para)
            cls = mode
            if text.startswith(("**Show:**", "**Do:**")):
                cls = "cue"
            elif text.startswith("**Transition:**"):
                cls = "transition"
            out.append(f'<p class="{cls}">{inline(text)}</p>')
            para = []

    def flush_table():
        nonlocal table
        if table:
            rows = [r for r in table if not re.match(r"^\|[\s|:-]+\|$", r)]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            h = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>")
            table = []

    for line in md.splitlines():
        if line.startswith("|"):
            flush(); table.append(line); continue
        flush_table()
        if line.startswith("# "):
            flush(); out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            flush(); mode = "plain"; out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip() == "---":
            flush(); out.append("<hr>")
        elif not line.strip():
            flush()
        elif line.startswith("**Say:**"):
            flush(); mode = "say"; out.append(f'<p class="plain">{inline(line)}</p>')
        else:
            if line.startswith(("**Show:**", "**Do:**", "**Transition:**")):
                flush()
            para.append(line.strip())
    flush(); flush_table()
    return f"<!doctype html><html><head><meta charset='utf-8'><title>GlucoTwin AI — 20-Minute Speaking Script</title><style>{CSS}</style></head><body>{''.join(out)}</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser")
    a = ap.parse_args()
    browser = a.browser or next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        raise SystemExit("No Chrome/Brave/Chromium found; pass --browser")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "script.html"
        page.write_text(to_html(SRC.read_text(encoding="utf-8")), encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", page.as_uri()], check=True, capture_output=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
