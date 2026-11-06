"""마크다운 문서를 A4 PDF로 만든다 (기획서·결과 설명자료용).

사용법:
  .venv/bin/python tools/build_pdf.py docs/proposal/기획서.md            # → outputs/docs/기획서.pdf
  .venv/bin/python tools/build_pdf.py docs/proposal/기획서.md -o out.pdf

필요: pip install -e ".[docs]" (markdown), Google Chrome 또는 Microsoft Edge (헤드리스 인쇄)
인터넷 연결은 쓰지 않는다. 글꼴은 시스템 한글 글꼴을 쓴다.
"""
import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
html { font-size: 9.6pt; }
body { font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
       color: #1a1a1a; line-height: 1.55; margin: 0; background: #fff; }
h1 { font-size: 20pt; margin: 0 0 2pt; color: #0b3d5c; }
h3:first-of-type { margin-top: 0; }
h2 { font-size: 13.5pt; color: #0b3d5c; border-bottom: 1.5pt solid #0b3d5c;
     padding-bottom: 2pt; margin: 16pt 0 6pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 16pt; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt; font-size: 8.6pt;
        break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 0.6pt solid #b8c4cc; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #e8f0f5; }
pre { background: #f5f7f8; border: 0.6pt solid #d5dde2; padding: 5pt; font-size: 7.4pt;
      line-height: 1.3; white-space: pre; overflow: hidden; break-inside: avoid;
      font-family: "D2Coding", "Menlo", "Consolas", monospace; }
code { font-family: "Menlo", "Consolas", monospace; font-size: 8.4pt; }
blockquote { margin: 5pt 0; padding: 3pt 8pt; border-left: 3pt solid #9fb7c6; color: #333; }
hr { border: 0; border-top: 0.6pt solid #ccc; margin: 10pt 0; }
.flow { display: flex; align-items: stretch; gap: 0; margin: 6pt 0 3pt; break-inside: avoid; break-after: avoid; }
.band { break-after: avoid; }
.flow .step { flex: 1 1 0; border: 0.8pt solid #7f9bb0; border-radius: 3pt; padding: 4pt 4pt;
              background: #f4f8fb; display: flex; flex-direction: column; gap: 2pt; }
.flow .step.key { background: #fff4e0; border-color: #c98a2b; }
.flow .step.out { background: #e6f2ea; border-color: #4f8a63; }
.flow .step b { font-size: 8.4pt; line-height: 1.25; }
.flow .step span { font-size: 7.6pt; color: #333; line-height: 1.3; }
.flow .step i { font-style: normal; font-size: 7.2pt; color: #8a4b00; line-height: 1.3;
                border-top: 0.5pt dashed #c9b28a; padding-top: 2pt; margin-top: auto; }
.flow .arrow { flex: 0 0 9pt; display: flex; align-items: center; justify-content: center;
               color: #557; font-size: 9pt; }
.flow-src { font-size: 7.8pt; color: #445; text-align: center; border: 0.6pt dashed #9fb2c0;
            border-radius: 3pt; padding: 2pt; margin: 0 0 6pt; }
.fig { margin: 4pt 0 6pt; break-inside: avoid; }
figure.shot { margin: 4pt 0 8pt; break-inside: avoid; }
figure.shot img { width: 100%; border: 0.6pt solid #c3c2b7; border-radius: 3pt; display: block; }
figure.shot figcaption { font-size: 8pt; color: #52514e; margin-top: 2pt; }
.fig svg { display: block; width: 100%; height: auto; }
.band { font-size: 7.8pt; color: #fff; background: #0b3d5c; border-radius: 3pt; text-align: center;
        padding: 2pt; margin: 6pt 0 0; }
"""


def fix_lists(md_text: str) -> str:
    """문단 바로 뒤에 붙은 목록 앞에 빈 줄을 넣는다 (Python-Markdown은 빈 줄이 있어야 목록으로 인식)."""
    out = []
    item = re.compile(r"^\s*([-*+]|\d+\.)\s")
    in_code = False
    for line in md_text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if (not in_code and item.match(line) and out and out[-1].strip()
                and not item.match(out[-1]) and not out[-1].startswith(("|", "<", "  "))):
            out.append("")
        out.append(line)
    return "\n".join(out)


def find_browser() -> str:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for name in ("google-chrome", "chromium", "chrome", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    raise SystemExit("Chrome 또는 Edge를 찾지 못했습니다.")


SVG_INCLUDE = re.compile(r"^<!--\s*svg:\s*(\S+)\s*-->\s*$", re.M)


def include_svgs(md_text: str, base: Path) -> str:
    """`<!-- svg: figures/x.svg -->` 줄을 그 SVG 내용으로 바꾼다(md 파일 위치 기준). 인터넷 없이 PDF에 그림을 넣는다."""
    def rep(m):
        f = (base / m.group(1)).resolve()
        if not f.exists():
            raise SystemExit(f"그림 파일 없음: {m.group(1)}")
        return '<div class="fig">' + f.read_text(encoding="utf-8") + "</div>"
    return SVG_INCLUDE.sub(rep, md_text)


IMG_INCLUDE = re.compile(r"^<!--\s*img:\s*(\S+)(?:\s+(.*?))?\s*-->\s*$", re.M)


def include_images(md_text: str, base: Path) -> str:
    """`<!-- img: images/x.jpg 설명 -->` 줄을 base64 그림으로 바꾼다(PDF 한 파일에 그림이 들어가게)."""
    import base64
    import mimetypes

    def rep(m):
        f = (base / m.group(1)).resolve()
        if not f.exists():
            raise SystemExit(f"그림 파일 없음: {m.group(1)}")
        mime = mimetypes.guess_type(f.name)[0] or "image/png"
        data = base64.b64encode(f.read_bytes()).decode()
        cap = f'<figcaption>{html.escape(m.group(2))}</figcaption>' if m.group(2) else ""
        return f'<figure class="shot"><img src="data:{mime};base64,{data}">{cap}</figure>'
    return IMG_INCLUDE.sub(rep, md_text)


def to_html(md_text: str, title: str, base: Path | None = None) -> str:
    if base is not None:
        md_text = include_images(include_svgs(md_text, base), base)
    body = markdown.markdown(fix_lists(md_text), extensions=["tables", "fenced_code", "sane_lists"])
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")


def count_pages(pdf: Path) -> int:
    data = pdf.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--max-pages", type=int, default=15)
    a = ap.parse_args()

    src = a.src if a.src.is_absolute() else ROOT / a.src
    out = a.out or ROOT / "outputs" / "docs" / (src.stem + ".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    page = to_html(src.read_text(encoding="utf-8"), src.stem, base=src.parent)
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / "doc.html"
        h.write_text(page, encoding="utf-8")
        cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               f"--print-to-pdf={out}", "--user-data-dir=" + str(Path(td) / "profile"),
               h.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not out.exists():
            raise SystemExit(f"PDF 생성 실패 (코드 {r.returncode})")

    n = count_pages(out)
    print(f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {n}쪽")
    if n > a.max_pages:
        print(f"경고: {a.max_pages}쪽 제한을 넘었습니다.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
