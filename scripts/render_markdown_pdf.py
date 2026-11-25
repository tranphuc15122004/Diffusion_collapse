"""Render Markdown report into GitHub/VS Code Markdown Preview-styled HTML and PDF via Pandoc + WeasyPrint."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MD_PATH = DOCS / "R5_PILOT_FULL_REPORT.md"
HTML_PATH = DOCS / "R5_PILOT_FULL_REPORT.html"
PDF_PATH = DOCS / "R5_PILOT_FULL_REPORT.pdf"

GITHUB_CSS = """
@page {
  size: A4 portrait;
  margin: 13mm 12mm 15mm 12mm;
  @bottom-right {
    content: "Trang " counter(page) " / " counter(pages);
    font-family: "Lato", "DejaVu Sans", sans-serif;
    font-size: 8pt;
    color: #57606a;
  }
  @bottom-left {
    content: "Diffusion Collapse — Pilot R5 Empirical Report";
    font-family: "Lato", "DejaVu Sans", sans-serif;
    font-size: 8pt;
    color: #57606a;
  }
}

* {
  box-sizing: border-box;
}

body {
  font-family: "Lato", "DejaVu Sans", sans-serif;
  font-size: 9.5pt;
  line-height: 1.55;
  color: #1f2328;
  background: #ffffff;
  margin: 0;
  padding: 0;
}

h1, h2, h3, h4 {
  font-family: "Lato", "DejaVu Sans", sans-serif;
  font-weight: 700;
  color: #1f2328;
  margin-top: 1.25em;
  margin-bottom: 0.55em;
  line-height: 1.25;
  page-break-after: avoid;
  break-after: avoid;
}

h1 {
  font-size: 16pt;
  padding-bottom: 0.35em;
  border-bottom: 2px solid #d0d7de;
  margin-top: 0;
}

h2 {
  font-size: 12.5pt;
  padding-bottom: 0.25em;
  border-bottom: 1px solid #d8dee4;
  color: #0969da;
}

h3 {
  font-size: 10.5pt;
  color: #1f2328;
}

p {
  margin: 0.45em 0 0.7em 0;
}

ul, ol {
  margin: 0.35em 0 0.75em 0;
  padding-left: 1.5em;
}

li {
  margin: 0.22em 0;
}

a {
  color: #0969da;
  text-decoration: none;
}

hr {
  border: 0;
  height: 1px;
  background: #d8dee4;
  margin: 1.1em 0;
}

code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.2pt;
  background-color: #eff1f3;
  color: #1f2328;
  padding: 1.5px 4px;
  border-radius: 4px;
}

td code, th code {
  font-size: 6.9pt;
  padding: 1px 3px;
}

.ci {
  white-space: nowrap;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.6em 0 0.9em 0;
  font-size: 7.2pt;
  line-height: 1.3;
}

thead {
  display: table-header-group;
}

tr {
  page-break-inside: avoid;
  break-inside: avoid;
}

thead tr {
  background-color: #f6f8fa;
}

th, td {
  border: 1px solid #d0d7de;
  padding: 3.5px 4.5px;
  vertical-align: middle;
}

th {
  font-weight: 700;
  color: #1f2328;
  background-color: #f6f8fa;
}

tbody tr:nth-child(even) {
  background-color: #f9fbfd;
}

figure {
  margin: 0.75em 0 1em 0;
  padding: 6px;
  border: 1px solid #d8dee4;
  border-radius: 6px;
  background: #ffffff;
  text-align: center;
  page-break-inside: avoid;
  break-inside: avoid;
}

figure img, p > img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}

figcaption {
  font-size: 8.2pt;
  color: #57606a;
  margin-top: 5px;
  font-style: italic;
}

blockquote {
  margin: 0.6em 0;
  padding: 0.4em 0.9em;
  color: #57606a;
  border-left: 3.5px solid #0969da;
  background: #f6f8fa;
  border-radius: 0 4px 4px 0;
}

.math-i {
  font-style: italic;
  font-family: "Lato", "DejaVu Sans", sans-serif;
  white-space: nowrap;
}
"""


def tex_inline_to_html(expr: str) -> str:
    """Convert simple inline LaTeX expressions into clean HTML for WeasyPrint."""
    s = expr.strip()
    replacements = [
        (r"\rightarrow", "&rarr;"),
        (r"\to", "&rarr;"),
        (r"\Delta", "&Delta;"),
        (r"\rho", "&rho;"),
        (r"\backslash", "&bsol;"),
        (r"\le", "&le;"),
        (r"\ge", "&ge;"),
        (r"\approx", "&approx;"),
        (r"\times", "&times;"),
    ]
    for k, v in replacements:
        s = s.replace(k, v)
    # Handle _{...} and ^{...}
    s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9]+)", r"<sup>\1</sup>", s)
    return f'<span class="math-i">{s}</span>'


def replace_inline_math(md_text: str) -> str:
    # Protect code spans first so `$` inside code isn't touched
    parts = re.split(r"(`[^`]+`)", md_text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
        else:
            out.append(re.sub(r"\$([^$\n]+)\$", lambda m: tex_inline_to_html(m.group(1)), part))
    return "".join(out)


def main() -> None:
    raw_md = MD_PATH.read_text(encoding="utf-8")
    clean_md = replace_inline_math(raw_md)

    tmp_md = DOCS / "_tmp_preview.md"
    tmp_md.write_text(clean_md, encoding="utf-8")

    # Convert to standalone HTML body via pandoc
    body_html = subprocess.check_output(
        [
            "pandoc",
            str(tmp_md),
            "--from",
            "markdown+pipe_tables+raw_html",
            "--to",
            "html5",
        ],
        cwd=str(DOCS),
        text=True,
    )
    tmp_md.unlink(missing_ok=True)
    # Keep short bracketed intervals (e.g., [.062, .130]) from breaking after '.'
    body_html = re.sub(r"<code>(\[[^\]]{1,24}\])</code>", r'<code class="ci">\1</code>', body_html)
    body_html = re.sub(r"(?<!<code>)(\[\.?\d+[^\]]{1,16}\])", r'<span class="ci">\1</span>', body_html)

    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<title>Báo cáo Toàn diện Pilot R5 — Diffusion Collapse</title>
<style>
{GITHUB_CSS}
</style>
</head>
<body class="markdown-body">
{body_html}
</body>
</html>
"""
    HTML_PATH.write_text(full_html, encoding="utf-8")

    subprocess.run(
        [
            "uvx",
            "--from",
            "weasyprint",
            "weasyprint",
            str(HTML_PATH),
            str(PDF_PATH),
        ],
        cwd=str(DOCS),
        check=True,
    )
    print(f"Generated HTML: {HTML_PATH} and PDF: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
