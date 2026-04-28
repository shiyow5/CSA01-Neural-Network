#!/usr/bin/env python3
"""Markdown (English) -> HTML -> PDF (Chrome headless)"""
import subprocess, os

SRC = "report_TP2_Eng.md"
DST = "report_TP2_Eng.pdf"
HTML_TMP = os.path.abspath("report_TP2_Eng_tmp.html")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Step 1: Markdown -> HTML body (pandoc, embed images)
body_tmp = "_body_eng_tmp.html"
subprocess.run(
    ["pandoc", SRC, "-o", body_tmp, "--embed-resources", "--standalone", "--mathml"],
    check=True
)

with open(body_tmp, "r", encoding="utf-8") as f:
    body_html = f.read()
os.remove(body_tmp)

# Step 2: Wrap with CSS
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
}}
body {{
    font-family: "Helvetica Neue", "Arial", sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #222;
}}
h1 {{
    font-size: 16pt;
    border-bottom: 2px solid #333;
    padding-bottom: 6px;
    margin-top: 0;
}}
h2 {{
    font-size: 14pt;
    border-bottom: 1px solid #999;
    padding-bottom: 4px;
    margin-top: 28px;
    page-break-after: avoid;
}}
h3 {{
    font-size: 12pt;
    margin-top: 20px;
    page-break-after: avoid;
}}
h4 {{
    font-size: 11pt;
    margin-top: 16px;
    page-break-after: avoid;
}}
code {{
    font-family: "Menlo", "Courier New", monospace;
    font-size: 9pt;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
}}
pre {{
    background: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px 12px;
    overflow-x: auto;
    font-size: 7.5pt;
    line-height: 1.35;
    page-break-inside: auto;
}}
pre code {{
    background: none;
    padding: 0;
    font-size: 7.5pt;
}}
table {{
    border-collapse: collapse;
    margin: 12px 0;
    width: 100%;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #aaa;
    padding: 5px 10px;
    text-align: center;
    font-size: 10pt;
}}
th {{
    background: #f0f0f0;
    font-weight: bold;
}}
img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 14px auto;
    page-break-inside: avoid;
}}
figure {{
    margin: 14px 0;
    page-break-inside: avoid;
}}
ul, ol {{
    padding-left: 24px;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 24px 0;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

with open(HTML_TMP, "w", encoding="utf-8") as f:
    f.write(full_html)

# Step 3: HTML -> PDF via Chrome headless
pdf_path = os.path.abspath(DST)
result = subprocess.run([
    CHROME,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    f"--print-to-pdf={pdf_path}",
    "--print-to-pdf-no-header",
    f"file://{HTML_TMP}",
], capture_output=True, timeout=30)

if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
    print(f"PDF generated: {DST} ({os.path.getsize(pdf_path)} bytes)")
    os.remove(HTML_TMP)
else:
    print("PDF generation failed.")
    print("stderr:", result.stderr.decode() if result.stderr else "(none)")
    print(f"HTML file kept: {HTML_TMP}")
