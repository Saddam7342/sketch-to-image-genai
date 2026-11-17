"""Render report/report.html to report/report.pdf with headless Chrome (or Edge) and print the page count.

  python scripts/build_report.py [--out report/report.pdf]
"""
import argparse
import shutil
import subprocess
from pathlib import Path

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]


def browser() -> str:
    for c in CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("No Chrome/Chromium/Edge found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="report/report.html")
    ap.add_argument("--out", default="report/report.pdf")
    a = ap.parse_args()
    html, out = Path(a.html).resolve(), Path(a.out).resolve()
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", html.as_uri()], check=True, capture_output=True)
    try:
        from pypdf import PdfReader
        print(f"{out}: {len(PdfReader(out).pages)} pages")
    except ImportError:
        print("wrote", out)


if __name__ == "__main__":
    main()
