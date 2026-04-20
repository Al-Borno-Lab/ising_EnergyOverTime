#!/usr/bin/env python3
"""
Convert any Markdown file to PDF using markdown-convert.

Image paths in the Markdown are resolved relative to the directory
that contains the Markdown file, so you can run this from anywhere.

Usage
-----
    python md_to_pdf.py path/to/report.md

Output
------
PDF is written next to the input file with the same stem:
    path/to/report.pdf

Dependencies
------------
markdown-convert is already installed in .pip_packages/ (same directory as
this script).  No other external tools are required.
"""

import argparse
import os
import sys
from pathlib import Path


def _find_pip_packages(script_dir: Path) -> Path | None:
    """Return the .pip_packages path next to this script, if it exists."""
    candidate = script_dir / ".pip_packages"
    return candidate if candidate.is_dir() else None


def md_to_pdf(md_path: Path, pip_packages_dir: Path | None = None) -> Path:
    if pip_packages_dir and str(pip_packages_dir) not in sys.path:
        sys.path.insert(0, str(pip_packages_dir))

    try:
        from markdown_convert import convert as mc_convert
    except ImportError:
        print(
            "Error: markdown-convert is not installed.\n"
            "Install it with:\n"
            "  pip install markdown-convert --target .pip_packages/",
            file=sys.stderr,
        )
        sys.exit(1)

    md_path = md_path.resolve()
    out_path = md_path.with_suffix(".pdf")

    orig_dir = os.getcwd()
    try:
        # markdown-convert resolves image paths relative to cwd, so
        # change into the directory that contains the markdown file.
        os.chdir(str(md_path.parent))
        mc_convert(
            md_path,
            None,           # no custom CSS file
            out_path,
            security_level="basic",   # allow inline HTML + local image files
        )
    finally:
        os.chdir(orig_dir)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Markdown file to PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python md_to_pdf.py notes/my_report.md",
    )
    parser.add_argument("markdown_file", help="Path to the .md file to convert")
    args = parser.parse_args()

    md_path = Path(args.markdown_file).expanduser().resolve()
    if not md_path.exists():
        print(f"Error: file not found: {md_path}", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).parent.resolve()
    pip_dir = _find_pip_packages(script_dir)

    print(f"Converting : {md_path}")
    out = md_to_pdf(md_path, pip_packages_dir=pip_dir)
    size_kb = out.stat().st_size // 1024
    print(f"Saved PDF  : {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
