#!/usr/bin/env python3
"""
Generate a per-session image report as a Markdown file and optionally a PDF.

Walks a root results directory, discovers every folder that contains PNG
output files, copies the images into a flat assets/ folder (named by
session path + graph stem), and writes a single Markdown document that
lays out every session with its images grouped by chart type.

Pass --pdf to also produce session_report.pdf directly via matplotlib
(no external dependencies required).

Usage:
    python generate_session_report.py <root_dir>
           [--output_dir notes/session_report]
           [--title "My Analysis Report"]
           [--columns 2]
           [--groups phase_transition model_quality correlations kinematics transition energy_dist]
           [--no_copy]
           [--pdf]
"""

import argparse
import math
import os
import shutil
from pathlib import Path
from datetime import date
from collections import defaultdict
from fnmatch import fnmatch

# ---------------------------------------------------------------------------
# Image group definitions  (name, display title, list of glob-style stems)
# ---------------------------------------------------------------------------
IMAGE_GROUPS = [
    (
        "phase_transition",
        "Phase Transition",
        ["avg_spin_vs_temp", "heat_capacity", "energy_vs_temp"],
    ),
    (
        "model_quality",
        "Model Quality Summary",
        ["model_quality_summary"],
    ),
    (
        "correlations",
        "Correlation Orders",
        ["correlation_order_*"],
    ),
    (
        "kinematics",
        "Energy & Kinematics",
        ["energy_kinematics_stim_*"],
    ),
    (
        "transition",
        "Transition Points",
        ["transition_points_stim_*"],
    ),
    (
        "energy_dist",
        "Energy Distributions",
        ["energy_histogram_stim_*", "energy_distribution_by_k_heatmap*"],
    ),
]

ALL_GROUP_NAMES = [g[0] for g in IMAGE_GROUPS]


def _group_for_stem(stem):
    """Return the group name whose pattern matches *stem*, or 'other'."""
    for name, _, patterns in IMAGE_GROUPS:
        for pat in patterns:
            if fnmatch(stem, pat):
                return name
    return "other"


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_session_dirs(root):
    """Return every directory (recursively) that contains at least one .png."""
    found = []
    for dirpath, _, files in os.walk(str(root)):
        if any(f.lower().endswith(".png") for f in files):
            found.append(Path(dirpath))
    return sorted(found)


def session_id(root, session_dir):
    """
    Return a flat, filesystem-safe identifier for *session_dir* relative to
    *root*, e.g. 'experiment_rep1_full_reach'.
    """
    try:
        rel = session_dir.relative_to(root)
    except ValueError:
        rel = Path(session_dir.name)
    parts = [p for p in rel.parts if p and p != "."]
    return "_".join(parts) if parts else session_dir.name


def session_display_name(root, session_dir):
    """Return a human-readable title derived from the relative path."""
    try:
        rel = session_dir.relative_to(root)
    except ValueError:
        return session_dir.name
    return str(rel).replace("_", " ")


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _heading(level, text):
    return f"{'#' * level} {text}"


def _image_grid(items, columns):
    """
    Render a list of (alt_text, rel_path) tuples as an HTML table grid.
    Returns a list of Markdown/HTML lines.

    A single image is rendered full-width without a table wrapper so it
    takes all available space (useful for wide summary plots).
    """
    if len(items) == 1:
        alt, rel_path = items[0]
        return [
            f'<div align="center">',
            f'<img src="{rel_path}" alt="{alt}" style="max-width:100%">',
            f"<br><em>{alt}</em>",
            "</div>",
            "",
        ]

    lines = ["<table>"]
    for row_start in range(0, len(items), columns):
        row = items[row_start : row_start + columns]
        lines.append("<tr>")
        for alt, rel_path in row:
            lines.append(
                f'<td align="center" width="{100 // columns}%">'
                f'<img src="{rel_path}" alt="{alt}" style="max-width:100%">'
                f"<br><em>{alt}</em></td>"
            )
        # Pad the last incomplete row
        for _ in range(columns - len(row)):
            lines.append("<td></td>")
        lines.append("</tr>")
    lines += ["</table>", ""]
    return lines


# ---------------------------------------------------------------------------
# Core report generator
# ---------------------------------------------------------------------------

def generate_report(root, output_dir, title, selected_groups, no_copy, columns=2, name="session_report"):
    assets_dir = output_dir / "assets"

    if not no_copy:
        assets_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = find_session_dirs(root)
    if not session_dirs:
        print(f"No directories with PNG files found under {root}")
        report_path = output_dir / f"{name}.md"
        return {}, report_path

    print(f"Found {len(session_dirs)} session(s).")

    # Build mapping:  session_id -> group_name -> [(stem, asset_filename)]
    session_images = defaultdict(lambda: defaultdict(list))
    copied_count = 0

    for sd in session_dirs:
        sid = session_id(root, sd)
        pngs = sorted(sd.glob("*.png"))
        for png in pngs:
            group = _group_for_stem(png.stem)
            if group != "other" and group not in selected_groups:
                continue

            # Unique asset filename: session_id__original_name.png
            safe_sid = sid.replace("/", "_").replace("\\", "_").replace(" ", "_")
            asset_fname = f"{safe_sid}__{png.name}" if safe_sid else png.name

            if not no_copy:
                dst = assets_dir / asset_fname
                shutil.copy2(str(png), str(dst))
                copied_count += 1

            session_images[sid][group].append((png.stem, asset_fname))

    # ------------------------------------------------------------------
    # Build Markdown
    # ------------------------------------------------------------------
    lines = []

    # Title block
    lines += [
        _heading(1, title),
        "",
        f"*Generated: {date.today().isoformat()}*",
        "",
        f"**Root directory:** `{root}`",
        "",
        f"**Sessions included:** {len(session_images)}",
        "",
    ]

    sorted_sids = sorted(session_images.keys())
    group_order = ALL_GROUP_NAMES + ["other"]

    # Table of contents — one entry per session
    lines += [_heading(2, "Contents"), ""]
    for sid in sorted_sids:
        display_name = session_display_name(root, next(
            sd for sd in session_dirs if session_id(root, sd) == sid
        ))
        anchor = sid.lower().replace(" ", "-").replace("_", "-")
        lines.append(f"- [{display_name}](#{anchor})")
    lines += ["", "---", ""]

    # Session-first layout: ## Session → ### Group → images
    for sid in sorted_sids:
        display_name = session_display_name(root, next(
            sd for sd in session_dirs if session_id(root, sd) == sid
        ))
        lines += [_heading(2, display_name), ""]

        for gname in group_order:
            if gname not in session_images[sid]:
                continue

            display_title = next(
                (g[1] for g in IMAGE_GROUPS if g[0] == gname),
                gname.replace("_", " ").title(),
            )
            lines += [_heading(3, display_title), ""]

            grid_items = [
                (stem.replace("_", " "), f"assets/{asset_fname}")
                for stem, asset_fname in sorted(session_images[sid][gname])
            ]
            lines += _image_grid(grid_items, columns)

        lines += ["---", ""]

    report_path = output_dir / f"{name}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report written to : {report_path}")
    if not no_copy:
        print(f"Copied {copied_count} image(s) to : {assets_dir}")

    return session_images, report_path


# ---------------------------------------------------------------------------
# PDF via markdown-convert (preferred: proper HTML rendering)
# ---------------------------------------------------------------------------

def convert_to_pdf_markdown_convert(md_path, output_path, pip_packages_dir=None):
    """
    Convert *md_path* → *output_path* (PDF) using markdown-convert.

    Strategy:
      1. Try converting the whole document at once.
      2. If Playwright hits its string-length limit (common with many images),
         split the document into per-session chunks, convert each separately,
         then merge them with pypdf.

    Returns True on success, False if markdown-convert is not installed.
    """
    import sys
    import os
    import tempfile

    if pip_packages_dir and str(pip_packages_dir) not in sys.path:
        sys.path.insert(0, str(pip_packages_dir))

    try:
        from markdown_convert import convert as mc_convert
    except ImportError:
        return False

    md_path = Path(md_path).resolve()
    out_path = Path(output_path).resolve()
    orig = os.getcwd()

    def _mc(src_md, dst_pdf):
        """Convert a single markdown file to PDF, cwd = markdown's directory."""
        try:
            os.chdir(str(Path(src_md).parent))
            mc_convert(Path(src_md), None, Path(dst_pdf), security_level="basic")
        finally:
            os.chdir(orig)

    # ── Attempt 1: whole document ──────────────────────────────────────────
    try:
        _mc(md_path, out_path)
        return True
    except (RuntimeError, Exception) as exc:
        if "Cannot create a string longer" not in str(exc):
            raise
        print(
            "  (Document too large for single-pass Playwright conversion — "
            "switching to per-session chunked mode …)"
        )

    # ── Attempt 2: split by top-level ## heading, merge with pypdf ────────
    try:
        from pypdf import PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfWriter  # older name
        except ImportError:
            print(
                "  pypdf / PyPDF2 not found — cannot merge chunk PDFs.\n"
                "  Install with: pip install pypdf --target .pip_packages/"
            )
            return False

    text = md_path.read_text(encoding="utf-8")

    # Split on lines that start with exactly "## " (session headings).
    # Keep the preamble (title block + TOC) as its own chunk.
    import re
    parts = re.split(r"(?m)^(?=## )", text)

    if len(parts) <= 1:
        # No session headings found; cannot chunk — give up
        return False

    writer = PdfWriter()
    tmp_dir = Path(tempfile.mkdtemp(prefix="session_report_chunks_"))
    chunk_dir = tmp_dir / "chunks"
    chunk_dir.mkdir()
    # Copy assets alongside the chunk markdown files so relative paths work
    assets_src = md_path.parent / "assets"
    assets_dst = chunk_dir / "assets"
    if assets_src.is_dir():
        import shutil as _shutil
        _shutil.copytree(str(assets_src), str(assets_dst))

    print(f"  Splitting into {len(parts)} chunk(s) …")
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        chunk_md = chunk_dir / f"chunk_{i:04d}.md"
        chunk_pdf = chunk_dir / f"chunk_{i:04d}.pdf"
        chunk_md.write_text(part, encoding="utf-8")
        try:
            _mc(chunk_md, chunk_pdf)
        except Exception as chunk_exc:
            print(f"  Warning: chunk {i} failed ({chunk_exc}); skipping.")
            continue
        if chunk_pdf.exists():
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            try:
                reader = PdfReader(str(chunk_pdf))
                for page in reader.pages:
                    writer.add_page(page)
            except Exception:
                pass

    if len(writer.pages) == 0:
        return False

    with open(str(out_path), "wb") as fh:
        writer.write(fh)

    # Clean up temp directory
    import shutil as _shutil2
    _shutil2.rmtree(str(tmp_dir), ignore_errors=True)

    return True


# ---------------------------------------------------------------------------
# PDF generator (matplotlib fallback – no external tools needed)
# ---------------------------------------------------------------------------

def generate_pdf(session_images, assets_dir, output_path, columns, title):
    """
    Build a multi-page PDF from the already-collected session_images dict.

    Layout:
      • Cover page  – title, date, session count
      • Per session – grey divider page, then one page per image group
                      with images arranged in an N-column grid
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages

    PAGE_W, PAGE_H = 8.5, 11.0          # letter size in inches
    IMAGES_PER_PAGE = columns * 2       # 2 rows of images per page
    group_order = ALL_GROUP_NAMES + ["other"]

    def _flat_axes(axes, n_rows, n_cols):
        """Return a flat list of Axes regardless of subplot shape."""
        if n_rows == 1 and n_cols == 1:
            return [axes]
        if n_rows == 1 or n_cols == 1:
            return list(axes.flat) if hasattr(axes, "flat") else list(axes)
        return [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

    with PdfPages(str(output_path)) as pdf:

        # ── Cover page ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.62, title,
                 ha="center", va="center", fontsize=26, fontweight="bold")
        fig.text(0.5, 0.52, f"Generated: {date.today().isoformat()}",
                 ha="center", va="center", fontsize=13, color="#444444")
        fig.text(0.5, 0.45, f"Sessions: {len(session_images)}",
                 ha="center", va="center", fontsize=13, color="#444444")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Per-session pages ──────────────────────────────────────────────
        for sid in sorted(session_images.keys()):

            # Session divider page
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            fig.patch.set_facecolor("#e8ecf0")
            fig.text(0.5, 0.58, sid.replace("_", " "),
                     ha="center", va="center", fontsize=22, fontweight="bold")
            fig.text(0.5, 0.48, sid,
                     ha="center", va="center", fontsize=9,
                     color="#666666", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            groups_here = session_images[sid]

            for gname in group_order:
                if gname not in groups_here:
                    continue

                display_title = next(
                    (g[1] for g in IMAGE_GROUPS if g[0] == gname),
                    gname.replace("_", " ").title(),
                )
                items = sorted(groups_here[gname])   # [(stem, asset_fname), …]

                # Paginate: IMAGES_PER_PAGE images per page
                for page_start in range(0, len(items), IMAGES_PER_PAGE):
                    batch = items[page_start : page_start + IMAGES_PER_PAGE]
                    n_rows = max(1, math.ceil(len(batch) / columns))

                    fig, axes = plt.subplots(
                        n_rows, columns,
                        figsize=(PAGE_W, PAGE_H * 0.88),
                        squeeze=False,
                    )
                    page_suffix = (
                        f" ({page_start // IMAGES_PER_PAGE + 1})"
                        if len(items) > IMAGES_PER_PAGE
                        else ""
                    )
                    fig.suptitle(
                        f"{sid.replace('_', ' ')}  —  {display_title}{page_suffix}",
                        fontsize=11, fontweight="bold", y=0.99,
                    )

                    flat = _flat_axes(axes, n_rows, columns)

                    for idx, (stem, asset_fname) in enumerate(batch):
                        ax = flat[idx]
                        img_path = assets_dir / asset_fname
                        if img_path.exists():
                            img = mpimg.imread(str(img_path))
                            ax.imshow(img)
                        ax.set_title(stem.replace("_", " "), fontsize=8, pad=3)
                        ax.axis("off")

                    # Hide unused axes in the last (potentially partial) row
                    for idx in range(len(batch), n_rows * columns):
                        flat[idx].axis("off")

                    fig.tight_layout(rect=[0, 0, 1, 0.97])
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    print(f"PDF  written to : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a per-session image report as a Markdown/PDF-ready document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root_dir",
        help="Root directory to scan recursively for session PNG files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory to write session_report.md and assets/ into. "
            "Defaults to <script_dir>/notes/session_report."
        ),
    )
    parser.add_argument(
        "--title",
        default="Session Analysis Report",
        help="Title shown at the top of the Markdown document.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=ALL_GROUP_NAMES + ["other"],
        default=ALL_GROUP_NAMES,
        metavar="GROUP",
        help=(
            "Which image groups to include. Available: "
            + ", ".join(ALL_GROUP_NAMES)
            + ". Add 'other' to also include unrecognized image names. "
            "Default: all recognised groups."
        ),
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        metavar="N",
        help="Number of image columns per row in each group grid (default: 2).",
    )
    parser.add_argument(
        "--name",
        default="session_report",
        metavar="NAME",
        help=(
            "Base filename for the output files (no extension). "
            "Produces <name>.md and, with --pdf, <name>.pdf. "
            "Default: session_report. "
            "Example: --name april_2026_full_reach"
        ),
    )
    parser.add_argument(
        "--no_copy",
        action="store_true",
        help=(
            "Skip copying images into assets/; only regenerate the Markdown. "
            "Assumes images are already present in assets/."
        ),
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help=(
            "Also generate session_report.pdf. "
            "Uses markdown-convert (preferred, renders HTML grids correctly) if installed, "
            "otherwise falls back to matplotlib. "
            "Install markdown-convert with: "
            "pip install markdown-convert --target .pip_packages/"
        ),
    )
    parser.add_argument(
        "--pip_packages",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing a local pip install of markdown-convert "
            "(e.g. .pip_packages/). "
            "Defaults to <script_dir>/.pip_packages if it exists."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    root = Path(args.root_dir).expanduser().resolve()

    script_dir = Path(__file__).parent.resolve()
    default_out = script_dir / "notes" / "session_report"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_out
    )

    # Sanitise name: strip extension if the user accidentally passed one
    report_name = Path(args.name).stem if args.name else "session_report"

    # Each report lives in its own subfolder: <output_dir>/<name>/
    output_dir = output_dir / report_name

    session_images, md_path = generate_report(
        root=root,
        output_dir=output_dir,
        title=args.title,
        selected_groups=set(args.groups),
        no_copy=args.no_copy,
        columns=args.columns,
        name=report_name,
    )

    if args.pdf and session_images and any(session_images.values()):
        pdf_path = output_dir / f"{report_name}.pdf"

        # Resolve pip_packages directory
        default_pip = script_dir / ".pip_packages"
        pip_dir = (
            Path(args.pip_packages).expanduser().resolve()
            if args.pip_packages
            else (default_pip if default_pip.is_dir() else None)
        )

        print(f"\nConverting to PDF …")
        ok = convert_to_pdf_markdown_convert(md_path, pdf_path, pip_packages_dir=pip_dir)

        if ok:
            size = pdf_path.stat().st_size
            print(f"PDF  written to : {pdf_path}  ({size // 1024} KB)")
            print("     (rendered via markdown-convert, security_level='basic')")
        else:
            print("markdown-convert not found — falling back to matplotlib PDF …")
            print("(Install for better output: pip install markdown-convert --target .pip_packages/)")
            generate_pdf(
                session_images=session_images,
                assets_dir=output_dir / "assets",
                output_path=pdf_path,
                columns=args.columns,
                title=args.title,
            )
