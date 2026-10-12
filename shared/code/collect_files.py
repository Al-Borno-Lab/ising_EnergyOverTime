#!/usr/bin/env python3
"""
Recursively find files matching a name or glob pattern under a root directory
and copy them to a destination folder.

Usage examples:

  # Copy all per_reach_state.csv files (flat — renamed to <id>_<stim>_<file>)
  python collect_files.py /data/.../energy_decomp_Jun_22 per_reach_state.csv ./collected/
  # → e.g. 220520_0_per_reach_state.csv

  # Copy arbitration box-whisker PNGs, preserving subdirectory structure
  python collect_files.py /data/.../energy_decomp_Jun_22 arbitration_boxwhisker.png ./collected/ --keep_structure

  # Glob pattern: copy all raster overlay PNGs from any stim folder
  python collect_files.py /data/.../energy_decomp_Jun_22 "raster_overlay_stim*.png" ./collected/

  # Only copy from paths that contain 'full_reach'
  python collect_files.py /data/.../energy_decomp_Jun_22 per_reach_state.csv ./collected/ --path_filter full_reach

  # Dry run — print what would be copied without copying
  python collect_files.py /data/.../energy_decomp_Jun_22 per_reach_state.csv ./collected/ --dry_run
"""

import argparse
import fnmatch
import os
import re
import shutil
import sys
from pathlib import Path


def _parse_id_and_stim(path: Path) -> tuple[str | None, str | None]:
    """
    Walk the parts of *path* and extract:
      - session id  : the numeric/alphanumeric prefix from a component matching
                      ``<id>_results``  (e.g. '220520_results' → '220520')
      - stim number : the digit(s) from a component matching ``stim_<N>``
                      (e.g. 'stim_0' → '0')
    Returns (id, stim) — either may be None if not found.
    """
    session_id = None
    stim_num   = None
    for part in path.parts:
        if session_id is None:
            m = re.fullmatch(r"(.+)_results", part)
            if m:
                session_id = m.group(1)
        if stim_num is None:
            m = re.fullmatch(r"stim_(\d+)", part)
            if m:
                stim_num = m.group(1)
    return session_id, stim_num


def find_and_collect(root: Path, pattern: str, dest: Path,
                     keep_structure: bool = False,
                     path_filter: str = None,
                     dry_run: bool = False) -> int:
    """
    Walk *root* recursively, find files matching *pattern* (exact name or glob),
    and copy them to *dest*.

    When keep_structure=False (default) files are renamed:
        <id>_<stim>_<original_filename>
    where <id> comes from the ``<id>_results`` path component and <stim> from
    ``stim_<N>``.  If either cannot be parsed the immediate parent folder name
    is used as a fallback prefix.

    When keep_structure=True the relative path from root is preserved under dest.

    Returns the number of files copied (or matched in dry-run mode).
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0

    for dirpath, _dirs, files in os.walk(root):
        dir_ = Path(dirpath)

        # Optional path substring filter
        if path_filter and path_filter not in str(dir_):
            continue

        for fname in files:
            if not fnmatch.fnmatch(fname, pattern):
                continue

            src = dir_ / fname

            if keep_structure:
                rel = src.relative_to(root)
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
            else:
                session_id, stim_num = _parse_id_and_stim(src)
                if session_id is not None and stim_num is not None:
                    prefix = f"{session_id}_{stim_num}"
                else:
                    # Fallback: use immediate parent folder name
                    prefix = dir_.name
                dst = dest / f"{prefix}_{fname}"

                # Resolve any remaining collision
                if dst.exists() and dst != src:
                    prefix = f"{dir_.parent.name}__{dir_.name}__{prefix}"
                    dst = dest / f"{prefix}_{fname}"

            if dry_run:
                print(f"  [dry-run] {src}  →  {dst}")
            else:
                shutil.copy2(src, dst)
                print(f"  Copied: {src.name}  →  {dst}")

            copied += 1

    return copied


def main():
    p = argparse.ArgumentParser(
        description="Recursively find files and copy them to a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("root_dir",
                   help="Root directory to search recursively")
    p.add_argument("pattern",
                   help="File name or glob pattern to match (e.g. 'per_reach_state.csv' "
                        "or 'raster_overlay_stim*.png')")
    p.add_argument("dest_dir",
                   help="Destination folder to copy files into")
    p.add_argument("--keep_structure", action="store_true",
                   help="Preserve subdirectory structure under dest_dir "
                        "(default: flat copy with <id>_<stim> prefix)")
    p.add_argument("--path_filter", default=None, metavar="SUBSTR",
                   help="Only copy files whose full path contains this substring "
                        "(e.g. 'full_reach' or 'stim_0')")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be copied without actually copying")
    args = p.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    dest = Path(args.dest_dir).expanduser().resolve()

    if not root.is_dir():
        print(f"Error: root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching: {root}")
    print(f"Pattern  : {args.pattern}")
    print(f"Filter   : {args.path_filter or '(none)'}")
    print(f"Dest     : {dest}")
    print(f"Structure: {'preserved' if args.keep_structure else 'flat (<id>_<stim>_<file>)'}")
    if args.dry_run:
        print("Mode     : DRY RUN")
    print()

    n = find_and_collect(
        root=root,
        pattern=args.pattern,
        dest=dest,
        keep_structure=args.keep_structure,
        path_filter=args.path_filter,
        dry_run=args.dry_run,
    )

    verb = "Would copy" if args.dry_run else "Copied"
    print(f"\n{verb} {n} file(s) to {dest}")


if __name__ == "__main__":
    main()
