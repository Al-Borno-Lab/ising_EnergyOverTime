#!/usr/bin/env python3
# coding: utf-8
"""
regen_model_quality_summary.py
-------------------------------
Regenerate model_quality_summary figures (updated formatting: bold panel
letters A-D, no legends, larger fonts, tighter layout) for every
already-processed session/stim under a results folder, WITHOUT re-fitting
the Ising model or re-sampling the Metropolis chain.

Each stim directory produced by main.py already contains everything needed
to redraw the figure:
    model_quality_summary_pairwise.csv   -> panel A (pairwise k=2 corr)
    model_quality_summary_J_values.csv   -> panel B (J coupling histogram)
    model_quality_summary_triplet.csv    -> panel C (triplet k=3 corr)
    model_quality_summary_P_K.csv        -> panel D (P(K): data/Ising/indep.)
    model_quality_summary_P_K_metadata.csv -> N (number of neurons)

This script just reads those CSVs and calls
visualization.plot_model_quality_summary_from_arrays(), so it's fast (no
Monte Carlo sampling) and reproduces exactly the numbers already reported in
the CSVs.

Usage
-----
  # Overwrite model_quality_summary.png/.svg in place, everywhere under root
  python regen_model_quality_summary.py --data_folder outputs/energy_decomp_Jul_10_svg

  # Write flat copies elsewhere instead, named <session>_<stim>_model_quality_summary.*
  python regen_model_quality_summary.py \\
      --data_folder outputs/energy_decomp_Jul_10_svg \\
      --output_dir outputs/energy_decomp_Jul_10_svg/model_quality_summary

  # Restrict to specific sessions, or preview without writing anything
  python regen_model_quality_summary.py --data_folder outputs/energy_decomp_Jul_10_svg \\
      --sessions 210421 210422 --dry_run
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualization import plot_model_quality_summary_from_arrays

REQUIRED_FILES = (
    "model_quality_summary_pairwise.csv",
    "model_quality_summary_triplet.csv",
    "model_quality_summary_J_values.csv",
    "model_quality_summary_P_K.csv",
    "model_quality_summary_P_K_metadata.csv",
)


def _parse_id_and_stim(path: Path):
    """
    Extract (session_id, stim_num) from *path*'s components:
      - session id  : prefix of a component matching ``<id>_results``
      - stim number : digits of a component matching ``stim_<N>``
    Either may be None if not found. Mirrors collect_files._parse_id_and_stim.
    """
    session_id, stim_num = None, None
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


def find_stim_dirs(root: Path, sessions=None, path_filter=None):
    """Yield every directory under *root* that has all REQUIRED_FILES."""
    for dirpath, _dirs, files in os.walk(root):
        if not all(f in files for f in REQUIRED_FILES):
            continue
        d = Path(dirpath)
        if path_filter and path_filter not in str(d):
            continue
        session_id, stim_num = _parse_id_and_stim(d)
        if sessions and session_id not in sessions:
            continue
        yield d, session_id, stim_num


def regenerate_one(stim_dir: Path, session_id, stim_num,
                    output_dir: Path = None, max_k_plot=None,
                    dry_run: bool = False) -> str:
    """Regenerate the figure for one stim directory. Returns the output path."""
    pw   = pd.read_csv(stim_dir / "model_quality_summary_pairwise.csv")
    tr   = pd.read_csv(stim_dir / "model_quality_summary_triplet.csv")
    jv   = pd.read_csv(stim_dir / "model_quality_summary_J_values.csv")
    pk   = pd.read_csv(stim_dir / "model_quality_summary_P_K.csv")
    meta = pd.read_csv(stim_dir / "model_quality_summary_P_K_metadata.csv")
    N = int(meta["N_neurons"].iloc[0])

    if output_dir is not None:
        prefix = f"{session_id}_{stim_num}" if session_id and stim_num is not None else stim_dir.name
        filename = f"{prefix}_model_quality_summary.png"
        dest_dir = output_dir
    else:
        filename = "model_quality_summary.png"
        dest_dir = stim_dir

    out_path = dest_dir / filename
    if dry_run:
        return str(out_path)

    dest_dir.mkdir(parents=True, exist_ok=True)
    plot_model_quality_summary_from_arrays(
        pw["pairwise_corr_orig"].values, pw["pairwise_corr_model"].values,
        jv["J"].values,
        tr["triplet_corr_orig"].values, tr["triplet_corr_model"].values,
        tr["triplet_corr_independent"].values,
        pk["K"].values, pk["P_data"].values, pk["P_ising"].values,
        pk["P_independent"].values, N,
        str(dest_dir), filename=filename, max_k_plot=max_k_plot,
    )
    return str(out_path)


def main():
    p = argparse.ArgumentParser(
        description="Regenerate model_quality_summary figures from already-saved CSVs "
                    "(no re-fitting or re-sampling).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_folder", required=True,
                   help="Root directory to search recursively for stim folders "
                        "(e.g. outputs/energy_decomp_Jul_10_svg).")
    p.add_argument("--output_dir", default=None,
                   help="If set, write flat copies here named "
                        "<session>_<stim>_model_quality_summary.png/.svg "
                        "(matches collect_files.py's naming). Default: overwrite "
                        "model_quality_summary.png/.svg in place in each stim folder.")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Restrict to these session IDs (e.g. 210421 210422).")
    p.add_argument("--path_filter", default=None, metavar="SUBSTR",
                   help="Only process stim folders whose path contains this substring "
                        "(e.g. 'full_reach').")
    p.add_argument("--max_k_plot", type=int, default=None,
                   help="Limit the P(K) x-axis to [0, max_k_plot] (default: 0..N).")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be regenerated without writing any files.")
    args = p.parse_args()

    root = Path(args.data_folder).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: data folder does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    sessions = set(args.sessions) if args.sessions else None
    stim_dirs = list(find_stim_dirs(root, sessions=sessions, path_filter=args.path_filter))

    if not stim_dirs:
        print("No stim folders with model_quality_summary CSVs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(stim_dirs)} stim folder(s) under {root}")
    if output_dir:
        print(f"Writing flat copies to: {output_dir}")
    else:
        print("Overwriting model_quality_summary.png/.svg in place.")
    if args.dry_run:
        print("Mode: DRY RUN\n")

    n_ok, n_err = 0, 0
    for stim_dir, session_id, stim_num in stim_dirs:
        label = f"{session_id or '?'} stim={stim_num or '?'}"
        try:
            out_path = regenerate_one(
                stim_dir, session_id, stim_num,
                output_dir=output_dir, max_k_plot=args.max_k_plot,
                dry_run=args.dry_run,
            )
            verb = "[dry-run] would write" if args.dry_run else "OK  ->"
            print(f"  {verb} {label:<20} {out_path}")
            n_ok += 1
        except Exception as e:
            print(f"  [WARN] {label}: {e}", file=sys.stderr)
            n_err += 1

    verb = "Would regenerate" if args.dry_run else "Regenerated"
    print(f"\n{verb} {n_ok} figure(s)" + (f", {n_err} failed" if n_err else "") + ".")


if __name__ == "__main__":
    main()
