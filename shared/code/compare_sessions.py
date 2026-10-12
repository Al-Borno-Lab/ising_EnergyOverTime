#!/usr/bin/env python3
# coding: utf-8
"""
compare_sessions.py
--------------------
Build paper-ready side-by-side comparison figures for two session/stim pairs,
so the "regenerate, then manually relabel/place in Illustrator" loop isn't
needed every time something upstream changes:

  1. Model-quality summary (pairwise/triplet correlations, J distribution,
     P(K)) -- panels A-D per session; the right-hand session's letters get a
     "2" suffix so the two copies stay distinguishable (A1-D1 | A2-D2).
  2. pk_analysis per-session kinematics/energy/J-coupling figure -- panels
     A-E per session, same side-by-side/suffix scheme (A1-E1 | A2-E2).

Both are written as PDF (primary -- drop straight into a paper/Illustrator,
already vector and already labeled) and PNG (quick preview). Output
filenames are deterministic (built from the two session/stim pairs), so
re-running after fixing something upstream just overwrites them in place --
this is meant to be looped over from a wrapper shell script covering every
comparison the paper needs.

Usage
-----
  python compare_sessions.py \\
      --data_folder outputs/energy_decomp_Jul_10_svg \\
      --session1 210421 --stim1 0 \\
      --session2 210422 --stim2 0 \\
      --output_dir outputs/energy_decomp_Jul_10_svg/comparisons

  # Only one of the two figure types:
  python compare_sessions.py --data_folder outputs/energy_decomp_Jul_10_svg \\
      --session1 210421 --session2 210422 --skip_pk_analysis

  # Custom column labels instead of the default "Session <id>, stim <n>":
  python compare_sessions.py --data_folder outputs/energy_decomp_Jul_10_svg \\
      --session1 210421 --session2 210422 --label1 Baseline --label2 Perturbation

Wrapper shell-script example (rerun every comparison the paper needs after
any upstream change):
  while read -r s1 t1 s2 t2; do
      python compare_sessions.py --data_folder "$DATA" \\
          --session1 "$s1" --stim1 "$t1" --session2 "$s2" --stim2 "$t2" || echo "FAILED: $s1/$t1 vs $s2/$t2"
  done < pairs.txt
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.util import find_file_recursive
from regen_model_quality_summary import find_stim_dirs
from visualization import _draw_model_quality_panels
import pk_j_analysis as pkj


def _label(session_id, stim, custom=None):
    return custom if custom else f"Session {session_id}, stim {stim}"


def _save_pdf_png(fig, path_no_ext: Path, dpi=200):
    """PDF (vector, for the paper) + PNG (quick preview) -- not SVG, since
    there's no need here for the editable-text tradeoffs that _savefig in
    visualization.py/pk_j_analysis.py exists for."""
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_no_ext) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path_no_ext) + ".png", dpi=dpi, bbox_inches="tight")


def _pearson_r(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _write_summary_txt(output_path: Path, header: str, blocks) -> Path:
    """Write a plain-text ``<output_path>.txt`` alongside the PDF/PNG, with
    one paragraph-ready block of numbers per session -- so results-paragraph
    figures (indices/values of peaks & troughs, whether a session was
    "collective", fit quality, etc.) don't have to be re-read off the plot
    by hand every time the upstream data changes."""
    text = header + "\n" + "=" * len(header) + "\n\n" + "\n\n".join(blocks) + "\n"
    txt_path = Path(str(output_path) + ".txt")
    txt_path.write_text(text)
    return txt_path


# ---------------------------------------------------------------------------
# Model-quality summary comparison
# ---------------------------------------------------------------------------

def _find_one_stim_dir(data_folder: Path, session_id: str, stim: int) -> Path:
    # Sessions live under "<data_folder>/<session_id>_results/..." -- search
    # that narrower subtree first so this doesn't walk the whole data folder
    # on every call (this runs twice per comparison, and is meant to be
    # looped over many comparisons from a wrapper shell script).
    scoped_root = data_folder / f"{session_id}_results"
    search_root = scoped_root if scoped_root.is_dir() else data_folder

    matches = [d for d, sid, stim_num in find_stim_dirs(search_root, sessions={session_id})
              if stim_num == str(stim)]
    if not matches:
        raise FileNotFoundError(
            f"No model_quality_summary CSVs found for session {session_id} stim {stim} "
            f"under {search_root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple stim dirs matched session {session_id} stim {stim}: "
            + ", ".join(str(m) for m in matches))
    return matches[0]


def _model_quality_metrics_block(session_id, stim, custom_label, pw, tr, jv, pk, N) -> str:
    """One results-paragraph-ready block of numbers for a single session's
    model-quality panels: fit quality (Pearson r for each correlation order),
    J-coupling statistics, the P(K) Ising-vs-independent metrics, and the
    resulting collective-behavior call (same rule used in pk_j_analysis.py)."""
    pk_mets = pkj._pk_metrics(pk)
    is_collective = bool(pk_mets["ising_closer"] and pk_mets["ising_indep_dist"] > pkj.DIST_THRESHOLD)
    r_pairwise      = _pearson_r(pw["pairwise_corr_orig"], pw["pairwise_corr_model"])
    r_triplet_ising = _pearson_r(tr["triplet_corr_orig"], tr["triplet_corr_model"])
    r_triplet_indep = _pearson_r(tr["triplet_corr_orig"], tr["triplet_corr_independent"])
    J = jv["J"].values.astype(float)
    K = pk["K"].values.astype(float)

    return "\n".join([
        _label(session_id, stim, custom_label) + ":",
        f"  N neurons:                                          {N}",
        f"  N pairwise couplings (J):                           {len(J)}",
        f"  J coupling:  mean={np.mean(J):+.4f}  std={np.std(J):.4f}  "
        f"min={np.min(J):+.4f}  max={np.max(J):+.4f}",
        f"  Pairwise correlation r (measured vs reconstructed): {r_pairwise:.4f}",
        f"  Triplet correlation r (measured vs Ising):          {r_triplet_ising:.4f}",
        f"  Triplet correlation r (measured vs independent):    {r_triplet_indep:.4f}",
        f"  P(K) fit: r_ising={pk_mets['r_ising']:.4f}  r_independent={pk_mets['r_independent']:.4f}  "
        f"sum|P_ising - P_independent|={pk_mets['ising_indep_dist']:.4f}",
        f"  Ising at least as close as independent (tol={pkj.R_TOL:g}): {pk_mets['ising_closer']}",
        f"  Collective behavior (Ising closer AND dist > {pkj.DIST_THRESHOLD:g}): {is_collective}",
        f"  K range: 0-{int(K.max())}",
    ])


def build_model_quality_comparison(data_folder: Path, session1, stim1, session2, stim2,
                                   output_path: Path, max_k_plot=None,
                                   label1=None, label2=None):
    """Side-by-side model-quality summary (panels A1-D1 | A2-D2). Also writes
    a companion ``<output_path>.txt`` with the numbers behind each panel."""
    dir1 = _find_one_stim_dir(data_folder, session1, stim1)
    dir2 = _find_one_stim_dir(data_folder, session2, stim2)

    def _load(stim_dir):
        pw   = pd.read_csv(stim_dir / "model_quality_summary_pairwise.csv")
        tr   = pd.read_csv(stim_dir / "model_quality_summary_triplet.csv")
        jv   = pd.read_csv(stim_dir / "model_quality_summary_J_values.csv")
        pk   = pd.read_csv(stim_dir / "model_quality_summary_P_K.csv")
        meta = pd.read_csv(stim_dir / "model_quality_summary_P_K_metadata.csv")
        return pw, tr, jv, pk, int(meta["N_neurons"].iloc[0])

    pw1, tr1, jv1, pk1, N1 = _load(dir1)
    pw2, tr2, jv2, pk2, N2 = _load(dir2)

    # Same per-figure overrides as plot_model_quality_summary_from_arrays,
    # applied explicitly here (rather than relying on whichever module's
    # module-level rcParams.update() happened to run last at import time).
    with plt.rc_context({
        "axes.grid": False,
        "font.size": 21, "axes.titlesize": 24, "axes.labelsize": 21,
        "xtick.labelsize": 19, "ytick.labelsize": 19,
        "axes.unicode_minus": False,
    }):
        fig = plt.figure(figsize=(18.0, 25.0), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.08)
        # constrained-layout's own wspace/hspace engine setting has no visible
        # effect on the gap between these two top-level columns (verified: the
        # measured gap was identical for wspace values from 0.01 to 20 -- some
        # other constraint from the nested 4x1 subgridspecs is fixing it
        # instead). An explicit empty spacer column gives direct, reliable
        # control over the gap instead of fighting that.
        outer = fig.add_gridspec(1, 3, width_ratios=[10, 0.8, 10])

        columns = [
            (outer[0, 0], session1, stim1, pw1, tr1, jv1, pk1, N1, "1", label1),
            (outer[0, 2], session2, stim2, pw2, tr2, jv2, pk2, N2, "2", label2),
        ]
        for spec, sess, stim, pw, tr, jv, pk, N, suffix, custom_label in columns:
            axes = _draw_model_quality_panels(
                fig, spec,
                pw["pairwise_corr_orig"].values, pw["pairwise_corr_model"].values,
                jv["J"].values,
                tr["triplet_corr_orig"].values, tr["triplet_corr_model"].values,
                tr["triplet_corr_independent"].values,
                pk["K"].values, pk["P_data"].values, pk["P_ising"].values,
                pk["P_independent"].values, N,
                max_k_plot=max_k_plot, panel_suffix=suffix,
            )
            # Aligned per column (not across both), since each column's
            # required label offset only depends on its own tick-label
            # widths -- combining columns would needlessly widen whichever
            # one has narrower tick labels to match the other.
            fig.align_ylabels(axes)

        _save_pdf_png(fig, output_path)
        plt.close(fig)

    header = (f"Model-quality comparison: {_label(session1, stim1, label1)}"
             f"  vs  {_label(session2, stim2, label2)}")
    blocks = [
        _model_quality_metrics_block(session1, stim1, label1, pw1, tr1, jv1, pk1, N1),
        _model_quality_metrics_block(session2, stim2, label2, pw2, tr2, jv2, pk2, N2),
    ]
    _write_summary_txt(output_path, header, blocks)


# ---------------------------------------------------------------------------
# pk_analysis per-session comparison
# ---------------------------------------------------------------------------

def _find_per_reach_csv(data_folder: Path, session_id: str, stim: int) -> str:
    scoped_root = data_folder / f"{session_id}_results"
    search_root = scoped_root if scoped_root.is_dir() else data_folder

    all_paths = find_file_recursive(str(search_root), "per_reach_state.csv", verbose=False)
    matches = [p for p in all_paths
              if pkj._stim_from_path(p) == stim and pkj._session_id_from_path(p) == session_id]
    if not matches:
        raise FileNotFoundError(
            f"No per_reach_state.csv found for session {session_id} stim {stim} "
            f"under {search_root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple per_reach_state.csv matched session {session_id} stim {stim}: "
            + ", ".join(matches))
    return matches[0]


def _pk_analysis_metrics_block(session_id, stim, custom_label, data) -> str:
    """One results-paragraph-ready block of numbers for a single session's
    pk_analysis panels: indices/values of the velocity, acceleration,
    firing-rate and J/H peaks (in both bin index and ms), the J-vs-kinematics
    timing classification, the P(K) fit, and the collective-behavior call."""
    bin_ms = pkj.BIN_MS

    def _val_at(ts, idx):
        return float(ts[int(idx)]) if idx is not None and len(ts) > int(idx) else float("nan")

    def _ms(idx):
        return idx * bin_ms if idx is not None else float("nan")

    vel_ts, accel_ts, fr_ts, j_ts, h_ts = (
        data["vel_ts"], data["accel_ts"], data["fr_ts"], data["j_ts"], data["h_ts"])
    pk_mets = data["pk_mets"]

    lines = [_label(session_id, stim, custom_label) + ":"]
    lines.append(f"  Analysis window: {data['w_lo'] * bin_ms:.0f}-{data['w_hi'] * bin_ms:.0f} ms "
                f"(bins {data['w_lo']}-{data['w_hi']})")
    lines.append(f"  Peak velocity:      {_val_at(vel_ts, data['vel_peak_idx']):+.3f} cm/s"
                f"    at t={_ms(data['vel_peak_idx']):.0f} ms (bin {data['vel_peak_idx']})")
    lines.append(f"  Peak acceleration:  {_val_at(accel_ts, data['accel_peak_idx']):+.3f} cm/s^2"
                f" at t={_ms(data['accel_peak_idx']):.0f} ms (bin {data['accel_peak_idx']})")
    if len(fr_ts):
        lines.append(f"  Mean firing rate: {np.nanmean(fr_ts):.4f}")
    if data["has_fr_peak"]:
        lines.append(f"  Firing-rate peak:   {_val_at(fr_ts, data['fr_peak_idx']):.4f}"
                    f" at t={_ms(data['fr_peak_idx']):.0f} ms (bin {data['fr_peak_idx']})")
    else:
        lines.append("  Firing-rate peak:   none detected in window")
    if data["has_fr_trough"]:
        lines.append(f"  Firing-rate trough: {_val_at(fr_ts, data['fr_trough_idx']):.4f}"
                    f" at t={_ms(data['fr_trough_idx']):.0f} ms (bin {data['fr_trough_idx']})")
    else:
        lines.append("  Firing-rate trough: none detected in window")
    if data["has_j_peak"]:
        lines.append(f"  J-coupling peak: {_val_at(j_ts, data['j_peak_idx']):.4f}"
                    f" at t={_ms(data['j_peak_idx']):.0f} ms (bin {data['j_peak_idx']})"
                    f"  (ratio={data['j_peak_ratio']:.3f})")
    else:
        lines.append(f"  J-coupling peak: none detected (ratio={data['j_peak_ratio']:.3f})")
    if data["has_h_peak"] and len(h_ts):
        lines.append(f"  H-field peak: {_val_at(h_ts, data['h_peak_idx']):.4f}"
                    f" at t={_ms(data['h_peak_idx']):.0f} ms (bin {data['h_peak_idx']})")
    else:
        lines.append("  H-field peak: none detected")
    lines.append(f"  Timing classification (J peak vs kinematic peaks): {data['timing']}")
    lines.append(f"  P(K) fit: r_ising={pk_mets['r_ising']:.4f}  r_independent={pk_mets['r_independent']:.4f}  "
                f"sum|P_ising - P_independent|={pk_mets['ising_indep_dist']:.4f}")
    lines.append(f"  Collective behavior (Ising closer AND dist > {pkj.DIST_THRESHOLD:g}): "
                f"{data['is_collective']}")
    return "\n".join(lines)


def build_pk_analysis_comparison(data_folder: Path, session1, stim1, session2, stim2,
                                 output_path: Path, window=(350, 475),
                                 peak_threshold=1.75, smooth_sigma=5.0,
                                 label1=None, label2=None):
    """Side-by-side pk_analysis per-session figure (panels A1-E1 | A2-E2).
    Also writes a companion ``<output_path>.txt`` with the numbers behind
    each panel (peak/trough indices & values, timing, collective call)."""
    w_lo, w_hi = window
    path1 = _find_per_reach_csv(data_folder, session1, stim1)
    path2 = _find_per_reach_csv(data_folder, session2, stim2)

    data1 = pkj.load_session_pk_series(path1, stim1, w_lo, w_hi, peak_threshold, smooth_sigma)
    data2 = pkj.load_session_pk_series(path2, stim2, w_lo, w_hi, peak_threshold, smooth_sigma)
    if data1 is None:
        raise RuntimeError(f"Could not compute pk_analysis series for session {session1} stim {stim1}")
    if data2 is None:
        raise RuntimeError(f"Could not compute pk_analysis series for session {session2} stim {stim2}")

    # Same per-figure overrides as pk_j_analysis.py's module-level rcParams,
    # applied explicitly (see comment in build_model_quality_comparison).
    with plt.rc_context({
        "axes.grid": False,
        "font.size": 26, "axes.titlesize": 30, "axes.labelsize": 26,
        "xtick.labelsize": 23, "ytick.labelsize": 23,
        "legend.fontsize": 23, "figure.titlesize": 30,
    }):
        fig = plt.figure(figsize=(32, 26), layout="constrained")
        pkj._tighten(fig, hspace=0.04, wspace=0.0)
        # See the matching comment in build_model_quality_comparison -- the
        # layout engine's wspace has no effect here either, so an explicit
        # empty spacer column controls the gap instead.
        outer = fig.add_gridspec(1, 3, width_ratios=[10, 0.8, 10])

        columns = [
            (outer[0, 0], session1, stim1, data1, "1", label1),
            (outer[0, 2], session2, stim2, data2, "2", label2),
        ]
        for spec, sess, stim, data, suffix, custom_label in columns:
            axes = pkj._draw_pk_session_panels(
                fig, spec,
                data["vel_ts"], data["accel_ts"], data["fr_ts"], data["energy_ts"], data["j_ts"],
                data["w_lo"], data["w_hi"], data["vel_peak_idx"], data["accel_peak_idx"],
                data["has_j_peak"], data["j_peak_idx"],
                h_ts=data["h_ts"] if len(data["h_ts"]) > 0 else None,
                h_peak_idx=data["h_peak_idx"],
                fr_peak_idx=data["fr_peak_idx"], fr_trough_idx=data["fr_trough_idx"],
                spike_times=data["spike_times"], spike_labels=data["spike_labels"],
                panel_suffix=suffix,
            )
            fig.align_ylabels(axes)

        _save_pdf_png(fig, output_path)
        plt.close(fig)

    header = (f"pk_analysis comparison: {_label(session1, stim1, label1)}"
             f"  vs  {_label(session2, stim2, label2)}")
    blocks = [
        _pk_analysis_metrics_block(session1, stim1, label1, data1),
        _pk_analysis_metrics_block(session2, stim2, label2, data2),
    ]
    _write_summary_txt(output_path, header, blocks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build side-by-side, panel-lettered comparison figures "
                    "(model-quality summary + pk_analysis) for two session/stim pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_folder", required=True,
                   help="Root directory containing both the model_quality_summary CSVs "
                        "and per_reach_state.csv files (e.g. outputs/energy_decomp_Jul_10_svg).")
    p.add_argument("--session1", required=True, help="First (left) session ID, e.g. 210421.")
    p.add_argument("--stim1", type=int, default=0, help="Stim index for session1.")
    p.add_argument("--session2", required=True, help="Second (right) session ID, e.g. 210422.")
    p.add_argument("--stim2", type=int, default=0, help="Stim index for session2.")
    p.add_argument("--label1", default=None,
                   help="Custom column label for session1 (default: 'Session <id>, stim <n>').")
    p.add_argument("--label2", default=None, help="Custom column label for session2.")
    p.add_argument("--output_dir", default=None,
                   help="Directory to write comparison PDFs/PNGs to "
                        "(default: <data_folder>/comparisons).")
    p.add_argument("--max_k_plot", type=int, default=None,
                   help="Limit the model-quality P(K) x-axis to [0, max_k_plot] (default: 0..N).")
    p.add_argument("--window", nargs=2, type=int, metavar=("LO", "HI"), default=[350, 475],
                   help="pk_analysis peak-search window (bin indices).")
    p.add_argument("--peak_threshold", type=float, default=1.75)
    p.add_argument("--smooth_sigma", type=float, default=5.0)
    p.add_argument("--skip_model_quality", action="store_true",
                   help="Don't build the model-quality summary comparison.")
    p.add_argument("--skip_pk_analysis", action="store_true",
                   help="Don't build the pk_analysis comparison.")
    return p.parse_args()


def main():
    args = parse_args()

    data_folder = Path(args.data_folder).expanduser().resolve()
    if not data_folder.is_dir():
        print(f"Error: data folder does not exist: {data_folder}", file=sys.stderr)
        return 1

    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                 else data_folder / "comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.session1}_stim{args.stim1}_vs_{args.session2}_stim{args.stim2}"
    ok = True

    if not args.skip_model_quality:
        mq_path = output_dir / f"model_quality_compare_{tag}"
        try:
            build_model_quality_comparison(
                data_folder, args.session1, args.stim1, args.session2, args.stim2,
                mq_path, max_k_plot=args.max_k_plot, label1=args.label1, label2=args.label2,
            )
            print(f"OK    model-quality comparison -> {mq_path}.pdf / .png / .txt")
        except Exception as e:
            print(f"[ERROR] model-quality comparison failed: {e}", file=sys.stderr)
            ok = False

    if not args.skip_pk_analysis:
        pk_path = output_dir / f"pk_analysis_compare_{tag}"
        try:
            build_pk_analysis_comparison(
                data_folder, args.session1, args.stim1, args.session2, args.stim2,
                pk_path, window=tuple(args.window),
                peak_threshold=args.peak_threshold, smooth_sigma=args.smooth_sigma,
                label1=args.label1, label2=args.label2,
            )
            print(f"OK    pk_analysis comparison   -> {pk_path}.pdf / .png / .txt")
        except Exception as e:
            print(f"[ERROR] pk_analysis comparison failed: {e}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
