#!/usr/bin/env python
# coding: utf-8
"""
Batch arbitration over all sessions under a data folder (full_reach per_reach_state.csv).

Equivalent to arbitration_many.ipynb: recursive CSV discovery, multiprocessing extrema
extraction, within-session energy vs firing-rate test with plots.

Run from the energy_over_time_script directory (same as the notebook):

    python arbitration_many.py --data_folder /path/to/energy_decomp_1/

See --help for window, rep range, and output options.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.arbitration import within_session_test_with_plots
from src.processing import process_session
from src.util import find_file_recursive


def extract_stats_to_df(stats_dict, **kwargs):
    """Flatten arbitration stats dict to one dataframe row (optional extra columns e.g. stim=0)."""
    row = {
        "energy_wins": stats_dict["energy_wins"],
        "firing_wins": stats_dict["firing_wins"],
        "ties": stats_dict["ties"],
        "n_sessions": stats_dict["n_sessions"],
        "energy_win_pct": stats_dict["energy_win_pct"],
        "mean_energy_distance": stats_dict["mean_energy_distance"],
        "mean_firing_distance": stats_dict["mean_firing_distance"],
        "std_energy_distance": stats_dict["std_energy_distance"],
        "std_firing_distance": stats_dict["std_firing_distance"],
        "mean_covariance": stats_dict["mean_covariance"],
        "mean_correlation": stats_dict["mean_correlation"],
        "energy_min_wins": stats_dict["extremum_wins"]["energy_min"],
        "energy_max_wins": stats_dict["extremum_wins"]["energy_max"],
        "firing_min_wins": stats_dict["extremum_wins"]["firing_min"],
        "firing_max_wins": stats_dict["extremum_wins"]["firing_max"],
        **kwargs,
    }
    return pd.DataFrame([row])


def _rep_from_session_path(session: str) -> str:
    """Parse rep index string from path like .../210421_results/210421_rep1/full_reach/..."""
    return session.split("_results/")[1].split("/")[0].split("_")[1][3:]


def _session_id_from_path(session: str) -> str:
    """Six-digit session id from path prefix before '_results'."""
    return session.split("_results")[0][-6:]


def parse_args():
    p = argparse.ArgumentParser(description="Batch within-session arbitration over per_reach_state.csv files.")
    p.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Root folder to search recursively for per_reach_state.csv (e.g. energy_decomp_1/)",
    )
    p.add_argument(
        "--rep_start",
        type=int,
        default=1,
        help="First rep index to include (folder ..._repN); default 1",
    )
    p.add_argument(
        "--rep_end_exclusive",
        type=int,
        default=2,
        help="One past last rep index (same semantics as range(rep_start, rep_end_exclusive)); default 2",
    )
    p.add_argument(
        "--window",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=[390, 410],
        help="Time index window for extrema search (default: 390 410)",
    )
    p.add_argument(
        "--stim_min",
        type=int,
        default=0,
        help="Inclusive minimum stimulus index (default 0)",
    )
    p.add_argument(
        "--stim_max_exclusive",
        type=int,
        default=3,
        help="Exclusive max stimulus index (default 3 → stimuli 0,1,2)",
    )
    p.add_argument(
        "--output_base",
        type=str,
        default=None,
        help=(
            "Base directory for outputs; default: "
            "./Arbitration/accel_results/within_session_test_rep_10ms_accel_wLO_HI_MCH"
        ),
    )
    p.add_argument(
        "--reference",
        choices=("acceleration", "velocity"),
        default="acceleration",
        help="Which peak index to compare distances against (notebook uses acceleration)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Pool worker count (default: multiprocessing default)",
    )
    p.add_argument(
        "--quiet_find",
        action="store_true",
        help="Less verbose output from find_file_recursive",
    )
    p.add_argument(
        "--quiet_arbitration",
        action="store_true",
        help="Less verbose output from within_session_test_with_plots",
    )
    p.add_argument(
        "--report_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If set, write arbitration_results.md into this directory "
            "(e.g. the session report folder produced by generate_session_report.py)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

_CLOSEST_LABELS = ["Energy Min", "Energy Max", "Firing Min", "Firing Max"]


def write_arbitration_markdown(all_results, all_closest, out_base, report_dir, args):
    """
    Write a self-contained arbitration_results.md into *report_dir*.

    Parameters
    ----------
    all_results  : {rep: results_dict}   from within_session_test_with_plots
    all_closest  : {rep: {stim: [e_min_wins, e_max_wins, f_min_wins, f_max_wins]}}
    out_base     : path where per-rep CSVs were saved
    report_dir   : destination directory for the markdown file
    args         : parsed CLI args (for metadata)
    """
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, "arbitration_results.md")

    lines = []
    lines += [
        "# Arbitration Results",
        "",
        f"*Generated: {date.today().isoformat()}*",
        "",
        f"**Data folder:** `{args.data_folder}`  ",
        f"**Window:** [{args.window[0]}, {args.window[1]}]  ",
        f"**Reference:** {args.reference}  ",
        f"**Reps:** {args.rep_start} – {args.rep_end_exclusive - 1}  ",
        f"**Stimuli:** {args.stim_min} – {args.stim_max_exclusive - 1}",
        "",
        "---",
        "",
    ]

    stim_range = range(args.stim_min, args.stim_max_exclusive)

    # ── Per-rep overall summary ───────────────────────────────────────────
    lines += ["## Overall Results (per Rep)", ""]
    lines += [
        "| Rep | Energy Wins | Firing Wins | Ties | N Sessions | Energy Win % |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        ov = res["overall"]
        lines.append(
            f"| {rep} | {ov['energy_wins']} | {ov['firing_wins']} | "
            f"{ov['ties']} | {ov['n_sessions']} | {ov['energy_win_pct']:.1f}% |"
        )
    lines += [""]

    # ── Per-stimulus breakdown ────────────────────────────────────────────
    lines += ["## Per-Stimulus Breakdown", ""]
    lines += [
        "| Rep | Stim | Energy Wins | Firing Wins | Ties | N Sessions | "
        "Energy Win % | Mean ΔEnergy | Mean ΔFiring |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        for stim in sorted(res["by_stimulus"].keys()):
            s = res["by_stimulus"][stim]
            lines.append(
                f"| {rep} | {stim} | {s['energy_wins']} | {s['firing_wins']} | "
                f"{s['ties']} | {s['n_sessions']} | {s['energy_win_pct']:.1f}% | "
                f"{s['mean_energy_distance']:.3f} | {s['mean_firing_distance']:.3f} |"
            )
    lines += [""]

    # ── Closest-extremum counts ───────────────────────────────────────────
    lines += [
        "## Closest Extremum to Kinematic Reference",
        "",
        f"Counts: how many sessions had each extremum closest to the "
        f"{args.reference} peak.",
        "",
        "| Rep | Stim | Energy Min | Energy Max | Firing Min | Firing Max |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, stim_map in sorted(all_closest.items()):
        for stim, cp in sorted(stim_map.items()):
            lines.append(
                f"| {rep} | {stim} | {cp[0]} | {cp[1]} | {cp[2]} | {cp[3]} |"
            )
    lines += [""]

    # ── Distance statistics ───────────────────────────────────────────────
    lines += ["## Distance Statistics", ""]
    lines += [
        "| Rep | Stim | Mean ΔEnergy | Std ΔEnergy | Mean ΔFiring | Std ΔFiring | "
        "Mean Covariance | Mean Correlation |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        for stim in sorted(res["by_stimulus"].keys()):
            s = res["by_stimulus"][stim]
            lines.append(
                f"| {rep} | {stim} | {s['mean_energy_distance']:.3f} | "
                f"{s['std_energy_distance']:.3f} | {s['mean_firing_distance']:.3f} | "
                f"{s['std_firing_distance']:.3f} | {s['mean_covariance']:.4f} | "
                f"{s['mean_correlation']:.4f} |"
            )
    lines += [""]

    # ── CSV locations ─────────────────────────────────────────────────────
    lines += ["## Output Files", ""]
    for rep in sorted(all_results.keys()):
        rep_out = os.path.join(out_base, str(rep))
        lines += [
            f"**Rep {rep}**  ",
            f"- Results CSV: `{os.path.join(rep_out, 'results.csv')}`  ",
            f"- Session summary: `{os.path.join(rep_out, 'session_summary.csv')}`  ",
            f"- Energy-win plots: `{os.path.join(rep_out, 'energy_wins')}/`  ",
            f"- Firing-win plots: `{os.path.join(rep_out, 'firing_wins')}/`  ",
            f"- Tie plots: `{os.path.join(rep_out, 'ties')}/`  ",
            "",
        ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Arbitration report written to: {md_path}")


def main() -> int:
    args = parse_args()
    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    if not os.path.isdir(data_folder):
        print(f"Error: data folder does not exist or is not a directory: {data_folder}", file=sys.stderr)
        return 1

    w_lo, w_hi = args.window
    if args.output_base is None:
        out_base = os.path.abspath(
            f"./Arbitration/accel_results/within_session_test_rep_10ms_accel_w{w_lo}_{w_hi}_MCH"
        )
    else:
        out_base = os.path.abspath(os.path.expanduser(args.output_base))

    all_reach_states = find_file_recursive(data_folder, "per_reach_state.csv", verbose=not args.quiet_find)
    if not all_reach_states:
        print(f"No per_reach_state.csv found under {data_folder}", file=sys.stderr)
        return 1

    stim_range = range(args.stim_min, args.stim_max_exclusive)

    # Collect results across all reps for the markdown report
    all_results = {}
    all_closest = {}

    for rep in range(args.rep_start, args.rep_end_exclusive):
        session_data = {}
        rep_s = str(rep)

        for session in all_reach_states:
            if "full_reach" not in session:
                continue
            try:
                if _rep_from_session_path(session) != rep_s:
                    continue
            except (IndexError, ValueError) as e:
                print(f"Skipping path (could not parse rep): {session} ({e})", file=sys.stderr)
                continue

            print(session)
            session_id = _session_id_from_path(session)
            print(f"Session_id: {session_id}")
            session_data[session_id] = pd.read_csv(session)

        if not session_data:
            print(f"No sessions matched rep={rep} and full_reach under {data_folder}", file=sys.stderr)
            continue

        window = [w_lo, w_hi]
        tasks = [
            (stim, session, session_data[session], window)
            for stim in stim_range
            for session in session_data.keys()
        ]

        stim_sessions_extrema = {}
        pool_kw = {}
        if args.workers is not None:
            pool_kw["processes"] = args.workers
        with Pool(**pool_kw) as pool:
            for stim, session, data_frame in pool.imap_unordered(process_session, tasks):
                print(f"← Received: stim={stim}, session={session}")
                stim_sessions_extrema.setdefault(stim, {})[session] = data_frame

        print("All done!")

        ref_key = "acceleration" if args.reference == "acceleration" else "velocity"
        ref_idx_attr = 1  # (value, index) tuple from find_extrema_in_range

        rep_closest = {}
        for stim in stim_range:
            if stim not in stim_sessions_extrema:
                print(f"(stim {stim}: no data)")
                continue
            closest_list = []
            closest_point = [0, 0, 0, 0]
            for session in stim_sessions_extrema[stim].keys():
                ref_idx = stim_sessions_extrema[stim][session][ref_key][ref_idx_attr]
                close_set = [
                    np.abs(ref_idx - stim_sessions_extrema[stim][session]["energy"][0]),
                    np.abs(ref_idx - stim_sessions_extrema[stim][session]["energy"][1]),
                    np.abs(ref_idx - stim_sessions_extrema[stim][session]["firing_rate"][0]),
                    np.abs(ref_idx - stim_sessions_extrema[stim][session]["firing_rate"][1]),
                ]
                idx_closest = int(np.argmin(close_set))
                closest_list.append(close_set)
                closest_point[idx_closest] += 1
            rep_closest[stim] = closest_point
            print(closest_point)
            if closest_list:
                print(np.array(closest_list))

        all_closest[rep] = rep_closest

        rep_out = os.path.join(out_base, str(rep))
        os.makedirs(rep_out, exist_ok=True)

        results = within_session_test_with_plots(
            stim_sessions_extrema,
            output_dir=rep_out,
            verbose=not args.quiet_arbitration,
        )
        all_results[rep] = results

        df = pd.concat(
            [
                extract_stats_to_df(results["by_stimulus"][stim], stim=stim, rep=rep)
                for stim in results["by_stimulus"].keys()
            ],
            ignore_index=True,
        )
        df.to_csv(os.path.join(rep_out, "results.csv"), index=False)
        print(f"Wrote {os.path.join(rep_out, 'results.csv')}")

    # Write markdown report if requested and we have results
    if args.report_dir and all_results:
        write_arbitration_markdown(all_results, all_closest, out_base, args.report_dir, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
