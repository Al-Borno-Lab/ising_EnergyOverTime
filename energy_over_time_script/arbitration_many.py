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
    return p.parse_args()


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
            print(closest_point)
            if closest_list:
                print(np.array(closest_list))

        rep_out = os.path.join(out_base, str(rep))
        os.makedirs(rep_out, exist_ok=True)

        results = within_session_test_with_plots(
            stim_sessions_extrema,
            output_dir=rep_out,
            verbose=not args.quiet_arbitration,
        )

        df = pd.concat(
            [
                extract_stats_to_df(results["by_stimulus"][stim], stim=stim, rep=rep)
                for stim in results["by_stimulus"].keys()
            ],
            ignore_index=True,
        )
        df.to_csv(os.path.join(rep_out, "results.csv"), index=False)
        print(f"Wrote {os.path.join(rep_out, 'results.csv')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
