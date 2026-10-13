#!/usr/bin/env bash
# Entry point for E11 (stimulation model: does optogenetic perturbation change coupling, kappa?).
# Reproduces every output from the raw data. Run from the repo root on math-alderaan (campus VPN). Heavy steps go through sbatch.
# The exact commands, job IDs and code commits that produced the committed results are in runs/RUN_LOG.md.
#   bash 11_stimulation_kappa/code/run.sh <stage>      stages: unit_tests | laser_times | validate | full | exact | analyse
set -euo pipefail
EXP="11_stimulation_kappa"
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
PY="singularity exec $SIF /entrypoint.sh python"
RAW_SRC=/data001/projects/enserrog/dylan/AbigailUpdated            # the container cannot see /data001 from sbatch: copy under $HOME
RAW="$HOME/ising_wt/e11_raw"                                       # (21 files, 16 GB; delete after stage laser_times)
SPIKES="$HOME/ising_EnergyOverTime/energy_over_time_organized/09_claude_science_handoff/data/spikes_v2.npz"   # E09 export, gitignored
SMALL=210421,210422,210423,210425,210511,210512,210514,210622,210623
A=210515,210606,210608,210614,$SMALL; B=210619,210620,220515,220516; C=220517,220518,220519,220520
mkdir -p "$EXP/results"
case "${1:?stage}" in
  unit_tests)  $PY $EXP/code/test_stimfit.py ;;                                        # likelihood vs brute force, gradient check
  laser_times)                                                                           # step 1: ReachS.stimtime -> assets/tables/laser_onsets.csv
    mkdir -p "$RAW"; for f in $(ls "$RAW_SRC" | grep -v 'WT2\|220519_DJC000\|220520_DJC002'); do cp -n "$RAW_SRC/$f" "$RAW/"; done
    srun -A ceas-alborno -p math-alderaan -c 4 $PY $EXP/code/extract_laser_onsets.py "$RAW" $EXP/results/reachs_raw.csv
    srun -A ceas-alborno -p math-alderaan -c 2 $PY $EXP/code/merge_laser_onsets.py $EXP/results/reachs_raw.csv "$SPIKES" $EXP/assets/tables/laser_onsets.csv ;;
  validate)    sbatch --cpus-per-task=60 $EXP/code/sbatch_validate.sh results/validation ;;   # SYNTHETIC, criteria in VALIDATION_CRITERIA_2026-10-12.md
  full)                                                                                  # step 5: 3 jobs x 64 CPUs, ~590 CPU-h
    cd $EXP; for S in "$A" "$B" "$C"; do sbatch --cpus-per-task=64 --mem=160G --time=08:00:00 --job-name=e11_full code/sbatch_run.sh results/full "$S"; done ;;
  exact)       cd $EXP; sbatch --cpus-per-task=2 --mem=16G --time=08:00:00 --job-name=e11_exact18 code/sbatch_run.sh results/exact 210515 --tasks=exactcheck --exact=210515 ;;
  analyse)     python3 $EXP/code/analyze_stimfit.py $EXP/results/full $EXP/assets/tables && python3 $EXP/code/make_figures.py $EXP/assets $EXP/results/full ;;   # needs numpy/scipy/pandas/statsmodels/matplotlib; run locally on copied results/
  *) echo "unknown stage" >&2; exit 1 ;;
esac
