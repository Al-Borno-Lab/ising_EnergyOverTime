#!/usr/bin/env bash
# Entry point for E12. Local (laptop) run, about 10 minutes on 12 cores; no cluster needed.
#   PY=/path/to/python-with-numpy-scipy-pandas-matplotlib-statsmodels-sklearn  DATA=/dir/with/spikes_v2.npz  bash 12_field_vs_rate_indicator/code/run.sh [validate|real|sens|report|all]
# Run from the repo root. spikes_v2.npz is not committed (see 09_claude_science_handoff/HANDOFF_CLAUDE_CODE.md).
set -euo pipefail
EXP="12_field_vs_rate_indicator"; PY="${PY:-python3}"; DATA="${DATA:?set DATA to the directory holding spikes_v2.npz}"
LM="09_claude_science_handoff/assets/tables/phase_landmarks.csv"; NPROC="${NPROC:-8}"
R="$EXP/results"; A="$EXP/assets/tables"; mkdir -p "$R" "$A" "$EXP/assets/figures"
step="${1:-all}"

if [[ "$step" == validate || "$step" == all ]]; then
  bash tools/log_run.sh "$EXP" submitted "METHOD VALIDATION (synthetic): validate_synthetic.py" - "$R/synthetic"
  (cd "$EXP/code" && $PY validate_synthetic.py --out "../results/synthetic" --nproc "$NPROC" > "../results/synthetic.log" 2>&1)
  cp "$R/synthetic/synthetic_validation.csv" "$A/synthetic_validation.csv"
  for s in dipJ0 dipJ null; do cp "$R/synthetic/$s/primary_family.csv" "$A/synthetic_${s}_primary_family.csv"; done
  bash tools/log_run.sh "$EXP" finished "synthetic validation done -> assets/tables/synthetic_validation.csv"
fi
if [[ "$step" == real || "$step" == all ]]; then
  bash tools/log_run.sh "$EXP" submitted "REAL primary: compute_signals.py (exclude_k==0) + stats_e12.py" - "$R/primary"
  (cd "$EXP/code" && $PY compute_signals.py --npz "$DATA/spikes_v2.npz" --landmarks "../../$LM" --out ../results/primary --nproc "$NPROC" > ../results/primary.log 2>&1 \
     && $PY stats_e12.py --rows ../results/primary/rows.csv --out ../results/primary)
  for f in session_metrics signal_summary primary_family secondary_vs_control descriptive_contrasts; do cp "$R/primary/$f.csv" "$A/primary_$f.csv"; done
  bash tools/log_run.sh "$EXP" finished "real primary done"
fi
if [[ "$step" == sens || "$step" == all ]]; then
  for v in "allreaches:--keep-all" "minN5:--min-n 5" "sigma2:--sigma 2" "sigma5:--sigma 5"; do
    n="${v%%:*}"; fl="${v#*:}"
    bash tools/log_run.sh "$EXP" submitted "REAL sensitivity $n ($fl)" - "$R/$n"
    (cd "$EXP/code" && $PY compute_signals.py --npz "$DATA/spikes_v2.npz" --landmarks "../../$LM" --out "../results/$n" --nproc "$NPROC" $fl > "../results/$n.log" 2>&1 \
       && $PY stats_e12.py --rows "../results/$n/rows.csv" --out "../results/$n")
    cp "$R/$n/primary_family.csv" "$A/sens_${n}_primary_family.csv"
    bash tools/log_run.sh "$EXP" finished "sensitivity $n done"
  done
fi
if [[ "$step" == report || "$step" == all ]]; then
  (cd "$EXP/code" && $PY make_figures.py --results ../results --assets ../assets)
fi
