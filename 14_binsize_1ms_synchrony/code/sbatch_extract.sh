#!/bin/bash
#SBATCH --account=ceas-alborno
#SBATCH --partition=math-alderaan
#SBATCH --job-name=e14_extract
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# E14 step 1: copy the 21 raw files under $HOME (the container cannot see /data001), extract 1-ms spikes + run the data gates, then
# delete the raw copies ONLY IF the extraction succeeded. Run from ~/ising_wt/e14/14_binsize_1ms_synchrony.
set -euo pipefail
cd "$HOME/ising_wt/e14/14_binsize_1ms_synchrony"
SRC=/data001/projects/enserrog/dylan/AbigailUpdated; RAW=results/raw; OUT=results/spk
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
SPK="$HOME/ising_EnergyOverTime/energy_over_time_organized/09_claude_science_handoff/data/spikes_v2.npz"
mkdir -p $RAW $OUT
for f in $(ls $SRC | grep -v 'WT2\|220519_DJC000\|220520_DJC002'); do cp -n $SRC/$f $RAW/; done
echo "copied $(ls $RAW | wc -l) raw files, code commit $(git rev-parse --short HEAD)"
ls $RAW | sed -E 's/Dylan_([0-9]{6})_.*/\1/' | xargs -P 8 -I{} singularity exec $SIF /entrypoint.sh python code/extract_spiketimes.py $RAW $SPK $OUT {}
python3 - <<PY
import csv,glob
rows=[r for f in glob.glob("$OUT/gates_*.csv") for r in csv.DictReader(open(f))]
bad=[r['session'] for r in rows if r['gate_pass']!='True']
print("sessions", len(rows), "gate failures:", bad)
open("$OUT/gate_summary.txt","w").write("sessions %d failures %s\n"%(len(rows),bad))
PY
rm -rf $RAW results/raw1
echo "raw copies deleted"
