#!/bin/bash
#SBATCH --job-name=ising_task
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=./logs_test/ising_task_%j.log

singularity exec ~/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python main.py \
    --matlab_file  /data001/projects/enserrog/AbigailData/energy_over_time/210421_fChR2_bin10_spike.mat \
    --output_dir /home/enserrog/ising_EnergyOverTime/energy_over_time_script/test_output_4_updated_code_smaller \
    --bin_size 1 \
    --sample_size 100000 \
    --n_cpus 64 \
    --max_iter 500 \
    --eta 0.05 \
    --temp_min 0.1 \
    --temp_max 2.0 \
    --temp_step 0.05 \
    --metropolis_samples 100000 \
    --truncate_idx_l 100 \
    --truncate_idx 800 \
    --confidence 0.8 \
    --firing_rate_window 1

# singularity exec ~/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh \
# python aggregate_model_quality_plots.py /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_model_quality_summary_plot