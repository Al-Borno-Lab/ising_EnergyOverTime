# python peak_near_max_velocity.py --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_Apr_16/ --sessions 210425 210511 210515 220515 220516 220517 220518 220519 220520 --stim_min 0 --stim_max_exclusive 1 --half_window 60 --output_dir ./notes/J_analysis

python peak_near_max_velocity.py \
    --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_Apr_16/ \
    --stim_min 0 --stim_max_exclusive 3 \
    --half_window 60 \
    --output_dir ./notes/J_analysis/J_analysis_peak_finder_arbitration \

python arbitration_j_many.py \
    --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_Apr_16/ \
    --stim_min 0 --stim_max_exclusive 3 \
    --output_base ./notes/J_analysis/J_analysis_arbitration_j_many \

# Same command but restricted to selected sessions only
python arbitration_j_many.py \
    --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_Apr_16/ \
    --stim_min 0 --stim_max_exclusive 3 \
    --output_base ./notes/J_analysis/J_analysis_arbitration_j_manual \
    --sessions 210425 210511 210515 220515 220516 220517 220518 220519 220520 \


python build_decision_tree_dataset.py \
    --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_Apr_16/ \
    --window 350 475 \
    --stim_min 0 --stim_max_exclusive 3 \
    --peak_threshold 1.75 \
    --output ./notes/decision_dataset/decision_tree_dataset.csv