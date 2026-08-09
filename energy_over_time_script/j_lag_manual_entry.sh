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


python run_decision_tree.py \
    --csv ./notes/decision_dataset/decision_tree_dataset.csv \
    --output_dir ./notes/decision_dataset/run_2_all_sessions_w350_475_updated \
    --max_depth_plot 6 \
    --no_cv


# P(K) × J-peak hypothesis analysis
# Tests: does a J peak occur when Ising fits data well AND differs from independent?
python pk_j_analysis.py \
    --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_may12_stimDecon/ \
    --window 350 475 \
    --stim_min 0 --stim_max_exclusive 3 \
    --peak_threshold 1.75 \
    --output_dir ./notes/pk_analysis/all_sessions_w350_475

# Same but restricted to selected sessions
# python pk_j_analysis.py \
#     --data_folder /data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_may12_stimDecon/ \
#     --window 350 475 \
#     --stim_min 0 --stim_max_exclusive 3 \
#     --peak_threshold 1.75 \
#     --sessions 210425 210511 210515 220515 220516 220517 220518 220519 220520 \
#     --output_dir ./notes/pk_analysis/selected_sessions_w350_475