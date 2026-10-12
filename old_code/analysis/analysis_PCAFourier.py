import sys
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Usage: python analysis_script.py firing_rate.csv energy.csv kinematics.csv

if len(sys.argv) != 4:
    print("Usage: python analysis_script.py firing_rate.csv energy.csv kinematics.csv")
    sys.exit(1)

firing_csv = sys.argv[1]
energy_csv = sys.argv[2]
kinematics_csv = sys.argv[3]

# Read CSVs
df_firing = pd.read_csv(firing_csv)
df_energy = pd.read_csv(energy_csv)
df_kinematics = pd.read_csv(kinematics_csv)

# Merge on index columns
index_cols = ['folder_level_1', 'folder_level_2', 'folder_level_3', 'Time']
df_merged = df_firing.merge(df_energy, on=index_cols).merge(df_kinematics, on=index_cols)

# Sort by Time (important for filtering)
df_merged = df_merged.sort_values(by='Time')

# Apply low-pass Butterworth filter to Mean_Energy
# Assumes Time is evenly spaced and numeric
order = 4
cutoff = 0.01  # As a fraction of Nyquist frequency
b, a = butter(order, cutoff, btype='low', analog=False)

def apply_lowpass(series):
    return filtfilt(b, a, series)

# Apply filter within each group (to avoid filtering across different trials)
df_merged['Mean_Energy_LP'] = df_merged.groupby(['folder_level_1', 'folder_level_2', 'folder_level_3'])['Mean_Energy'].transform(apply_lowpass)
df_merged['Mean_Firing_Rate_LP'] = df_merged.groupby(['folder_level_1', 'folder_level_2', 'folder_level_3'])['Mean_Firing_Rate'].transform(apply_lowpass)
df_merged['Mean_Position_LP'] = df_merged.groupby(['folder_level_1', 'folder_level_2', 'folder_level_3'])['Mean_Position'].transform(apply_lowpass)

# Filter for only 'full_reach'
df_full_reach = df_merged[df_merged['folder_level_3'] == 'full_reach']

# DATA ACCESS GUIDE:
# ===================
# 
# The processed data is now available in the following DataFrames:
#
# 1. df_merged: Contains all merged data from firing rate, energy, and kinematics
#    - Original columns: folder_level_1, folder_level_2, folder_level_3, Time, 
#                       Mean_Firing_Rate, Mean_Energy, Mean_Position
#    - Filtered columns: Mean_Energy_LP, Mean_Firing_Rate_LP, Mean_Position_LP
#
# 2. df_full_reach: Subset of df_merged filtered for 'full_reach' trials only
#
#   folder_level_1 - Mouse
#   folder_level_2 - iteration of experiment per mouse
#   folder_level_3 - Reach component inside iteration of experiment per mouse
#
# COLUMN DESCRIPTIONS:
# 
# - folder_level_1, folder_level_2, folder_level_3: Hierarchical trial identifiers
# - Time: Time points for each measurement
# - Mean_Firing_Rate: Original firing rate data
# - Mean_Energy: Original energy data  
# - Mean_Position: Original position data
# - Mean_Energy_LP: Low-pass filtered energy data (4th order Butterworth, cutoff=0.01)
# - Mean_Firing_Rate_LP: Low-pass filtered firing rate data
# - Mean_Position_LP: Low-pass filtered position data
#
# EXAMPLE USAGE:
# 
# # Access specific trial data:
# trial_data = df_full_reach[
#     (df_full_reach['folder_level_1'] == 'your_folder1') & 
#     (df_full_reach['folder_level_2'] == 'your_folder2')
# ]
#
# # Compute correlations between filtered signals:
# correlation_matrix = df_full_reach[['Mean_Position_LP', 'Mean_Energy_LP', 'Mean_Firing_Rate_LP']].corr()
#
# # Group analysis by trial:
# for (f1, f2), group in df_full_reach.groupby(['folder_level_1', 'folder_level_2']):
#     # Perform analysis on each trial group
#     pass
#
# # Save processed data:
# df_full_reach.to_csv('processed_full_reach_data.csv', index=False)
# df_merged.to_csv('processed_all_data.csv', index=False)


# TODO
# Find the Fourier decomposition of each energy path
# Normalize amplitudes of fourier
# Use PCA and look at component vectors to idenitify 