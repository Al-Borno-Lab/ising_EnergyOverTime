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

# Filter for only 'full_reach'
df_full_reach = df_merged[df_merged['folder_level_3'] == 'full_reach']

# Compute covariance matrix using both original and filtered energy
cols_for_cov = ['Mean_Position', 'Mean_Energy_LP', 'Mean_Firing_Rate']
cov_matrix = df_full_reach[cols_for_cov].cov()

print("Covariance matrix for Mean_Position, Mean_Energy (original and low-pass filtered), Mean_Firing_Rate (full_reach only):")
print(cov_matrix)

# Save to file
cov_matrix.to_csv("covariance_matrix_full_reach.csv")
print("Covariance matrix saved to covariance_matrix_full_reach.csv")

# Select a unique trial (first unique combination of folder_level_1 and folder_level_2)
unique_trial = df_full_reach[['folder_level_1', 'folder_level_2']].drop_duplicates().iloc[0]
trial_df = df_full_reach[
    (df_full_reach['folder_level_1'] == unique_trial['folder_level_1']) &
    (df_full_reach['folder_level_2'] == unique_trial['folder_level_2'])
].sort_values(by='Time')

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Energy subplot
axs[0].plot(trial_df['Time'], trial_df['Mean_Energy'], label='Mean_Energy (original)', alpha=0.7)
axs[0].plot(trial_df['Time'], trial_df['Mean_Energy_LP'], label='Mean_Energy (low-pass)', linewidth=2)
axs[0].set_ylabel('Energy')
axs[0].legend()
axs[0].set_title('Energy over Time (one full_reach trial)')

# Firing Rate subplot
axs[1].plot(trial_df['Time'], trial_df['Mean_Firing_Rate'], color='g', label='Mean_Firing_Rate')
axs[1].set_ylabel('Firing Rate')
axs[1].legend()
axs[1].set_title('Firing Rate over Time (one full_reach trial)')

# Position subplot
axs[2].plot(trial_df['Time'], trial_df['Mean_Position'], color='r', label='Mean_Position')
axs[2].set_ylabel('Position')
axs[2].set_xlabel('Time')
axs[2].legend()
axs[2].set_title('Position over Time (one full_reach trial)')

plt.tight_layout()
plt.show()

print('Plot saved as energy_firing_position_plot.png')

# Print variance for verification
print('Variance of Mean_Energy (original):', trial_df['Mean_Energy'].var())
print('Variance of Mean_Energy_LP (low-pass):', trial_df['Mean_Energy_LP'].var())

# Optional: print first few values for visual check
print(trial_df[['Time', 'Mean_Energy', 'Mean_Energy_LP']].head())

# Compute per-reach covariance matrices and save to CSV (only for full_reach)
cov_rows = []
for keys, group in df_merged.groupby(['folder_level_1', 'folder_level_2', 'folder_level_3']):
    if keys[2] != 'full_reach':
        continue
    cov = group[['Mean_Position', 'Mean_Energy_LP', 'Mean_Firing_Rate']].cov()
    cov_row = {
        'folder_level_1': keys[0],
        'folder_level_2': keys[1],
        'folder_level_3': keys[2],
        'cov_Mean_Position_Mean_Position': cov.loc['Mean_Position', 'Mean_Position'],
        'cov_Mean_Position_Mean_Energy_LP': cov.loc['Mean_Position', 'Mean_Energy_LP'],
        'cov_Mean_Position_Mean_Firing_Rate': cov.loc['Mean_Position', 'Mean_Firing_Rate'],
        'cov_Mean_Energy_LP_Mean_Energy_LP': cov.loc['Mean_Energy_LP', 'Mean_Energy_LP'],
        'cov_Mean_Energy_LP_Mean_Firing_Rate': cov.loc['Mean_Energy_LP', 'Mean_Firing_Rate'],
        'cov_Mean_Firing_Rate_Mean_Firing_Rate': cov.loc['Mean_Firing_Rate', 'Mean_Firing_Rate'],
    }
    cov_rows.append(cov_row)

cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv('per_reach_covariances.csv', index=False)
print('Per-reach covariance matrix saved to per_reach_covariances.csv') 