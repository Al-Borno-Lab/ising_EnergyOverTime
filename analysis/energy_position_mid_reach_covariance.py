import sys
import pandas as pd
from scipy.signal import butter, filtfilt

# Usage: python energy_position_mid_reach_covariance.py energy.csv kinematics.csv
if len(sys.argv) != 3:
    print("Usage: python energy_position_mid_reach_covariance.py energy.csv kinematics.csv")
    sys.exit(1)

energy_csv = sys.argv[1]
kinematics_csv = sys.argv[2]

df_energy = pd.read_csv(energy_csv)
df_kinematics = pd.read_csv(kinematics_csv)

# Merge on index columns
index_cols = ['folder_level_1', 'folder_level_2', 'folder_level_3', 'Time']
df = df_energy.merge(df_kinematics, on=index_cols)

# Sort by Time (important for filtering)
df = df.sort_values(by='Time')

# Apply low-pass Butterworth filter to Mean_Energy
order = 4
cutoff = 0.01  # As a fraction of Nyquist frequency
b, a = butter(order, cutoff, btype='low', analog=False)

def apply_lowpass(series):
    return filtfilt(b, a, series)

# Apply filter within each group (to avoid filtering across different trials)
df['Mean_Energy_LP'] = df.groupby(['folder_level_1', 'folder_level_2', 'folder_level_3'])['Mean_Energy'].transform(apply_lowpass)

# Filter for full_reach and Time between 350 and 500 (inclusive)
full_df = df[(df['folder_level_3'] == 'full_reach') & (df['Time'] >= 350) & (df['Time'] <= 500)]

# Compute per-reach covariance between Mean_Energy_LP and Mean_Position
cov_rows = []
for keys, group in full_df.groupby(['folder_level_1', 'folder_level_2']):
    if group[['Mean_Energy_LP', 'Mean_Position']].shape[0] < 2:
        continue  # Need at least 2 points for covariance
    cov = group[['Mean_Energy_LP', 'Mean_Position']].cov().loc['Mean_Energy_LP', 'Mean_Position']
    cov_rows.append({
        'folder_level_1': keys[0],
        'folder_level_2': keys[1],
        'cov_Mean_Energy_LP_Mean_Position': cov
    })

cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv('full_reach_energy_position_covariances.csv', index=False)
print('Per-reach covariances saved to full_reach_energy_position_covariances.csv')

# Group by folder_level_1 and summarize
summary = cov_df.groupby('folder_level_1')['cov_Mean_Energy_LP_Mean_Position'].agg(['mean', 'std', 'count']).reset_index()
summary.to_csv('full_reach_energy_position_covariances_summary.csv', index=False)
print('Summary by folder_level_1 saved to full_reach_energy_position_covariances_summary.csv')

# Group by folder_level_1 and average the covariances
mean_covs = cov_df.groupby('folder_level_1')['cov_Mean_Energy_LP_Mean_Position'].mean().reset_index()

print('\nAverage covariance between Mean_Energy_LP (low-pass) and Mean_Position for each folder_level_1:')
for _, row in mean_covs.iterrows():
    print(f"folder_level_1: {row['folder_level_1']}, mean covariance: {row['cov_Mean_Energy_LP_Mean_Position']}") 