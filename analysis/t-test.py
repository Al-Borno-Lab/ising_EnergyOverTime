import sys
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from itertools import combinations

# Usage: python t-test.py <csv_file>
if len(sys.argv) != 2:
    print("Usage: python t-test.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Remove rows where 'folder_level_3' is NaN
df = df[df['folder_level_3'].notna()]

group_col = 'folder_level_3'
value_col = 'Critical_Temperature'

groups = df[group_col].unique()

output_lines = []
output_lines.append(f"Pairwise t-tests on '{value_col}' grouped by '{group_col}':\n")

for g1, g2 in combinations(groups, 2):
    vals1 = df[df[group_col] == g1][value_col]
    vals2 = df[df[group_col] == g2][value_col]
    t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)  # Welch's t-test
    output_lines.append(f"{g1} vs {g2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")

output_lines.append("\nOne-way ANOVA across all groups:")
grouped_values = [df[df[group_col] == g][value_col] for g in groups]
F_stat, p_anova = f_oneway(*grouped_values)
output_lines.append(f"F-statistic = {F_stat:.4f}, p-value = {p_anova:.4e}")

with open("t_test_results.txt", "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print("Results written to t_test_results.txt")
