# Analysis Results Interpretation

This document summarizes and interprets the statistical results found in `t_test_results.txt` for the Critical_Temperature variable grouped by `folder_level_3`.

## Statistical Tests Performed

### 1. Pairwise t-tests
Pairwise t-tests were conducted to compare the means of Critical_Temperature between each pair of groups in the `folder_level_3` column. The t-test evaluates whether the difference in means between two groups is statistically significant (i.e., unlikely to have occurred by random chance).

- **t-statistic**: Measures the size of the difference relative to the variation in the data.
- **p-value**: Indicates the probability that the observed difference is due to random chance. A p-value less than 0.05 is typically considered statistically significant.

### 2. One-way ANOVA
A one-way ANOVA was performed to test if there are any statistically significant differences among the means of all groups in `folder_level_3`.

- **F-statistic**: Measures the ratio of variance between the groups to the variance within the groups.
- **p-value**: Indicates the probability that the observed differences among group means are due to random chance. A p-value less than 0.05 suggests at least one group mean is significantly different from the others.

## Results Summary

### Pairwise t-tests
All pairwise comparisons between groups (begin_reach, post_reach, mid_reach, full_reach) resulted in p-values much greater than 0.05. This means:

- **There are no statistically significant differences in Critical_Temperature between any pairs of groups.**
- The observed differences in means are likely due to random variation rather than a true effect.

### One-way ANOVA
- **F-statistic = 0.4117, p-value = 0.74465**
- The ANOVA p-value is also much greater than 0.05, indicating **no statistically significant difference in Critical_Temperature across all groups**.

## Conclusion
Based on both the pairwise t-tests and the one-way ANOVA, there is no evidence to suggest that the Critical_Temperature differs significantly between the different phases (begin_reach, post_reach, mid_reach, full_reach) in the dataset. Any observed differences are likely due to random chance. 


/Volumes/GunnarTB/mazen_lab/allMice/combined_results/energy_stim_0/master_energy_stim_0.csv

/Volumes/GunnarTB/mazen_lab/allMice/combined_results/firing_rates_stim_0/master_firing_rates_stim_0.csv 

/Volumes/GunnarTB/mazen_lab/allMice/combined_results/kinematics_stim_0/master_kinematics_stim_0.csv

## Covariance Matrix Analysis (Full Reach)

The covariance matrix below summarizes the relationships between Mean_Position, Mean_Energy, and Mean_Firing_Rate for the 'full_reach' phase:

|                | Mean_Position | Mean_Energy | Mean_Firing_Rate |
|----------------|--------------|-------------|------------------|
| Mean_Position  | 0.0446       | 0.0414      | 0.0007           |
| Mean_Energy    | 0.0414       | 6.5964      | -0.2354          |
| Mean_Firing_Rate | 0.0007     | -0.2354     | 0.0249           |

**Interpretation:**
- **Diagonal values** (e.g., 0.0446 for Mean_Position) represent the variance of each variable. Higher values indicate greater variability.
- **Off-diagonal values** represent the covariance between pairs of variables:
    - A positive value indicates that the two variables tend to increase or decrease together.
    - A negative value indicates that as one variable increases, the other tends to decrease.
    - Values close to zero suggest little to no linear relationship.

**Key findings:**
- The covariance between Mean_Position and Mean_Energy (0.0414) is positive, suggesting a weak tendency for these variables to increase together during the 'full_reach' phase.
- The covariance between Mean_Energy and Mean_Firing_Rate (-0.2354) is negative, indicating a weak inverse relationship: as energy increases, firing rate tends to decrease, or vice versa.
- The covariance between Mean_Position and Mean_Firing_Rate (0.0007) is very close to zero, suggesting almost no linear relationship between these two variables in the 'full_reach' phase.

**Conclusion:**
- The relationships between these variables during 'full_reach' are generally weak, with the strongest (though still modest) relationship being a negative covariance between Mean_Energy and Mean_Firing_Rate.
- These results provide a quantitative summary of how these behavioral and physiological measures co-vary during the full reach phase.

## Covariance Matrix Analysis (Full Reach, Low-pass Filtered Energy)

The following covariance matrix uses the low-pass filtered energy (Mean_Energy_LP) instead of the original Mean_Energy, providing insight into the relationships after removing high-frequency noise:

|                  | Mean_Position | Mean_Energy_LP | Mean_Firing_Rate |
|------------------|--------------|---------------|------------------|
| Mean_Position    | 0.0446       | 0.0351        | 0.0007           |
| Mean_Energy_LP   | 0.0351       | 6.4364        | -0.2194          |
| Mean_Firing_Rate | 0.0007       | -0.2194       | 0.0249           |

**Interpretation:**
- The diagonal values represent the variance of each variable after filtering. The variance of Mean_Energy_LP (6.4364) is slightly lower than the original Mean_Energy, indicating some reduction in high-frequency noise.
- The covariance between Mean_Position and Mean_Energy_LP (0.0351) remains positive and similar in magnitude to the unfiltered case, suggesting a weak tendency for these variables to increase together.
- The covariance between Mean_Energy_LP and Mean_Firing_Rate (-0.2194) is still negative, indicating a weak inverse relationship.
- The covariance between Mean_Position and Mean_Firing_Rate (0.0007) remains very close to zero, suggesting almost no linear relationship between these two variables.

**Conclusion:**
- Applying the low-pass filter to the energy data does not substantially change the relationships between these variables during the 'full_reach' phase, but it does slightly reduce the variance in the energy signal, as expected.
- The overall interpretation remains: the relationships are generally weak, with the strongest (though still modest) relationship being a negative covariance between energy and firing rate.

## Per-Group Covariance Analysis (Full Reach, Low-pass Filtered Energy, Time 350-500)

The following are the average covariances between Mean_Energy_LP (low-pass filtered) and Mean_Position for each folder_level_1, computed per reach in the full_reach phase (Time 350-500):

```
folder_level_1: 210421_results, mean covariance: -0.0335
folder_level_1: 210422_results, mean covariance: -0.0286
folder_level_1: 210423_results, mean covariance: -0.0011
folder_level_1: 210425_results, mean covariance: 0.0123
folder_level_1: 210511_results, mean covariance: -0.0057
folder_level_1: 210512_results, mean covariance: -0.0086
folder_level_1: 210514_results, mean covariance: 0.0044
folder_level_1: 210515_results, mean covariance: -0.00001
folder_level_1: 210606_results, mean covariance: -0.0262
folder_level_1: 210608_results, mean covariance: -0.0317
folder_level_1: 210614_results, mean covariance: -0.0379
folder_level_1: 210619_results, mean covariance: -0.0140
folder_level_1: 210620_results, mean covariance: 0.0602
folder_level_1: 210622_results, mean covariance: -0.0051
folder_level_1: 210623_results, mean covariance: 0.0008
folder_level_1: 220515_results, mean covariance: 0.0355
folder_level_1: 220516_results, mean covariance: 0.2177
folder_level_1: 220517_results, mean covariance: 0.0747
folder_level_1: 220518_results, mean covariance: 0.1977
folder_level_1: 220519_results, mean covariance: 0.0456
folder_level_1: 220520_results, mean covariance: 0.0340
```

### Grouped by Covariance Sign

**Positive covariance (energy and position tend to increase/decrease together):**
- 210425_results, 210514_results, 210620_results, 210623_results, 220515_results, 220516_results, 220517_results, 220518_results, 220519_results, 220520_results

**Negative covariance (energy and position tend to move in opposite directions):**
- 210421_results, 210422_results, 210511_results, 210512_results, 210606_results, 210608_results, 210614_results, 210619_results, 210622_results

**Near-zero covariance (little to no linear relationship):**
- 210423_results, 210515_results, 210623_results

### Interpretation
- **Positive covariance** indicates that, for these groups, increases in energy are generally associated with increases in position (or decreases with decreases), suggesting a direct relationship between energy expenditure and movement in the full reach phase.
- **Negative covariance** suggests that, for these groups, increases in energy are associated with decreases in position (or vice versa), indicating an inverse relationship between energy and position.
- **Near-zero covariance** means there is little to no consistent linear relationship between energy and position for those groups.

**Conclusion:**
- There is heterogeneity across folder_level_1 groups: some show positive, some negative, and some negligible covariance between energy and position during the full reach (Time 350-500). This suggests that the relationship between energy expenditure and position is not uniform across all experimental groups or sessions, and may depend on specific conditions or individual differences.