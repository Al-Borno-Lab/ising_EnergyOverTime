# Verification: does the J-coupling peak fall in the acceleration phase or the deceleration phase of the reach?

**Question tested.** The user's claim (after reading Nashef et al. 2023, PMC10370034) was that
Purkinje cells enter a collective state *during arm acceleration*. This analysis pulls every
already-computed session x stimulus timing table on the cluster that carries both a J-peak
(or collective-state) timestamp and the reach's peak-velocity timestamp, and asks directly:
does the J-peak fall *before* peak velocity (acceleration phase) or *after* it (deceleration
phase)? `accel_peak_idx` (peak forward acceleration) precedes `vel_peak_idx` by a median of 3
samples across all 63 session x stimulus reaches, confirming `vel_peak_idx` marks the
acceleration-to-deceleration boundary as intended.

**Sign convention.** `lag_to_vel = j_peak_idx - vel_peak_idx`. Negative = J-peak occurs before
peak velocity (acceleration phase). Positive = J-peak occurs after peak velocity (deceleration
phase).

## Result

Across every existing analysis run on the cluster that reports a significant J-peak together
with a peak-velocity timestamp, the **median lag is positive** (peaks tend to fall in the
deceleration phase, not the acceleration phase), but the **raw count of acceleration-phase vs.
deceleration-phase peaks is close to even and never reaches significance** by a sign test. A
magnitude-weighted test (Wilcoxon signed-rank on the lag values, treating leads and lags as
different in size, not just in sign) is nominally significant (p < 0.05) in 5 of 8 configurations
checked, always in the deceleration direction.

**Bottom line: this dataset does not confirm that J peaks during acceleration.** If anything, the
weight of evidence across window choices and detector settings leans slightly toward J peaking
*after* peak velocity, i.e. during deceleration, not before it.

### Table 1 — significant J-peaks vs. peak velocity, no collective-state filter (default 325-450 window)

| analysis run | n | accel-phase (leads) | decel-phase (lags) | % decel | median lag | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| jun22 | 29 | 12 | 16 | 55.2% | +2.0 | 0.572 | 0.040 |
| jun30 (re-run) | 29 | 12 | 16 | 55.2% | +2.0 | 0.572 | 0.040 |
| arbitration_j_many | 24 | 11 | 13 | 54.2% | +8.0 | 0.839 | 0.054 |

### Table 2 — collective-state (`is_collective=True`) AND significant J-peak, 3 peak-detector variants

| detector variant | n | accel-phase | decel-phase | % decel | median lag | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| default | 37 | 15 | 21 | 56.8% | +1.0 | 0.405 | 0.035 |
| variant 2 | 37 | 15 | 21 | 56.8% | +1.0 | 0.405 | 0.035 |
| scipy-prominence | 37 | 15 | 21 | 56.8% | +1.0 | 0.405 | 0.035 |

All three peak-detection settings return an identical breakdown, so the result is not an artifact
of the detector's peak-finding parameters.

### Table 3 — window-size sensitivity (significant J-peaks only, no collective filter)

| window | n | accel-phase | decel-phase | % decel | median lag | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| 325-450 (= arbitration_j_manual default) | 20 | 8 | 12 | 60.0% | +18.5 | 0.503 | 0.046 |
| 350-425 | 12 | 8 | 4 | 33.3% | -8.0 | 0.388 | 0.582 |
| 350-475 | 27 | 10 | 17 | 63.0% | +15.0 | 0.248 | 0.027 |
| 350-475 (all-stimulus variant) | 12 | 4 | 8 | 66.7% | +2.5 | 0.388 | 0.331 |

The narrowest window (350-425) is the one exception: it flips the median lag negative
(acceleration-leaning), but on n=12 and without reaching significance either way. Every wider
window (325-450, 350-475, and the original stimulus-0-only run below) leans deceleration.

### Cross-check — original `peak_near_max_velocity_summary.csv` (21 sessions, stimulus 0 only)

| n | accel-phase | decel-phase | median lag | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| 21 | 10 | 11 | +4.0 | 1.000 | 0.455 |

Essentially an even split here (10 vs. 11), consistent with "no significant binary lead/lag
preference" but again a small positive median.

### Per-session control for stimulus non-independence

Each session contributes up to 3 stimulus repetitions to Tables 1-2; those repetitions are not
independent trials. Collapsing Table 2's collective+J-peak rows to one median lag per session
(n = 14 sessions) gives: 5 acceleration-phase, 8 deceleration-phase, 1 tied; median lag = +19.75;
sign-test p = 0.581; Wilcoxon p = 0.039. The direction and significance pattern is unchanged from
the pooled analysis.

## Interpretation

1. **No configuration shows a significant excess of acceleration-phase J-peaks.** The sign test
   (which only asks "more leads or more lags") never rejects the 50/50 null in either direction.
2. **Where a significant effect appears, it points toward deceleration, not acceleration.** The
   Wilcoxon test, which weighs *how far* the J-peak sits from peak velocity, is significant in
   5 of 8 checked configurations and in every one of those 5 the direction is positive (deceleration
   phase).
3. **This is sensitive to window choice at the margins.** The single narrowest search window
   (350-425 samples) is the only configuration to flip the median lag negative, on the smallest n
   (12) and without significance. This is exactly the kind of window-choice sensitivity the
   project's own notes have flagged as a caution before treating a J-timing result as settled.
4. Session 220515 was previously flagged in project notes as an outlier; it appears in Table 2
   with lag = +27 and +46 across two stimuli (deceleration-leaning) and -21 in a third — it is not
   driving the deceleration lean on its own.

**Net assessment of the Nashef-paper comparison:** the current dataset does not reproduce a clean
"collective state coincides with acceleration" pattern. The pattern that *does* survive across
detector settings and most window choices is closer to "J-peak follows peak velocity into the
deceleration phase," which is the opposite emphasis from the acceleration-onset framing suggested
by PMC10370034's synchrony findings. Before treating either direction as confirmed, this should be
re-run with the accel/decel boundary defined directly from the kinematic acceleration trace
(zero-crossing) session-by-session, rather than relying only on the `vel_peak_idx` proxy, and with
a pre-registered window rather than the several window choices swept here post hoc.

## Data provenance

All input tables were read directly from
`/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/` on `math-alderaan.ucdenver.pvt`
(paths: `03_j_coupling_vs_kinematics_timing/{results,notes}/...`,
`04_collective_activity_pk/notes/pk_analysis_jul10/...`). No values were recomputed from raw
spike data; this is a re-analysis of existing per-session summary CSVs already produced by the
project's own `arbitration_j_*.py` / `pk_j_analysis.py` scripts, cross-checked across every
matching summary table found in the reorganized tree.
