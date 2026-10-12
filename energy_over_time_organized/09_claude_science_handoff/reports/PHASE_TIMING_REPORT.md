# Phase-resolved timing of energy, coupling (J), field (h) and firing rate relative to reach acceleration and deceleration

Data: 21 sessions x 3 stimuli (63 conditions; 8-37 reaches, 3-32 neurons each), from `01_ising_model_fitting/results/*/full_reach/stim_*/per_reach_state.csv` in `energy_over_time_organized` on math-alderaan. Unit of replication for every group test: the session (n = 21; the 3 stimuli are averaged within a session first). Time is in samples (700 per reach). Conversions to ms assume the 10 ms bin used in earlier project scripts, which was not re-verified here.

Sign convention (checked on all 874,300 rows): `energy = -(j + h)` exactly, so the `j` column is the coupling sum ΣJ_ij s_i s_j and the `h` column is the field sum Σh_i s_i. A rise in `j` therefore lowers the energy.

## 1. Kinematic phases defined from the acceleration trace

![Phase landmarks]({{artifact:art_215abd50-5b81-471e-8285-6293c18c5d21}})

Velocity and acceleration were computed from x position, as in `shared/src/arbitration.py`, with Savitzky-Golay smoothing (11 samples, order 3). The landmarks are: onset (velocity reaches 10% of its peak), peak acceleration, the acceleration zero-crossing (= peak velocity, the accel/decel boundary), peak deceleration, and deceleration end (acceleration returns to 0).

The outward reach is brief. The acceleration phase lasts a median of 9 samples and the deceleration phase a median of 12. This changes how the earlier verification should be read. It concluded that J peaks "lean toward the deceleration phase" because they fell after peak velocity. At +20 to +60 samples, most of those peaks actually fall after the deceleration phase has ended, in the post-movement period.

## 2. Energy decomposition: the coupling term drives the energy drop during movement

![Energy decomposition]({{artifact:art_99fe21b3-98be-43e4-ab8c-c9144dbe2f2d}})

| Phase | Coupling ΔE contribution (median) | p | Field ΔE contribution (median) | p |
|---|---|---|---|---|
| Acceleration | -0.30 | 1e-5 | -0.09 | 0.92 |
| Deceleration | -0.21 | 1e-4 | -0.13 | 0.84 |
| Post-movement (50 samples) | -0.21 | 7e-4 | -0.01 | 1.00 |

- Energy falls below baseline during and after the reach. The whole consistent part of that fall comes from the coupling term. The field term moves in both directions across sessions.
- The coupling share of energy variance rises from 0.30 at baseline to 0.36-0.38 during movement (baseline vs. movement p ≈ 0.004). It does not differ between acceleration and deceleration (p = 1.0).
- The coupling-driven drop starts about 30 samples before movement onset and is deepest at about +35 samples, after the outward stroke.

## 3. Is any signal correlated with acceleration or deceleration?

The answer depends on the timescale.

- **Within a phase, sample by sample: no.** The Spearman r between each signal and signed acceleration inside the acceleration or deceleration phase is about 0 for all four signals. All p_FDR are above 0.05; the closest is the field term in deceleration (r = 0.03, p_FDR = 0.055).
- **Reach to reach: no.** The phase-mean change in each signal does not track that reach's peak acceleration or peak deceleration magnitude (all p_FDR > 0.3).
- **Movement-locked structure: yes, for all four.** The cross-correlation with acceleration over −50 to +80 samples beats a circular-shift null (500 shifts per condition, matched lag grid). Session-median z is 1.44 for energy, 1.59 for J, 1.21 for h and 1.58 for firing rate (18-20 of 21 sessions positive; p_FDR < 0.001 for each). The sign of the peak correlation is inconsistent across sessions. Firing rate's alignment z exceeds each Ising term's (paired p = 0.014 for energy, 0.016 for J, 0.032 for h; uncorrected).

## 4. Peak timing relative to acceleration landmarks

The search window (peak acceleration −50 to +100 samples) was fixed in advance.

| Signal (maximum) | In acceleration phase / 63 (expected ≈ 4.1 from duration) | Post-movement / 63 | Median lag to peak acceleration (sessions after / 21) | p (session Wilcoxon) |
|---|---|---|---|---|
| Coupling term ΣJss | 10 (p = 0.008) | 33 | +30 (17) | 0.002 |
| Field term Σhs | 8 | 28 | +5 (12) | 0.51 (FDR) |
| Energy | 3 | 19 | −22 (7) | 0.21 (FDR) |
| Firing rate | 8 | 22 | −2 (10) | 0.68 (FDR) |

The coupling-term maximum shows two things. It lands inside the short acceleration phase more often than duration alone predicts (about 2.4x). But its typical position is after the movement ends: median +30 samples from peak acceleration, and +26 from peak velocity. The field term and firing rate peak around the movement itself, with no directional bias. Within the full FDR family for this table, the coupling results fall just short of 0.05 (p_FDR = 0.051-0.064). They reach p_FDR ≈ 0.02-0.04 in the sensitivity family (Section 7).

## 5. Arbitration: Ising terms vs. firing rate

| Metric (session level) | Energy vs. FR | Coupling vs. FR | Field vs. FR |
|---|---|---|---|
| Movement modulation depth (median \|z\|: FR 9.4) | 7.5, FR wins 16/21, p_FDR = 0.022 | 7.8, FR wins 15/21, p_FDR = 0.044 | 7.1, FR wins 18/21, p_FDR = 0.003 |
| Alignment with acceleration (xcorr z) | FR stronger, p_FDR = 0.063 | FR stronger, p_FDR = 0.063 | FR stronger, p_FDR = 0.085 |
| Distance of dominant extremum to peak acceleration | n.s. | J further, p = 0.029 (p_FDR = 0.085) | n.s. |

No Ising term tracks acceleration or deceleration better than firing rate. What distinguishes the coupling term is when it responds (later), not how well it tracks kinematics.

![Summary]({{artifact:art_f9fc4ce1-3967-445a-b46f-fd00f00c2018}})

Panel b uses each signal's dominant (largest |z|) extremum. Session-median lags from peak velocity: coupling +23 (p = 0.01), energy +14 (p < 0.01), field +11 (p = 0.03), firing rate +14 (p = 0.06).

## 6. Per-session timing figures

There are 63 six-panel figures in the arbitration.py style: velocity, acceleration, energy, coupling, field and firing rate, each with phase shading, landmark lines, the dominant-extremum marker and an annotation of lag, phase and r. Each is a PNG+PDF+TXT triplet (house convention) and all are in [per_session_phase_timing_figures.zip](/Users/gunnarenserro/.claude-science/orgs/f552423d-f1dd-4c5d-879b-edcfa3506e26/artifacts/proj_aa374d75609c/2dc92219-1031-4216-b292-1fee5cdc7475/vbac8deb0_per_session_phase_timing_figures.zip). Example:

![Example 220517 stim 1]({{artifact:art_41d9c994-4866-4ece-8b30-c7bf6ff1a565}})

**Interpretive caveat raised by these figures.** In several high-N sessions (for example 220516-220518), population firing rate pauses after the reach while ΣJss rises. With ±1 spin coding, neurons falling silent together (s_i = s_j = −1) add +J_ij to the coupling sum, just as neurons firing together do. Across sessions the effect is not simply a pause artefact: J rises in 16/21 sessions while firing rate falls in only 11/21, and ΔJ vs. ΔFR gives ρ = −0.31, p = 0.18. It cannot be excluded within those sessions, though. ΔJ also scales with population size (ρ = 0.58, p = 0.006), as expected for a sum over N(N−1)/2 pairs. **Recommended follow-up:** split ΣJss into co-active (+1,+1), co-silent (−1,−1) and discordant pair contributions using the spike columns and fitted J_ij, and normalise by pair count.

## 7. Sensitivity

![Robustness]({{artifact:art_33b2c7a6-1dc7-423d-9fad-4ecdf713bc12}})

- **Holds in every variant:** the coupling-driven energy drop, with no field contribution; the coupling peak following peak acceleration (p = 0.001-0.008 across windows −25/−75 before and +75/+125 after, and with 210620/220515 excluded); firing rate having the deepest movement modulation.
- **Weaker:** acceleration-phase enrichment of coupling peaks (p ≤ 0.03 in 5 of 6 variants; p = 0.06 with the window extended to +125).
- **Did not replicate:** the earlier criticality-proxy pilot result (Var(J) peak +21 samples after peak velocity, p_FDR = 0.0075). Re-implemented here with the refined landmarks, the Var(J) peak lag is +9, +11 and +2 samples at W = 15, 25 and 40 (p = 0.03, 0.05, 0.33; none survive FDR). The robust fluctuation result is instead a quenching of Var(energy) and Var(h) after the reach (troughs at +31 to +38 samples, p_FDR ≤ 0.02 at every window size).

## Verified vs. not verified

- **Verified (robust to windows, exclusions, FDR):** (a) the energy decrease accompanying movement comes from the coupling term; (b) the coupling-term peak follows peak acceleration and mostly falls after the reach ends; (c) firing rate is more strongly movement-modulated than energy, J or h; (d) no signal covaries with acceleration sample-by-sample within a phase, or with acceleration magnitude reach-to-reach.
- **Suggestive only:** coupling peaks occur in the acceleration phase more often than chance (10/63; FDR-borderline and window-dependent).
- **Not supported:** "J peaks during acceleration" as the typical pattern; "J peaks during deceleration" (the deceleration phase is too short to contain most peaks, which are post-movement); the pilot's Var(J)-peak timing.
- **Not tested here:** whether the coupling rise reflects co-activation or co-silence (see Section 6), and whether the pairwise model is required (the surrogate-refit check is still open).

## Files

- Tables: `phase_landmarks.csv`, `energy_decomposition_by_phase.csv`, `phase_correlations.csv`, `phase_correlations_summary.csv`, `peak_timing_by_phase.csv`, `peak_timing_summary.csv`, `arbitration_vs_firing_rate.csv`, `arbitration_per_condition.csv`, `dominant_extremum_lags_by_session.csv`, `cosilence_check.csv`, `sensitivity_summary.csv`.
- Figures: `phase_landmarks_qc.png`, `energy_decomposition_by_phase.png/.pdf`, `phase_timing_summary.png/.pdf`, `sensitivity_robustness.png`, `per_session_phase_timing_figures.zip`.
- Code: `phase_core.py` (array building, phase definitions) and `extract_compact.py` (cluster-side extraction). The full analysis code is recorded in artifact lineage.


## 8. Correction (added after the J-timing figure): the "+30 sample" J-peak lag was caused by the search window

![J timing]({{artifact:art_88c084a6-efe1-4709-aa5b-fa1b91c8fe78}})

The search window used in Sections 4 and 7 (peak acceleration −50 to +100) is asymmetric, so its centre sits at +25. A peak landing at random inside it therefore has a median lag near +25, and a Wilcoxon test against 0 is biased toward "after". Rerunning the identical search on pre-reach control windows (shifted −150, −200 and −250) gives a control median of +24 for J. Paired against that control per session, the real J-peak lag is +30 versus +24: a difference of +9, later in 13/21 sessions, **p = 0.24**. Energy peaks are significantly *earlier* than control (−35, p = 0.001); the h term and firing rate do not differ (p ≈ 0.22). Section 4's "J peaks +30 samples after peak acceleration (p = 0.002)" and the matching robustness row in Section 7 are therefore **withdrawn**. The "33/63 post-movement" count is also about what the window length alone predicts (≈ 53%).

What survives, now tested against a proper reference:
- **Mean J course (panel b, relative to the pre-reach baseline):** J rises from about −34 samples, crosses half-maximum at −20, peaks at z = 4.7 at +43 and falls back through half-maximum at +105. This is a sustained elevation spanning the reach, not a timed peak. It is the same effect as the coupling-driven energy drop in Section 2, which is unaffected by this correction.
- **Single-reach J peaks (panel c, n = 1249):** 12.4% fall within ±5 samples of peak velocity, against 6.4% in the control windows (binomial p < 1e-14; KS test on the full distribution p = 0.01). This excess is concentrated in some sessions: 220516 (49% of reaches), 210425 (31%), 210608 (26%), 210512 (25%). Outside that mode the peak times are close to flat, and the fraction falling after the movement matches control (49% vs 55%).

Revised J timing summary: J is raised throughout the reach and stays raised for about 100 samples. Single-reach J maxima occur at peak velocity about twice as often as chance, driven by a subset of sessions. The claim that J peaks *after* the movement is not supported once the window's asymmetry is accounted for.
