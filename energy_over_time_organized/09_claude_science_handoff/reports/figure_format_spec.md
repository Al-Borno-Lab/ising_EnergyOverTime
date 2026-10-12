# Target Figure-Format Spec (captured, not yet applied)

Purpose: document the house convention already in use under
`shared/results/comparisons/` on the cluster so that when figure regeneration
is authorized (after claims are verified), new/updated figures match it
exactly. **No figures are being regenerated in this step** — this is a spec
capture only, per instruction to hold off on figure edits until the audit,
falsification test, and decision-tree review are complete.

Source examined: `model_quality_compare_210421_stim0_vs_210422_stim0.{png,pdf,txt}`
and `pk_analysis_compare_210421_stim0_vs_210422_stim0.{png,pdf,txt}` under
`energy_over_time_organized/shared/results/comparisons/`.

## 1. File triplet convention

Every comparison figure ships as **three files sharing one basename**:
- `<analysis>_compare_<sessionA>_stim<N>_vs_<sessionB>_stim<M>.png` — raster for quick viewing
- `<analysis>_compare_<sessionA>_stim<N>_vs_<sessionB>_stim<M>.pdf` — vector, for print/paper insertion
- `<analysis>_compare_<sessionA>_stim<N>_vs_<sessionB>_stim<M>.txt` — plain-text sidecar with the
  quantitative summary that becomes the figure caption

`<analysis>` observed so far: `model_quality`, `pk_analysis`. The two session/stim
identifiers being compared are always encoded directly in the filename, in
comparison order (left panel column = first-named session/stim, right column =
second).

## 2. Panel layout and labeling

- **Two-column layout**: column 1 = first session/stim, column 2 = second
  session/stim — a direct visual comparison, not an overlay.
- **Panel labels**: bold, top-left inside each axes box, letter = row (metric/
  feature type), number = column (1 = first session, 2 = second session).
  E.g. in `model_quality_compare_*`: `A1`/`A2` = pairwise-correlation
  reconstruction scatter, `B1`/`B2` = coupling-strength histogram P(J),
  `C1`/`C2` = triplet-correlation scatter, `D1`/`D2` = P(K) fit comparison
  (log-scale y). In `pk_analysis_compare_*`: `A`–`F` = velocity, acceleration,
  spike raster, firing rate, energy, and J-coupling/H-field traces, each row
  shared across both columns (`A1`/`A2` … `F1`/`F2`).
- **No in-plot legend.** Series identity is carried by axis labels and color,
  not a legend box — e.g. in the J-coupling/H-field row, the left y-axis label
  ("J Coupling") is colored blue and the right y-axis label ("H Field") is
  colored red, matching their trace colors, instead of a legend.
- **Shared reference marks across both columns**: a shaded band (light yellow)
  marks the analysis window, and a pair of green dashed vertical lines mark
  the detected event time — drawn identically in both columns so the reader
  compares like-for-like.
- **Log-scale P(K) plots** use decade tick labels (`1e-6`, `1e-4`, …), never
  raw exponents.
- One inset (a "zoom" panel) is used at least once (in `A1` of the
  model-quality figure) to magnify a small-magnitude cluster near the origin —
  drawn as a nested axes box in the corner of the parent panel, not a separate
  figure.

## 3. Sidecar `.txt` structure (becomes the caption)

Verified field order and formatting from both examples:

```
<Analysis label> comparison: Session <A>, stim <n>  vs  Session <B>, stim <m>
====... (rule matching the title length) ...====

Session <A>, stim <n>:
  <field label, left-padded/colon-aligned>:    <value with unit>
  ...
  (blank line)
Session <B>, stim <m>:
  <same field order>
```

For `model_quality_compare_*`, the field order is:
`N neurons` → `N pairwise couplings (J)` → `J coupling: mean/std/min/max` →
`Pairwise correlation r (measured vs reconstructed)` →
`Triplet correlation r (measured vs Ising)` →
`Triplet correlation r (measured vs independent)` →
`P(K) fit: r_ising / r_independent / sum|P_ising - P_independent|` →
`Ising at least as close as independent (tol=0.02)` →
`Collective behavior (Ising closer AND dist > 0.1)` → `K range`.

For `pk_analysis_compare_*`, the field order is:
`Analysis window` (ms and bin range) → `Peak velocity` (value + unit + time) →
`Peak acceleration` (value + unit + time) → `Mean firing rate` →
`Firing-rate peak` / `Firing-rate trough` (or "none detected in window") →
`J-coupling peak` (or "none detected", with ratio) → `H-field peak` →
`Timing classification (J peak vs kinematic peaks)` → `P(K) fit: ...` →
`Collective behavior (...)`.

Precision observed: correlation/r values to 4 decimals (`0.9996`), coupling
statistics to 4 decimals with explicit sign (`mean=+0.0100`), physical
quantities (velocity, acceleration, firing rate) to 3 decimals with units
inline, times to whole ms with the corresponding bin index in parentheses.
Boolean/classification fields are spelled out (`True`/`False`,
`no_peak`), not abbreviated.

## 4. What this means for regenerated figures (future step, not yet done)

Any figure produced for the paper should:
1. Ship as a PNG+PDF+TXT triplet under a name that encodes the analysis and
   the sessions/stims being compared.
2. Use bold top-left `<row-letter><col-number>` panel labels, never a legend
   box — carry series identity through axis-label color instead.
2. Reuse the shaded-window + dashed-event-time convention wherever a time
   series is shown, so every figure in the paper reads the same way.
3. Write the sidecar `.txt` with the same field order/precision shown above so
   the caption text can be generated mechanically from it rather than
   hand-typed per figure.

This spec is descriptive of the existing convention only; it is not applied to
any of this audit's own diagnostic figures (`falsification_test.png`,
`session_diagnostics.png`, `decision_tree_generalization.png`), which remain
in the plainer verification style used throughout this review and are not
paper-facing deliverables.
