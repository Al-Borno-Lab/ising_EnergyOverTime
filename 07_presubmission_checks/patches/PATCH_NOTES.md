# Patches for three code issues (NOT applied to project code)

Each diff was applied to a copy of the code on math-alderaan and tested. Apply from the repo root with `patch -p1 < patches/<file>.diff`.

| Patch | File | Problem | Fix | Test |
|---|---|---|---|---|
| 01_aggregate_session_label.diff | shared/code/aggregate_transition_analysis.py | `session_label()` returns `sdir.parent.name`, which is `full_reach` for every session in the organized layout. All 21 sessions collapse into one "animal" and overwrite each other's per-session CSV | Walk up to the nearest `<experiment>_repN` folder; fall back to the parent name | Run directly on `01_ising_model_fitting/results` for stim 0: finds 21 sessions, and the meta-analysis matches the copy-tree workaround exactly (pooled d identical for position, velocity and acceleration) |
| 02_pk_j_analysis_fstring.diff | 04_collective_activity_pk/code/pk_j_analysis.py | Three f-strings contain `\u2713` inside `{...}` expressions (lines 448, 665 and 667). This is a SyntaxError before Python 3.12, and the project container runs 3.10 | Module constant `CHECK = "\u2713"`, used as `'MET ' + CHECK` | Imports and runs in the container (Python 3.10) |
| 03a_arbitration_sys_path.diff | 03_j_coupling_vs_kinematics_timing/code/arbitration_j_many.py | `from src...` fails unless `shared/` is on PYTHONPATH | Insert `shared/` and `04_collective_activity_pk/code/` into `sys.path`, resolved relative to the script | `--help` runs with PYTHONPATH unset |
| 03b_test_collective_sys_path.diff | 03_j_coupling_vs_kinematics_timing/code/test_collective_j_timing.py | Same problem, plus `pk_j_analysis` is not found | Same fix | Full run in the container with PYTHONPATH unset: 63 rows, identical to the regenerated output |
