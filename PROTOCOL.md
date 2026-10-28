# Experiment protocol (for people and agents)

Every analysis in this project is an **experiment folder** `NN_short_name/` at the repo root.
The status of all of them is the table in [README.md](README.md). Follow these steps in order.

## Folder contract

```
NN_short_name/
├── experiment.json   # status metadata, read by tools/update_status.py
├── README.md         # experiment write-up: question, hypothesis, data, method, how to reproduce
├── RESULTS.md        # results write-up: claims, each one citing an asset and a run
├── code/             # EVERY script needed to recreate the experiment; code/run.sh is the entry point
├── assets/
│   ├── figures/      # every figure cited in RESULTS.md (png/pdf/svg), committed
│   └── tables/       # every table cited in RESULTS.md (csv/json/md), committed
├── runs/RUN_LOG.md   # append-only log of submissions and outcomes
└── results/          # raw outputs, cluster only (gitignored)
```

Rules:
- An experiment folder must be **self-contained**: `code/run.sh` reproduces it from the raw data with no
  scripts from elsewhere, except `shared/` and the pipeline in `old_code/energy_over_time_script/`, referenced by
  path and git commit.
- Assets stay small: keep each file under 5 MB and the folder under about 50 MB. Large outputs stay in
  `results/` on the cluster. `.gitignore` already lets files in `assets/` through.
- Nobody edits another experiment's folder. To change a finished experiment, create a new one that
  `depends_on` it, or mark the old one `superseded`.

## Pipeline

1. **Claim.** Run `bash tools/new_experiment.sh <short_name> "<Title>" <owner>`. Fill in `README.md`
   (question, hypothesis, decision rule, data, method) and `experiment.json`.
   Commit on your branch (`exp/NN_short_name`) **before running anything**.
   Anything pre-registered goes in a dated document; later changes are dated amendments.
2. **Build.** Write the scripts in `code/` and wire every step into `code/run.sh`. Smoke-test on one
   session as a small sbatch job. Commit the code.
3. **Run.** Submit with `sbatch`: heavy work never runs on the login node, and nothing runs on the cluster
   without the user's go-ahead. Log every submission with `tools/log_run.sh <dir> submitted ...`. Set
   `status` to `running`.
4. **Record.** Log the outcome with `tools/log_run.sh <dir> finished|failed ...`. Failed runs stay in
   the log; move their outputs to `results_failed_<reason>_<date>/`.
5. **Analyse.** Set `status` to `analysis`. Copy the figures and tables you will cite into `assets/`,
   then write `RESULTS.md`. Every claim names its asset and run. Statistics: the unit is the session
   (n = 21), with Wilcoxon signed-rank tests and BH-FDR; also report cohort (2021 vs 2022) and mouse.
   Timing claims need control windows at rest. Synthetic data is only for method validation, labelled
   as such.
6. **Close.** Set `status` to `complete` and fill in `headline` and `updated`. Run
   `python3 tools/update_status.py`. Add the claims you are sure of to
   [RESULTS_LOG_FOR_PAPER.md](RESULTS_LOG_FOR_PAPER.md), the single paper-facing log: add to it, never
   replace it. Commit only this experiment's paths (plus README/log), then open a PR into `main`.

## Review before merge (required for every experiment PR)
The orchestrator does three checks, and records them in `NN_*/REVIEW_<date>.md`, before recommending a merge:
1. **Independent code review** of the analysis code (by a separate reviewing agent). It checks every constant against the pre-registration (windows, bin ranges, bases, seeds, folds, exclusions), window indexing, null construction, statistics logic (direction of each test and its decision rule), and silent drops. **Run this while long cluster jobs are still running**, so bugs are caught before results are read.
2. **Spot-check:** recompute the headline numbers independently from the committed tables or raw outputs.
3. **Reproduce:** re-run `code/run.sh` (local experiments) or a single condition (cluster experiments) from a clean worktree and compare against the committed assets.

Agents must also:
- **assert** in code that fitted ranges, windows and fold assignments equal the pre-registered values, and store them in every output row;
- make synthetic validation **as hard as the real data**: real reach counts, real timing jitter, block- or condition-specific rates, the same null procedure, and noise matched to the real data;
- **write the statistics script and have it reviewed before reading results.**

Bugs found after results are seen are fixed with a dated amendment that discloses what was seen. Superseded outputs are kept, renamed `*_runN_*`.

## Status values
`planned` → `running` → `analysis` → `complete`. `superseded` means a newer experiment replaces it
(say which in `headline`). `archived` means a legacy folder kept for reference.

## Git
- One branch per experiment: `exp/NN_short_name`. Commit only that experiment's paths with `git add <paths>`,
  never `git add -A`. Never commit someone else's uncommitted edits.
- Never commit raw data (`.mat`, `.npz`, `.pkl`) or anything over 5 MB.
- Merge into `main` through a PR whose description links `RESULTS.md`.
- **One git worktree per experiment and per agent.** Never switch branches in a checkout another agent
  is using. Create a separate working copy next to the repo:
  `git worktree add ../ising_wt_eNN exp/NN_short_name` (or `-b exp/NN_short_name ../ising_wt_eNN origin/main` for a new
  branch). Work, commit and push only from that worktree. After the PR is merged, clean up with
  `git worktree remove ../ising_wt_eNN`. On the cluster, do the same under `~/ising_wt/`, and never use
  `~/ising_EnergyOverTime` itself for experiment work.
- Before starting, `git fetch` and branch from the latest `origin/main`. Before opening a PR, rebase onto `origin/main`.

## Cluster cheatsheet
- **Job pattern: one SLURM job per task, fanned out.** Submit a SLURM **array** with one element per task (condition, subset, ...), 16–32 CPUs per element, and parallelize the independent fits *inside* the task (real fit, shuffles, CV folds) with a process pool. Don't run one big job with a pool of single-core tasks: it hits the wall-time limit and loses in-progress work. Each task writes its own output file and skips work whose file already exists, so a resubmission resumes. Throttle concurrency (`--array=...%20`, adjusted to cluster load) and give the reason in the run log. Put the heaviest tasks first. Results must not depend on worker order (per-fit deterministic seeds).
- Host `math-alderaan.ucdenver.pvt` (VPN). SLURM account `ceas-alborno`, partition `math-alderaan`.
  Nodes have 64 CPUs, so parallelise inside the job.
- Python: `singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python ...`
  (Python 3.10, no backslashes inside f-string braces). Never `pip install coniii`.
- The container can't see `/data001/...` from sbatch jobs, so copy inputs under `$HOME` first.
- The MCH pipeline master script writes `job_template.sh` into the cwd: never run two masters in the same directory.
