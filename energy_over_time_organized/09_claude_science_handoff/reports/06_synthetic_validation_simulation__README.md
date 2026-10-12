# 06_synthetic_validation_simulation/

## What this answers

Do the empirical findings (J-coupling leading kinematics, energy-spikes coinciding with synchrony epochs) replicate in a synthetic spiking network with known ground-truth structure? This folder builds a controlled sanity check to rule out the possibility that the pipeline's Ising/energy signatures are artifacts of the fitting procedure rather than genuine collective effects.

## Scripts

- `simulate_neural_regimes.py` — Simulates an EI spiking network alternating between an independent Poisson regime (recurrent connections off) and a recurrent/collective regime (full EI dynamics) in 200-step blocks over 5,000 total steps (500 ms at dt=0.1 ms); subsamples excitatory neurons, fits an inverse Ising model, computes the H(t) = -J_term(t) - h_term(t) decomposition, and saves raster/rate/energy visualizations. Run via `python simulate_neural_regimes.py --output_dir ... --seed ... --total_steps 5000 --seg_len 200 --bin_ms ... --ne 100 --ni 25 --n_ising 20 --sample_size ...` (see script for full CLI arg list).
- `run_neural_simulation_sbatch.sh` — Submits `simulate_neural_regimes.py` as a Slurm batch job; logs land in `logs/neural_sim/`.

## Key findings

(Drawn from `notes__neural_sim__neural_simulation__simulation_writeup.md`, an LLM-generated summary of the original note.)

- The simulation uses N_e=100 "Purkinje" units and N_i=25 interneurons, alternating 200 ms random (independent Poisson, ~60 Hz) and collective (shared two-state Markov burst/quiet, p_on=0.005, p_off=0.02) blocks, with mean rates deliberately matched (~56 Hz collective vs. 60 Hz random) so any Ising differences reflect correlation structure, not firing-rate changes.
- After 5 ms binning (needed because raw 0.1 ms bins are too sparse/degenerate for fitting), co-firing probability is ~0.41 in burst bins vs. ~0.009 in quiet bins — roughly a 45-fold difference.
- Mean per-neuron activation σ is ≈+0.28 in burst bins vs. ≈−0.81 in quiet bins, giving Var(E[σ|B])≈0.19.
- Ising fitting used 20 subsampled neurons (210 h/J parameters), MCH algorithm with Metropolis MCMC (50,000 samples/iteration, up to 100 iterations, convergence target ‖Δθ‖₂<0.05) via the `coniii` package.
- Overall, the writeup frames this as evidence that the J-jump phenomenon seen in real cerebellar reach data can plausibly arise from synchrony epochs with near-constant mean rate, supporting the pipeline's validity.

## Caveats

- Only one note summary was available for this folder; no second note exists to cross-check or contradict these results, and the summary itself was truncated (cut off mid-sentence on the Var(E[σ|B]) result), so some quantitative detail may be missing from this README.
- This is explicitly synthetic/ground-truth validation data, not real neural recordings — findings here support pipeline validity but are not themselves a claim about cerebellar physiology.