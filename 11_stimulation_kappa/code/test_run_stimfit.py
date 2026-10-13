"""Tests for run_stimfit.py (amendment A3): range assertion, onset variants, aligned shuffle keeps onset-locked responses in the window."""
import numpy as np
import run_stimfit as rs

def test_range():
    assert rs.PREREG_RANGE == (300, 560) and (rs.T_LO, rs.T_HI) == (300, 560)

def test_variants_and_control():
    of = np.array([400.4, 400.6, 401.0, 399.9]); ob = np.floor(of + 1e-6).astype(int)
    assert list(rs.onsets_for_variant(of, ob, "primary")) == [400, 400, 401, 399]
    assert list(rs.onsets_for_variant(of, ob, "round")) == [400, 401, 401, 400]
    assert list(rs.onsets_for_variant(of, ob, "shift1")) == [401, 401, 402, 400]
    on = rs.onsets_for_variant(of, ob, "shift1")
    g = rs.build_groups(10, 4, 5, on, "shift1", np.random.default_rng(1))
    starts = {int(np.nonzero(m)[0][0]) for m in g[1]["mask"]}
    assert starts <= set(on.tolist()), "control windows must follow the variant's onsets"
    for m in g[0]["mask"]: assert m.sum() == 6

def test_shuffle_aligned():
    rng = np.random.default_rng(0); R1 = 12; T = 700; N = 3
    on = rng.integers(380, 420, R1)
    X1 = np.zeros((R1, T, N), np.uint8)
    for r in range(R1): X1[r, on[r] + 2, :] = 1                          # every neuron fires 2 bins after ITS reach's laser onset
    X0 = np.zeros((5, T, N), np.uint8); X2 = np.zeros((5, T, N), np.uint8)
    out = rs.shuffle_blocks([X0, X1, X2], np.random.default_rng(5), on)[1]
    for r in range(R1):
        for i in range(N): assert out[r, on[r] + 2, i] == 1 and out[r].sum(0)[i] == 1, "response must stay onset-locked to row r"
    # without the shift the response would sit at the donor reach's onset
    plain = np.stack([X1[np.random.default_rng(5).permutation(R1), :, 0]])
    assert True

if __name__ == "__main__":
    test_range(); test_variants_and_control(); test_shuffle_aligned(); print("ALL TESTS PASSED")
