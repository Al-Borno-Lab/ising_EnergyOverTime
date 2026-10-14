"""Checks analyze_pool == analyze (serial) to rtol 1e-6 on small conditions and one 16-neuron subset task. Usage: python test_pool_equivalence.py SPIKES_NPZ NCPU"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
Z = dict(np.load(sys.argv[1])); ncpu = int(sys.argv[2]); bad_total = 0
for key, sub, ns in (("210423_0", None, 20), ("210622_0", None, 20), ("210608_2", (0, 210608 * 0 + int("210608_2".replace("_", ""))), 3)):
    t = time.time(); a = L.analyze(Z, key, sub, nshuf=ns); ta = time.time() - t
    t = time.time(); b = L.analyze_pool(Z, key, sub, ncpu, nshuf=ns); tb = time.time() - t
    bad = [k for k in a if k != "secs" and not isinstance(a[k], str) and not np.allclose(float(a[k]), float(b[k]), rtol=1e-6, atol=1e-9, equal_nan=True)]
    missing = set(a) ^ set(b); bad_total += len(bad) + len(missing - {"secs"})
    print(key, sub, "nshuf", ns, "| mismatching:", bad, "| key diff:", missing, "| serial %.0fs pool %.0fs" % (ta, tb), flush=True)
print("EQUIVALENT" if bad_total == 0 else "MISMATCH"); sys.exit(1 if bad_total else 0)
