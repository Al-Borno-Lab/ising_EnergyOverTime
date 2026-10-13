"""Score validate_realistic output against VALIDATION_CRITERIA amendments 6 and 6b. usage: python score_realistic.py <rows.json> <out_csv> [high|low]
low = structures 210421, 210511, 210514 (mild onset jitter); high = 220519, 210608, 220520 (jitter up to 20 bins)."""
import sys, json, csv, numpy as np
HIGH = {"220519", "210608", "220520"}
grp = sys.argv[3] if len(sys.argv) > 3 else None
rows = [x for x in json.load(open(sys.argv[1])) if grp is None or (x["date"] in HIGH) == (grp == "high")]; out = []; ok = True
bias = {}
for sc in "ABCDE":
    r = [x for x in rows if x["scen"] == sc]; dc = np.array([x["d_corr"] for x in r]); dr = np.array([x["d_raw"] for x in r]); kc = np.array([x["kappa_control"] for x in r])
    bias[sc] = (dc.mean(), dr.mean())
    row = dict(scenario=sc, n=len(r), true=0.5 if sc in "AE" else 0.0, mean_dcorr=dc.mean(), sd_dcorr=dc.std(ddof=1), mean_draw=dr.mean(), sd_draw=dr.std(ddof=1),
               mean_kcontrol=kc.mean(), null_sd_kp=np.mean([x["null_sd"][0] for x in r]), null_sd_kc=np.mean([x["null_sd"][1] for x in r]), n_fits_below_minus1=sum(x["n_below_m1"] for x in r))
    row["c1"] = (0.35 <= dc.mean() <= 0.65) if sc in "AE" else None
    row["c2"] = (abs(dc.mean()) <= 0.10) if sc in "BCD" else None
    row["c3"] = (abs(kc.mean()) <= 0.10) if sc == "D" else None
    out.append(row)
conv = np.mean([x["n_conv"] / x["n_fits"] for x in rows])
pool_c = np.mean([abs(bias[s][0]) for s in "BCD"]); pool_r = np.mean([abs(bias[s][1]) for s in "BCD"])
c4 = conv >= 0.95; c5 = pool_c <= pool_r + 0.02
for o in out: print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in o.items()})
print("crit4 convergence %.3f -> %s ; crit5 pooled |bias| corrected %.3f vs raw %.3f -> %s" % (conv, c4, pool_c, pool_r, c5))
passed = all(all(o[k] in (True, None) for k in ("c1", "c2", "c3")) for o in out) and c4 and c5
print("OVERALL:", "PASS" if passed else "FAIL")
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
