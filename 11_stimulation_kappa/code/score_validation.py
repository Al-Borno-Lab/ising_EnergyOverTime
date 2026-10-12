"""Score validation_rows.json against VALIDATION_CRITERIA_2026-10-12.md. usage: python score_validation.py <rows.json> <out_csv>"""
import sys, json, numpy as np, csv
rows = json.load(open(sys.argv[1])); out = []
ok_all = True
for sc in "ABCDE":
    r = [x for x in rows if x["scen"] == sc]
    d = np.array([x["dkappa"] for x in r]); kc = np.array([x["kappa_control"] for x in r])
    row = dict(scenario=sc, n=len(r), true_dkappa=0.5 if sc in "AE" else 0.0, mean_dkappa=d.mean(), sd_dkappa=d.std(ddof=1),
               mean_kappa_control=kc.mean(), n_abs_dkappa_lt_0p3=int((np.abs(d) < 0.3).sum()))
    c1 = 0.40 <= d.mean() <= 0.60 if sc in "AE" else None
    c2 = (abs(d.mean()) <= 0.10 and row["n_abs_dkappa_lt_0p3"] >= 7) if sc in "BCD" else None
    c3 = abs(kc.mean()) <= 0.10
    row.update(crit1_recover=c1, crit2_no_false_positive=c2, crit3_control=bool(c3)); out.append(row)
conv = np.mean([x["converged"] for x in rows]); jc = np.median([x["Jcorr"] for x in rows])
for o in out: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()})
c4 = conv >= 0.95; c5 = jc >= 0.90
print("crit4 convergence %.3f -> %s ; crit5 median Jcorr %.4f -> %s" % (conv, c4, jc, c5))
passed = all((o["crit1_recover"] in (True, None)) and (o["crit2_no_false_positive"] in (True, None)) and o["crit3_control"] for o in out) and c4 and c5
print("OVERALL:", "PASS" if passed else "FAIL")
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
