import os, shutil, io, textwrap, zipfile, csv
from PIL import Image
import img2pdf, pypdf, fitz
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from manifest import *

import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/Users/gunnarenserro/Documents/AlBornoMazenlab/code/evidence_package"
os.makedirs(OUT, exist_ok=True)
ALL = []  # (folder, fig dict, png path, pdfkind)

def raster_pdf(png, pdf):
    im = Image.open(png)
    dpi = im.info.get("dpi")
    dpi = (round(dpi[0]), round(dpi[1])) if dpi else (150, 150)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA"); bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); im = bg
    else:
        im = im.convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG", dpi=dpi)
    layout = img2pdf.get_layout_fun((img2pdf.in_to_pt(im.width / dpi[0]), img2pdf.in_to_pt(im.height / dpi[1])))
    with open(pdf, "wb") as f: f.write(img2pdf.convert(buf.getvalue(), layout_fun=layout))
    return dpi[0]

for F in FOLDERS + [X]:
    d = os.path.join(OUT, F["name"]); os.makedirs(d, exist_ok=True)
    for t, newname in F["tables"]:
        if os.path.exists(t): shutil.copy2(t, os.path.join(d, newname or os.path.basename(t)))
        else: F.setdefault("missing_tables", []).append(t)
    for g in F["figs"]:
        png = os.path.join(d, g["n"] + ".png"); pdf = os.path.join(d, g["n"] + ".pdf")
        shutil.copy2(g["png"], png)
        if g["pdf"]:
            shutil.copy2(g["pdf"], pdf); g["kind"] = "vector (copied from source)"
        else:
            dpi = raster_pdf(g["png"], pdf); g["kind"] = f"rasterized from PNG ({Image.open(g['png']).size[0]}x{Image.open(g['png']).size[1]} px at {dpi} dpi)"
        ALL.append((F, g, png))

# September findings PDF: split pages 2-16 (vector) + render PNG
d = os.path.join(OUT, X["name"])
rd = pypdf.PdfReader(SEPT); doc = fitz.open(SEPT)
for k, title in enumerate(SEPT_TITLES):
    pg = k + 1
    n = f"{11+k:02d}_SUPERSEDED_Sept30_findings_Figure{k+1}"
    w = pypdf.PdfWriter(); w.add_page(rd.pages[pg])
    with open(os.path.join(d, n + ".pdf"), "wb") as f: w.write(f)
    doc[pg].get_pixmap(dpi=200).save(os.path.join(d, n + ".png"))
    g = dict(n=n, png=SEPT + f" (page {pg+1})", pdf=SEPT, kind="vector (page split from source PDF); PNG rendered at 200 dpi",
             axes="Page of the 2026-09-30 findings PDF: figure plus its caption.", pattern="-",
             why="SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).",
             nums=f"Sept-30 Figure {k+1}: {title}.")
    X["figs"].append(g); ALL.append((X, g, os.path.join(d, n + ".png")))

# remove placeholders from the first build (S2-S6 are now regenerated)
for root, _, files in os.walk(OUT):
    for fn in files:
        if fn in ("MISSING.txt", "M1_vs_M0_exact_UNVERIFIED.csv"): os.remove(os.path.join(root, fn))

# ---------- ALL_FIGURES.pdf ----------
with PdfPages(os.path.join(OUT, "ALL_FIGURES.pdf")) as pp:
    fig = plt.figure(figsize=(11, 8.5)); fig.text(0.5, 0.6, "Evidence package: time-varying-field Ising model of Purkinje-cell populations", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.52, f"{len(ALL)} figures in INDEX.md order. Each page: figure + short caption. Folder X_ = withdrawn/superseded (do not cite).", ha="center", fontsize=10)
    pp.savefig(fig); plt.close(fig)
    for F, g, png in ALL:
        im = Image.open(png).convert("RGB"); im.thumbnail((2400, 2400))
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.03, 0.22, 0.94, 0.70]); ax.imshow(im); ax.axis("off")
        head = f"{F['name']} / {g['n']}   [{F['status']}]"
        fig.text(0.03, 0.96, textwrap.fill(head, 130), fontsize=10, weight="bold", color="firebrick" if F["name"].startswith("X_") else "black", va="top")
        cap = f"Shows: {g['axes']}  Pattern: {g['pattern']}  Why: {g['why']}  Numbers: {g['nums']}"
        fig.text(0.03, 0.20, textwrap.fill(cap, 190), fontsize=7, va="top")
        pp.savefig(fig, dpi=150); plt.close(fig)

# ---------- INDEX.md ----------
SHORT = lambda p: p.replace("/Users/gunnarenserro/Documents/AlBornoMazenlab/code/", "").replace("/private/tmp/claude-501/-Users-gunnarenserro-Documents-AlBornoMazenlab-code-ising-EnergyOverTime/ed64efe7-7195-42e3-9e96-fff305689ec3/scratchpad/evpkg/e11/", "[E11 via git] ")
L = []
L.append(open(os.path.join(os.path.dirname(__file__), "index_head.md")).read())
for F in FOLDERS + [X]:
    L.append(f"\n## {F['name']}\n\n**Claim.** {F['claim']}\n\n**Status.** {F['status']}\n")
    if F["tables"]:
        L.append("**Tables copied here:** " + ", ".join(f"`{n or os.path.basename(t)}`" for t, n in F["tables"]) + "\n")
    if not F["figs"]:
        L.append("**No figures: source files missing (see Missing).**\n")
    for g in F["figs"]:
        src = g.get("src") and ((g["src"] if g["src"].startswith("evidence/") else "ising_wt_protocol: " + g["src"])) or SHORT(g["png"])
        if g["pdf"] and g["pdf"] != SEPT: src += f"; PDF: {SHORT(g['pdf'])}"
        L.append(f"\n### {g['n']}.png / .pdf\n- **Source:** {src}\n- **Axes:** {g['axes']}\n- **Pattern:** {g['pattern']}\n- **Why it supports the claim:** {g['why']}\n- **Key numbers:** {g['nums']}\n- **PDF:** {g['kind']}\n")
L.append(open(os.path.join(os.path.dirname(__file__), "index_tail.md")).read())
open(os.path.join(OUT, "INDEX.md"), "w").write("\n".join(L))

# ---------- zip ----------
zp = os.path.join(OUT, "evidence_package.zip")
if os.path.exists(zp): os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if p == zp or fn == ".DS_Store": continue
            z.write(p, os.path.join("evidence_package", os.path.relpath(p, OUT)))
nv = sum(1 for _, g, _ in ALL if g["kind"].startswith("vector")); print("figures", len(ALL), "vector", nv, "raster", len(ALL) - nv)
