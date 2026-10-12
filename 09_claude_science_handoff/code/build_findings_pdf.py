import os, matplotlib
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", f"{fd}/DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DVB", f"{fd}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", f"{fd}/DejaVuSans-Oblique.ttf"))
from reportlab.lib.fonts import addMapping
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DVI"); addMapping("DV", 1, 1, "DVB")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DVB", fontSize=16, leading=20, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DVB", fontSize=12.5, leading=16, spaceBefore=6, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName="DV", fontSize=9.2, leading=12.6, spaceAfter=4)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8.6, leading=11.6, textColor=colors.HexColor("#222222"))
ELI = ParagraphStyle("E", parent=BODY, fontSize=9, leading=12.4, backColor=colors.HexColor("#F2EEF7"), borderPadding=(5, 6, 5, 6),
                     leftIndent=6, rightIndent=6, spaceBefore=6, spaceAfter=10)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.8, leading=10)

W = 6.9 * inch; HMAX = 5.3 * inch


def img(path, hmax=HMAX):
    w, h = PILImage.open(path).size; r = h / w
    ww = W; hh = ww * r
    if hh > hmax: hh = hmax; ww = hh / r
    return Image(path, width=ww, height=hh)


story = []
def P(t, st=BODY): story.append(Paragraph(t, st))


def figure(n, title, path, caption, plain, hmax=HMAX):
    block = [Paragraph(f"Figure {n}. {title}", H2), img(path, hmax), Spacer(1, 4),
             Paragraph(f"<b>Figure {n}.</b> {caption}", CAP),
             Paragraph(f"<b>In plain terms.</b> {plain}", ELI)]
    story.append(KeepTogether(block))


FIGS = __import__("json").load(open("fig_paths.json"))

# ---------------- Title and summary ----------------
P("Pairwise maximum-entropy (Ising) analysis of Purkinje-cell populations during reaching: findings to date", H1)
P("Compiled 2026-09-30 from the project's saved artifacts (RESULTS_LOG_FOR_PAPER.md v3 plus later updates). "
  "Dataset: Calame, Becker &amp; Person (bioRxiv 10.1101/2021.12.17.473247). There are 21 sessions × 3 blocks (baseline, closed-loop mossy-fiber "
  "stimulation, washout) = 63 conditions, with 3–32 Purkinje cells per session, 1,249 reaches and 10-ms bins. The unit of replication is the session "
  "(n = 21), tested with session-level Wilcoxon tests and BH-FDR correction. Model: E = −(Σ J<sub>ij</sub> s<sub>i</sub> s<sub>j</sub> + Σ h<sub>i</sub> s<sub>i</sub>), "
  "with s ∈ {−1, +1}; J is the pairwise coupling and h the field (each neuron's own firing tendency).")
P("Summary", H2)
rows = [["Status", "Finding"],
        ["Established", "Populations show collective structure that independent neurons cannot produce (Fig. 2); the fitted couplings are not a fitting artefact (Fig. 3)."],
        ["Established", "Couplings improve prediction of held-out reaches beyond shared, reach-locked firing-rate changes in 21/21 sessions (Fig. 5)."],
        ["Established, with bounds", "No evidence that coupling strengthens during the reach. The estimate is imprecise: −15% change during acceleration, 95% CI −28% to +93% (Fig. 5)."],
        ["Robust, model-dependent", "The energy drop during the reach sits in the coupling term only in the stationary model (Fig. 7). With time-varying fields it sits in the field term (Fig. 5c)."],
        ["Robust", "Firing rate is the most movement-modulated signal (Fig. 10c). J is a broad elevation spanning the reach, not a timed peak (Fig. 8)."],
        ["Robust", "Timing of J, h and firing-rate maxima relative to the reach does not differ from control windows (Fig. 9)."],
        ["Robust", "J changes without firing-rate changes are rare (one clear example, Figs. 11–12)."],
        ["Suggestive", "A late post-reach coupling increase, confined to the 2022 mice (Fig. 13)."],
        ["Preliminary", "Coupling term around peak velocity rises during stimulation (uncorrected p ≈ 0.01, not FDR-significant; per-block fits) (Fig. 15)."],
        ["Withdrawn", "\"J peaks ~300 ms after peak acceleration\"; \"J lags acceleration in ~70% of sessions\"; \"J is the best indicator of peak acceleration\"; "
                      "a collective state locked to arm acceleration; the criticality proxy; the decision-tree J-peak predictor."]]
t = Table([[Paragraph(a, SMALL), Paragraph(b, SMALL)] for a, b in rows], colWidths=[1.35 * inch, 5.55 * inch])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTNAME", (0, 0), (-1, -1), "DV")]))
story.append(t); story.append(Spacer(1, 6))
P("<b>Recommended wording for the paper.</b> \"Purkinje-cell populations show pairwise coupling beyond their shared, reach-locked rate modulation. "
  "This coupling is already present at rest. At 10-ms resolution we found no evidence that it strengthens during reaching, although moderate changes cannot be "
  "excluded. Movement-related changes in population activity were carried mainly by the fields (firing rates).\"")
P("<b>The big picture in plain terms.</b> Think of each neuron as having a volume knob (its firing rate, the field h) and invisible strings tying "
  "it to other neurons (the couplings J), which make pairs fire or stay quiet together. The strings are real: they show up in every session and are not an "
  "artefact of fitting. They are there before the mouse moves, and we could not detect them tightening during the reach. What changes when the mouse reaches "
  "is mostly the volume knobs. Several earlier claims about J peaking at specific moments of the reach did not survive fair comparisons and were withdrawn.", ELI)
story.append(PageBreak())

# ---------------- Figures ----------------
figure(1, "The outward reach is brief", FIGS["phase_landmarks_qc"],
       "Mean x-velocity (top) and x-acceleration (bottom) of the hand for all 63 session × block conditions (grey), aligned to peak velocity "
       "(the acceleration zero-crossing, sample 0; 1 sample = 10 ms). Blue shading: acceleration phase; red shading: deceleration phase. Dashed lines: median peak "
       "acceleration and peak deceleration. Median acceleration phase lasts 90 ms and deceleration 120 ms. Position is min-max scaled within each reach, so units are arbitrary.",
       "The actual reach is over in about a fifth of a second. That matters because any neural signal that lasts longer than this cannot be \"locked\" to "
       "acceleration or deceleration just by where its peak lands: those windows are too short. Every later timing test was built with this in mind.")

figure(2, "Population activity is more collective than independent neurons predict", FIGS["falsification_test"],
       "Left: probability that K of the N = 25 neurons are active in the same 10-ms bin, P(K), for session 210614 stim 0 (red), against the prediction "
       "for independent neurons with the same firing rates (grey). Right: deviation of the observed P(K) from a shuffled independent-firing null, as a z-score, for each of "
       "the 27 conditions with raw data (z = 4.9 to 116; dashed line z = 3). This is a model-free test. The shuffle also removes reach-locked rate modulation, so it "
       "establishes structure beyond stationary independence only.",
       "If neurons fired independently, the count of neurons active at once would follow the grey curve. The real data have fatter tails: more bins where many "
       "or few neurons fire together. Every condition passes this test. It proves the population is not a set of independent coin flips, but not yet why.")

figure(3, "Fitted couplings are not a fitting artefact", FIGS["surrogate_refit_J"],
       "(a) Mean |J<sub>ij</sub>| of the refitted Ising model for the real data (purple; mean of 3 refits) and the range over 100 rate-preserving shuffled "
       "surrogates (grey bars) per condition, sorted by real value; the real value exceeds every surrogate in 63/63 conditions (median 4.6×). (b) Net coupling ΣJ<sub>ij</sub> "
       "of the real fit as a z-score against the surrogates, against population size N. Dotted line z = 2. The one exception, 210622 stim 2, has only N = 3 neurons.",
       "A fitting algorithm can report non-zero couplings even from pure noise. So we scrambled each neuron's spikes to destroy any co-firing while keeping its "
       "firing rate, then refit. The scrambled data give couplings several times smaller in every condition, so the real couplings reflect genuine co-firing.")

figure(4, "Validation of the time-varying-field model", FIGS["tvfield_validation"],
       "(a) Synthetic data with known parameters: held-out log-likelihood gain over the time-varying independent model (M0) from adding couplings (M1 − M0, purple), "
       "from other neurons' states lagged 200 ms (grey), and for the trial-shuffle null (green). The couplings help only when true J is present (+0.107 nats/bin; J recovered with "
       "r = 0.98), including when a per-reach gain fluctuation is simulated with J = 0. (b) Real data: couplings from the pseudo-likelihood fit against the stored MCH (coniii) fit. "
       "They agree (r = 0.995–0.997) except in 210511 stim 1 (r = 0.65), where one neuron fires in 99% of bins. There the pseudo-likelihood fit reproduces the "
       "data's pairwise correlations more closely (rms error 0.0006 vs 0.0011).",
       "Before using a new tool on real data, we tested it on fake data where we know the answer. It finds couplings when they exist and reports none when they "
       "don't, even when the fake neurons share slow ups and downs. On real data it agrees with the lab's original fitting method.")

figure(5, "Couplings add prediction beyond shared rate changes, and the drop in energy moves to the fields", FIGS["tvfield_vs_independent"],
       "(a) Held-out gain over a time-varying-rate independent model (nats per 10-ms bin; session median and IQR, n = 21), for couplings on simultaneous states (purple), "
       "for a control using states lagged 200 ms (grey), and for the trial-shuffle null (green). The shading marks acceleration (blue) and deceleration (red). "
       "(b) Gain minus null, by phase, per session. The gain is above the null in 21/21 sessions and 63/63 conditions (median +0.26 nats/bin) and does not differ between "
       "baseline and the reach (change p ≥ 0.45). The acceleration change relative to baseline is −15% (bootstrap 95% CI −28% to +93%), and couplings fitted separately for "
       "the reach epoch do not improve held-out prediction (9/21 sessions, p = 0.71). (c) Energy change per neuron against baseline: the coupling term in the stationary model "
       "(dashed), and the coupling term (purple) and field term (orange) in the time-varying-field model. With time-varying fields the reach-related energy drop is carried by "
       "the field term (acceleration 20/21 sessions, p<sub>FDR</sub> = 2×10<sup>−5</sup>). (d) Pseudo-likelihood gain against exact-likelihood gain for the 39 conditions with N ≤ 20 "
       "(38/39 positive; ρ = 0.999).",
       "This is the main result. We gave the model the ability to follow each neuron's firing rate moment by moment during the reach. The couplings still helped "
       "predict reaches the model had never seen, in every session, so the neurons really are tied together beyond just speeding up and slowing down in sync. "
       "The size of that help is the same at rest and during the reach. Once rates are properly accounted for, the energy change during the reach comes from the "
       "volume knobs (fields), not the strings (couplings). We cannot rule out a moderate change in coupling during the reach, because the reach is short and each "
       "session has only about 60 reaches.", hmax=5.6 * inch)

figure(6, "Where the stationary model's movement-time coupling signal comes from", FIGS["coupling_source_checks"],
       "(a) Change in the coupling term per pair against baseline (total, purple), split into the part predicted from each neuron's rate alone (green) and the "
       "trial-to-trial covariance part (orange). A median of only 36% comes from covariance, and neither part is consistent across sessions (p<sub>FDR</sub> &gt; 0.16). "
       "(b) Held-out log-likelihood ratio of the stationary Ising model against a constant-rate independent model: its advantage rises during the reach (20/21 sessions). "
       "(c) The same model against a time-varying-rate independent model: the reach-time advantage disappears; it is worse during acceleration (2/21 sessions, "
       "p<sub>FDR</sub> = 0.004) and rises again after the reach.",
       "Our original model uses a fixed volume knob for each neuron. When all neurons change their rates together during the reach, that model has nowhere to put the "
       "change except the strings, so the strings look like they tighten. Once rates are allowed to move (panel c), the apparent tightening during the reach goes "
       "away. This is why the stationary-model result in Figure 7 has to be interpreted carefully.")

figure(7, "Energy decomposition in the stationary model", FIGS["energy_decomposition_by_phase"],
       "(a) Change in energy against the pre-reach baseline (dashed black), split into the coupling contribution −ΔΣJss (purple) and the field contribution −ΔΣhs (orange), "
       "session mean ± SEM, n = 21; x-axis in 10-ms samples from peak velocity. (b) Phase-mean contributions per session. The coupling term falls below baseline in acceleration "
       "(p = 10<sup>−5</sup>), deceleration (p = 10<sup>−4</sup>) and post-movement (p = 7×10<sup>−4</sup>); field term p ≥ 0.84. <b>Model dependence:</b> this holds for the stationary "
       "model only (per-block fits with fixed fields). With time-varying fields the drop moves to the field term (Fig. 5c). The panel title states the stationary-model result.",
       "In the original model, the population's \"energy\" (lower means a more typical, more probable pattern) drops during the reach, and the drop comes "
       "from the coupling part. It is a real feature of that model. But Figures 5 and 6 show it happens because the fixed-rate model pushes the rate changes into the "
       "coupling part. So it is not evidence that neurons coordinate more during the reach.")

figure(8, "J is a broad elevation around the reach, not a timed peak", FIGS["J_timing_rise_decay"],
       "(a) Coupling term ΣJss (z against the pre-reach baseline) for all 1,249 reaches, sorted by the time of each reach's J peak (black dots). The diagonal is produced by the "
       "sorting and is not a result. (b) Session-mean J (n = 21, ± SEM): above z = 1 from −340 ms, half-maximum at −200 ms, peak z = 4.7 at +430 ms, back below half-maximum at "
       "+1,050 ms relative to peak velocity. (c) Distribution of single-reach J-peak times (purple), trial-averaged peaks per condition (black outline) and the same search in "
       "a pre-reach control window (grey). Single-reach peaks fall within ±50 ms of peak velocity in 12.4% of reaches against 6.4% in control windows (p &lt; 10<sup>−14</sup>), "
       "concentrated in 4 sessions (220516, 210425, 210608, 210512). Time axis in 10-ms samples.",
       "The coupling term rises before the hand moves, stays up for about a second, and slowly comes down. There is no sharp spike at a particular instant of the "
       "movement. The one timing hint: single reaches have their J peak right at top speed about twice as often as chance, but mainly in four sessions, and we "
       "found no property that explains why.", hmax=6.2 * inch)

figure(9, "Peak timing of J, h and firing rate matches chance", FIGS["peak_timing_vs_control"],
       "(a) Per-session lag of each signal's maximum after peak velocity minus the lag from the identical search in control windows at rest (shifted −1.5, −2 and −2.5 s). "
       "Black bars: medians. Only energy maxima deviate (earlier by about 350 ms, p<sub>FDR</sub> = 0.02). (b) Median real − control lag across six search-window and exclusion "
       "variants; filled markers p<sub>FDR</sub> &lt; 0.05. The energy deviation holds in only 2 of 6 variants; J, h and firing rate never differ from control.",
       "Earlier we reported that J peaks about 300 ms after acceleration. The search window was lopsided (it looked further forward than backward), so a peak "
       "would land \"after\" even in random noise. When the same search is run on quiet pre-reach data, the lags look the same, so that claim was withdrawn. Always "
       "compare against a fair control.")

figure(10, "Firing rate is the most movement-modulated signal", FIGS["phase_timing_summary"],
       "(a) Session-mean z-scored time courses of the coupling term, field term, firing rate and energy (x-axis in 10-ms samples from peak velocity). (b) Dominant-extremum lag per "
       "session. <b>Superseded by Fig. 9:</b> these p-values come from the asymmetric window and are withdrawn. (c) Movement modulation depth (peak |z|) of energy, coupling and "
       "field against firing rate in the same session; firing rate wins 16/21, 15/21 and 18/21 sessions (p<sub>FDR</sub> = 0.022, 0.044, 0.003). (d) Within-phase Spearman "
       "correlation of each signal with acceleration: none differs from zero (all p<sub>FDR</sub> &gt; 0.05). The title of panel a reflects the earlier framing; see Fig. 8 for "
       "the current description (broad elevation spanning the reach).",
       "Plain firing rate changes more strongly with the movement than any of the model's quantities. None of the signals rises and falls with acceleration within "
       "the reach, from moment to moment. The coupling term is interesting for what it measures (co-firing), not because it tracks the arm better.", hmax=5.8 * inch)

figure(11, "J changes without firing-rate changes are rare", FIGS["flat_FR_vs_J_modulation"],
       "Coupling-term movement modulation against firing-rate movement modulation (peak |z| against the pre-reach baseline; log axes) for 63 conditions, "
       "coloured by the Ising model's P(K) advantage over the independent model (Jensen–Shannon distance ratio). Grey region: bottom third of firing-rate modulation "
       "(peak |z| ≤ 6.8). Dashed line: J = FR. Open circles: no P(K) file on the cluster. Spearman ρ = 0.74.",
       "We looked for reaches where firing rates stay flat but coupling changes, which would be strong evidence of coordination on its own. They are rare: J and "
       "firing rate usually move together. One condition, 210511 stim 1, is a clear example (next figure).")

figure(12, "The clearest dissociation: 210511 stim 1", FIGS["example_flatFR_210511_stim1"],
       "Session 210511, stim 1 (21 reaches, N = 12 neurons), mean ± SEM, bold lines 5-sample smoothed. From top: x-velocity, x-acceleration, energy, coupling term ΣJss, field term "
       "Σhs, firing rate. Firing rate barely moves (peak |z| 2.8, about noise level), while the coupling term peaks 13 samples (130 ms) after peak velocity (|z| 5.5) "
       "and energy dips at the same time. One neuron in this condition fires in 99% of bins, which also affects the fit (Fig. 4b).",
       "Here the neurons' overall activity stays roughly flat during the reach, yet the co-firing signal jumps. It is the kind of case one would want if coupling "
       "carried information on its own. But it is one condition out of 63, so it is an example, not a pattern.", hmax=6.4 * inch)

figure(13, "Suggestive: a late post-reach coupling increase in the 2022 mice", FIGS["late_epoch_consistency"],
       "(a) Late-epoch (+0.5 to +1.5 s) Ising advantage over the time-varying independent model per session, from odd against even reaches (ρ = 0.89 across conditions). "
       "(b) Late minus baseline coupling gain for simultaneous states (purple) and 200-ms lagged states (grey), by cohort (symmetric-log axis). 2022: 6/6 sessions, median "
       "+1.97 nats/bin; 2021: 7/15, p = 0.45. The lagged control also rises in 2022 (6/6), so part of the effect is slow shared fluctuation. (c) Held-out gain from letting "
       "couplings change in the late epoch: no improvement in either cohort. The 2022 cohort is 2 mice (DJC000, DJC002).",
       "About a second after the reach, a coupling increase shows up, but only in recordings from two mice made in 2022. It is consistent within those sessions, "
       "but some of it is slow drift, and letting the couplings change there does not improve predictions. Treat it as a lead to check in new recordings, not a finding.")

figure(14, "What distinguishes the sessions with the strongest effects", FIGS["strong_session_profile"],
       "(a) Spearman ρ between each session-level effect (rows) and recording properties (columns), n = 21; * p<sub>FDR</sub> &lt; 0.05. Coupling gain tracks the maximum pair "
       "correlation (ρ = 0.91), N (0.66), mean |pair correlation| (0.56) and the 2022 cohort (0.70). (b) Coupling gain minus null after removing one unit of every pair with "
       "r &gt; 0.5, against the session's maximum pair correlation before pruning. The gain stays above the null in 63/63 conditions after pruning.",
       "Sessions with more strongly correlated neurons show more coupling, which makes sense. We removed near-duplicate neuron pairs, in case one cell had been "
       "recorded twice. The coupling survives that. The 2022 recordings stand out as more correlated overall; the lab records may show why (animals, site, sorting).")

figure(15, "Preliminary: changes across baseline, stimulation and washout", FIGS["stim_block_reach_trajectories"],
       "Change from the last 5 baseline reaches (in units of the SD across baseline reaches), in a ±100-ms window around each reach's peak velocity, for (a) the coupling term "
       "per pair, (b) the field term per neuron and (c) firing rate. Positions along x: last 5 baseline reaches, then the first reach, reaches 1–5, the middle 5 and the last 5 of the "
       "stimulation block (blue shading) and of the washout block. Thin lines: sessions; thick line: median; band: IQR; n = 21. * p &lt; 0.05 uncorrected; none survives FDR. "
       "Coupling term: first stimulated reach +0.93 SD (16/21, p = 0.009), last 5 stimulated reaches +0.33 (17/21, p = 0.013), first 5 washout reaches +0.48 (14/21, p = 0.038). "
       "<b>Caveats:</b> each block had its own Ising fit, so the comparison mixes parameter changes with activity changes; the window is aligned to peak velocity as a proxy for the "
       "1-cm stimulation trigger. A joint one-model-per-session test has been built and validated on synthetic data; it awaits the raw stimulation timing.",
       "In the stimulation block, a brief light pulse drives the cerebellum's inputs as the hand passes 1 cm. The coupling term around that moment is higher from the "
       "very first stimulated reach and fades in washout, loosely like the behavioral adaptation in the dataset paper. But it is not statistically solid yet, and the "
       "comparison is not yet fair, because each block was fitted separately. The proper test, one model per session with the true pulse times, is next.")

doc = SimpleDocTemplate("ISING_FINDINGS_FIGURES.pdf", pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                        title="Purkinje-cell Ising analysis: findings and figures", author="")
def footer(c, d):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(colors.grey)
    c.drawRightString(letter[0] - 0.8 * inch, 0.45 * inch, f"{d.page}"); c.drawString(0.8 * inch, 0.45 * inch, "Purkinje-cell Ising analysis: findings to date (2026-09-30)")
    c.restoreState()
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("pages built")
