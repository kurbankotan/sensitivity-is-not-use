"""EnsAff makale sekilleri. python3 figures.py -> paper/figs/*.png + *.pdf"""
import sys, numpy as np
sys.path.insert(0, "/home/claude/paper")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (setup, save, PALETTE as C, COL1, COL2, HATCH, MARKER,
                      DECOMP, DECOMP_TOTAL, PROBES, PROBE_SD, FORCING_SENS,
                      FORCING_RBASE, FORCING_ACC, CONTROLS, POSITION, POSE,
                      RESOLUTION, PRIMARY, CALIB, POSCTRL, POSCTRL_BDB,
                      DISSOCIATION, SHELLS, DYN, GEOM_SIGNAL, SEED_NOISE)
setup()
import matplotlib as _mpl
_mpl.rcParams["hatch.linewidth"] = 0.55


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=C["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Sekil 2
def fig_decomposition():
    """Serugu: temel modelden tam modele giden yol, her bilesen ayri.

    Geometri basamagi bir SUTUN DEGILDIR. Olculen etki ters yonde (-0.013 pK,
    yani geometri yokken hafifce daha iyi) ve tohum testinden gecmiyor; yukari
    dogru bir sutun onu gercek bir katki gibi gosterirdi. Bunun yerine tam
    modelin duzeyi isaretlenir ve sabit mesafe kolunun nerede durdugu bos bir
    daireyle gosterilir.
    """
    fig, ax = plt.subplots(figsize=(COL2 * 0.70, 2.5))
    base, full = 1.602, 1.488
    gconst = 1.475                      # sabit mesafe kolu, 5 tohum toplulugu
    steps = [("Baseline", "total", base),
             ("Per-residue\nrepresentation", "step", -0.063),
             ("Pocket\nselection", "step", -0.051),
             ("3D\ngeometry", "geom", gconst),
             ("Structure\nmodel", "total", full)]
    x = np.arange(len(steps))
    run = base
    for i, (lab, kind, val) in enumerate(steps):
        if i:
            ax.plot([i - 0.69, i + 0.31], [run, run], color=C["mute"], lw=0.7,
                    ls=(0, (2, 2)), zorder=2)
        if kind == "total":
            ax.bar(i, val, 0.62, color=C["zero"], edgecolor=C["ink2"], zorder=3)
            ax.text(i, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
            run = val
        elif kind == "step":
            new_ = run + val
            ax.bar(i, -val, 0.62, bottom=new_, color=C["s1"], edgecolor=C["ink2"],
                   zorder=3)
            ax.text(i, run + 0.012, f"{val:+.3f}", ha="center", va="bottom",
                    fontsize=7.5)
            run = new_
        else:                            # geometri: sutun yok
            ax.plot([i - 0.31, i + 0.31], [run, run], color=C["s2"], lw=2.2,
                    solid_capstyle="butt", zorder=5)
            ax.plot([i, i], [run, val], color=C["s2"], lw=1.0, ls=(0, (2, 1.5)),
                    zorder=4)
            ax.plot(i, val, "o", ms=4.6, color="white", mec=C["s2"], mew=1.2,
                    zorder=5)
            ax.annotate("-0.013 pK,\nnot significant\nacross seeds",
                        xy=(i, run), xytext=(i, run + 0.052), ha="center",
                        fontsize=6.6, fontweight="bold", color=C["s2"],
                        arrowprops=dict(arrowstyle="-", color=C["s2"], lw=0.8))
            ax.text(i, val - 0.006, "constant distances", fontsize=6.3,
                    va="top", ha="center", color=C["s2"])
            # run degismez: bu basamak yolu ilerletmez
    ax.plot([len(steps) - 1.69, len(steps) - 1 + 0.31], [full, full],
            color=C["mute"], lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in steps], fontsize=7.5)
    ax.set_ylim(1.448, 1.648)
    ax.set_ylabel("Test RMSE (pK)")
    _grid(ax, "y")
    ax.set_title("Where the structural advantage comes from", loc="left",
                 fontsize=8.5, pad=6)
    save(fig, "fig2_decomposition")


# ------------------------------------------------------------------ Sekil 3
def fig_dose():
    """Yer degistirme dozu buyurken tahmin degisimi ve korelasyon."""
    sig = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    key = [f"translate {s:g} A".replace(".0 ", " ") for s in sig]
    key = ["translate 0.5 Å", "translate 1 Å", "translate 2 Å",
           "translate 4 Å", "translate 8 Å"]
    dmu = np.array([PROBES[k][2] for k in key])
    rb = np.array([PROBES[k][3] for k in key])
    mean_disp = sig * 1.596                      # 3B Gauss yer degistirmenin ortalamasi

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    a, b = axes

    a.plot(mean_disp, dmu, "-", color=C["s1"], marker=MARKER["s1"], ms=4.5,
           mfc="white", mew=1.2, zorder=4)
    a.axhline(0.05, color=C["ink2"], lw=0.8, ls=(0, (4, 2)), zorder=2)
    a.text(1.0, 0.0545, "probe threshold 0.05 pK", fontsize=6.5, color=C["ink2"])
    a.set_yscale("log")
    a.set_ylim(3e-3, 1.2e-1)
    a.set_xlabel("Mean ligand displacement (Å)")
    a.set_ylabel("Change in prediction (pK)")
    a.set_title("(a) Change in prediction", loc="left", fontsize=8)
    _grid(a, "y")

    b.plot(mean_disp, rb, "-", color=C["s1"], marker=MARKER["s1"], ms=4.5,
           mfc="white", mew=1.2, zorder=4)
    b.axhline(0.99, color=C["ink2"], lw=0.8, ls=(0, (4, 2)), zorder=2)
    b.text(1.0, 0.9906, "probe threshold 0.99", fontsize=6.5, color=C["ink2"])
    b.set_ylim(0.9855, 1.0015)
    b.set_xlabel("Mean ligand displacement (Å)")
    b.set_ylabel("Correlation with unperturbed")
    b.set_title("(b) Correlation with unperturbed run", loc="left", fontsize=8)
    _grid(b, "y")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_dose")




# ------------------------------------------------------------------ Sekil 4
def fig_forcing():
    """Zorlama duyarliligi buyuttu, dogrulugu buyutmedi."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    a, b = axes

    labs = [r[0] for r in FORCING_SENS]
    y = np.arange(len(labs))[::-1]
    for yi, (_, un, fo, fac) in zip(y, FORCING_SENS):
        a.plot([un, fo], [yi, yi], color=C["mute"], lw=1.1, zorder=2)
        a.plot(un, yi, MARKER["s1"], color=C["s1"], ms=5.5, mec="white", mew=0.8,
               zorder=4)
        a.plot(fo, yi, MARKER["s2"], color=C["s2"], ms=5.5, mec="white", mew=0.8,
               zorder=4)
        a.text(fo * 1.35, yi, fac, fontsize=7, va="center", color=C["ink2"])
    a.set_xscale("log")
    a.set_xlim(2e-3, 2.2)
    a.set_yticks(y)
    a.set_yticklabels(labs, fontsize=7)
    a.set_xlabel("pK (log scale)")
    a.set_title("(a) Sensitivity to geometry", loc="left", fontsize=8)
    un0, fo0 = FORCING_SENS[0][1], FORCING_SENS[0][2]
    a.annotate("unmodified", xy=(un0, y[0]), xytext=(un0, y[0] + 0.42),
               ha="center", fontsize=6.8, color=C["s1"])
    a.annotate("forced", xy=(fo0, y[0]), xytext=(fo0, y[0] + 0.42),
               ha="center", fontsize=6.8, color=C["s2"])
    a.set_ylim(-0.6, len(labs) - 0.25)
    _grid(a, "x")

    labs2 = [r[0] for r in FORCING_ACC]
    y2 = np.arange(len(labs2))[::-1]
    for yi, (_, m, lo, hi) in zip(y2, FORCING_ACC):
        worse = lo > 0
        col = C["s2"] if worse else C["s1"]
        b.plot([lo, hi], [yi, yi], color=col, lw=1.4, zorder=3,
               solid_capstyle="butt")
        for e in (lo, hi):
            b.plot([e, e], [yi - 0.13, yi + 0.13], color=col, lw=1.4, zorder=3)
        b.plot(m, yi, MARKER["s1"] if not worse else MARKER["s2"], color=col,
               ms=5.5, mec="white", mew=0.8, zorder=4)
        b.text(hi + 0.012, yi, f"{m:+.3f}", fontsize=6.8, va="center",
               color=C["ink2"])
    b.axvline(0, color=C["ink2"], lw=0.9, zorder=2)
    b.set_yticks(y2)
    b.set_yticklabels(labs2, fontsize=7)
    b.set_xlim(-0.06, 0.20)
    b.set_xlabel("Paired RMSE difference (pK)")
    b.set_title("(b) Affinity accuracy", loc="left", fontsize=8)
    b.text(-0.055, -0.80, "forced better", fontsize=6.5, color=C["s1"], ha="left")
    b.text(0.195, -0.80, "forced worse", fontsize=6.5, color=C["s2"], ha="right")
    b.set_ylim(-1.15, len(labs2) - 0.5)
    _grid(b, "x")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig4_forcing")


# ------------------------------------------------------------------ Sekil 5
def fig_controls():
    """Yeniden egitilen kontrol kollari."""
    fig, ax = plt.subplots(figsize=(COL2 * 0.66, 2.35))
    rows = sorted(CONTROLS, key=lambda r: -r[2])
    y = np.arange(len(rows))
    for yi, (lab, ens, mu, sd, n) in zip(y, rows):
        ax.plot([mu - sd, mu + sd], [yi, yi], color=C["s1"], lw=1.4, zorder=3,
                solid_capstyle="butt")
        for e in (mu - sd, mu + sd):
            ax.plot([e, e], [yi - 0.12, yi + 0.12], color=C["s1"], lw=1.4, zorder=3)
        ax.plot(mu, yi, MARKER["s1"], color=C["s1"], ms=5.5, mec="white", mew=0.8,
                zorder=5)
        ax.plot(ens, yi, "D", color="white", mec=C["ink2"], mew=1.0, ms=4.2,
                zorder=5)
    ax.axvline(rows[0][2], color=C["zero"], lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(rows[0][2] - 0.004, len(rows) - 0.52, "baseline", fontsize=6.8,
            ha="right", color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\n({r[4]} seeds)" for r in rows], fontsize=7.5)
    ax.set_xlim(1.466, 1.648)
    ax.set_xlabel("Test RMSE (pK)")
    ax.legend(handles=[Line2D([], [], ls="", marker=MARKER["s1"], color=C["s1"],
                              ms=5.5, label="Per seed (mean +/- s.d.)"),
                       Line2D([], [], ls="", marker="D", color="white",
                              mec=C["ink2"], ms=4.2, label="Ensemble")],
              loc="lower left", handlelength=1.0, ncol=1,
              bbox_to_anchor=(0.01, 0.02))
    _grid(ax, "x")
    save(fig, "fig5_controls")


# ------------------------------------------------------------------ Sekil 6
def fig_pose():
    """Poz ayirt etme: sans duzeyi, ulasilan, on-kayitli tahmin."""
    fig, ax = plt.subplots(figsize=(COL1 * 1.22, 2.15))
    labs = ["Forced arm", "Forced arm,\nhidden contacts"]
    vals = [POSE["forced"], POSE["wide"]]
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals, 0.46, color=[C["s2"], C["mute"]],
            hatch=[HATCH["s2"], ""], edgecolor=C["ink2"], zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.5,
                fontweight="bold")
    ax.axvline(POSE["chance"], color=C["ink2"], lw=1.0, zorder=4)
    ax.axvline(POSE["predicted"], color=C["ink2"], lw=1.0, ls=(0, (4, 2)),
               zorder=4)
    ax.text(POSE["chance"] - 0.012, 1.42, "chance $\\leq$ 0.200", fontsize=7,
            ha="right", color=C["ink2"])
    ax.text(POSE["predicted"] - 0.012, 1.42, "prespecified\nprediction > 0.80",
            fontsize=7, ha="right", color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.45, 1.75)
    ax.set_xlabel("Pose discrimination accuracy")
    _grid(ax, "x")
    save(fig, "fig6_pose")


# ------------------------------------------------------------------ Sekil 7
def fig_position():
    """Yayimlanmis modellere gore konum. Bas basa karsilastirma DEGILDIR.

    Yalnizca yeniden egitilmis degerler cizilir; ozgun bildirilen degerler
    tabloda kalir, cunku farkli egitim duzenlerinden gelirler.
    """
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 2.3))
    rows = sorted(POSITION, key=lambda r: (r[4], -r[2]))
    y = np.arange(len(rows))
    for yi, (lab, orig, retr, sd, mine) in zip(y, rows):
        ax.barh(yi, retr, 0.56, color=C["s1"] if mine else C["mute"],
                hatch="///" if mine else "", edgecolor=C["ink2"], zorder=3)
        if sd:
            ax.plot([retr - sd, retr + sd], [yi, yi], color=C["ink"], lw=1.0,
                    zorder=5)
            ax.plot([retr + sd, retr + sd], [yi - 0.11, yi + 0.11],
                    color=C["ink"], lw=1.0, zorder=5)
        ax.text(retr + (sd or 0) + 0.07, yi, f"{retr:.2f}", va="center",
                fontsize=7.5, fontweight="bold" if mine else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 3.05)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Test error (kcal/mol)")
    # gruplari ayiran cizgi ve kenar disinda parantez etiketleri
    ax.axhline(len(rows) - 2.5, color=C["ink2"], lw=0.7, ls=(0, (3, 2)), zorder=6)
    ax.text(0.04, len(rows) - 1.0, "this work, CL1 split", fontsize=6.8,
            color="white", va="center", fontweight="bold")
    ax.text(0.04, 0, "published, retrained by Li et al., CL2-filtered split",
            fontsize=6.8, color="white", va="center")
    _grid(ax, "x")
    save(fig, "fig8_position")


# ------------------------------------------------------------------ Sekil 8
def fig_resolution():
    """Dis kumeler olculen etkiyi cozemez."""
    fig, ax = plt.subplots(figsize=(COL2 * 0.66, 2.0))
    y = np.arange(len(RESOLUTION))[::-1]
    eff = abs(PRIMARY["rmse"][2])
    ax.axvspan(-eff, 0, color=C["s1"], alpha=0.10, zorder=1)
    ax.text(-eff / 2, 2.52, f"effect being measured\n({eff:.3f} pK)", fontsize=6.8,
            ha="center", color=C["s1"])
    for yi, (lab, m, lo, hi, n) in zip(y, RESOLUTION):
        hw = (hi - lo) / 2
        resolves = hw < eff
        col = C["s1"] if resolves else C["s2"]
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.5, zorder=3,
                solid_capstyle="butt")
        for e in (lo, hi):
            ax.plot([e, e], [yi - 0.12, yi + 0.12], color=col, lw=1.5, zorder=3)
        ax.plot(m, yi, MARKER["s1"] if resolves else MARKER["s2"], color=col,
                ms=5.5, mec="white", mew=0.8, zorder=4)
        ax.text(0.30, yi, f"n = {n}", fontsize=7, va="center", color=C["ink2"])
        ax.text(0.455, yi, f"+/-{hw:.3f}", fontsize=7, va="center", color=col)
    ax.axvline(0, color=C["ink2"], lw=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in RESOLUTION], fontsize=7.5)
    ax.set_xlim(-0.33, 0.56)
    ax.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylim(-0.6, 2.95)
    ax.set_xlabel("RMSE difference, structure model minus baseline (pK)")
    ax.text(0.385, 2.52, "half-width", fontsize=6.8, ha="center", color=C["ink2"])
    _grid(ax, "x")
    save(fig, "fig7_resolution")


# ------------------------------------------------------------------ Sekil 1
def fig_design():
    """Mesafe kanali modele nereden girer ve her mudahale hangi teli keser."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(COL2, 3.45))
    ax.set_xlim(0, 100); ax.set_ylim(-27, 47); ax.axis("off")

    def box(x0, x1, y0, y1, text, fc="white", fs=6.8, bold=False):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.5,rounding_size=1.1",
                                    fc=fc, ec=C["ink2"], lw=0.8, zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4,
                fontweight="bold" if bold else "normal")

    def wire(pts):
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            ax.plot([xa, xb], [ya, yb], color=C["ink2"], lw=0.9, zorder=2,
                    solid_capstyle="round")
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.9, color=C["ink2"],
                                     zorder=2, shrinkA=0, shrinkB=0))

    def mark(x, y, n, ms=9.5, fs=6.0):
        # nokta isaretcisi punto cinsindendir, eksen orani ne olursa olsun yuvarlak kalir
        ax.plot(x, y, "o", ms=ms, color=C["s2"], mec="white", mew=0.9, zorder=7,
                clip_on=False)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=8)

    box(2, 20, 37, 45, "Protein\nsequence")
    box(2, 20, 21, 32, "Pocket residues\nnearest 64 within 10 Å")
    box(2, 20, 1, 9, "Ligand\nheavy atoms")
    box(26, 42, 37, 45, "ESM-2 650M")
    box(26, 42, 21, 32, "Distance matrix\nD [residue, atom]", fc="#eaf1fc")
    box(26, 42, 1, 9, "Atom features")
    box(47, 60, 21, 32, "RBF encoding\n16 centres", fc="#eaf1fc")
    box(65, 83, 6, 40,
        "Pair representation\nz [residue, atom]\n\naxial attention\n"
        "+ triangle update\nx 3 recycles\n\nlearned pooling")
    box(87, 99, 21, 32, "Affinity\nmu, sigma", bold=True)

    wire([(20, 41), (25.4, 41)])
    wire([(42, 41), (74, 41), (74, 40.6)])
    wire([(20, 26.5), (25.4, 26.5)])
    wire([(42, 26.5), (46.4, 26.5)])
    wire([(60, 26.5), (64.4, 26.5)])
    wire([(20, 5), (25.4, 5)])
    wire([(42, 5), (74, 5), (74, 5.4)])
    wire([(83, 26.5), (86.4, 26.5)])

    mark(22.7, 41, 1)
    mark(22.7, 26.5, 2)
    mark(44.2, 29.6, 3)
    mark(44.2, 23.4, 4)
    mark(84.7, 26.5, 5)

    key = [
        "baseline arm: the protein branch is removed and a whole-chain mean "
        "embedding is used instead",
        "random-pocket arm: the contact residues are replaced by residues drawn "
        "at random from the same protein",
        "retrained damage: every distance is set to 8 Å, or the residue axis of D "
        "is permuted, in training and evaluation alike",
        "perturbation probes: D is damaged at inference only, on an already "
        "trained model",
        "forcing arm: an auxiliary loss requires the crystal pose to outscore "
        "rotated decoys",
    ]
    for i, line in enumerate(key):
        yy = -5.0 - 4.4 * i
        mark(2.2, yy, i + 1, ms=8.0, fs=5.4)
        ax.text(6.0, yy, line, fontsize=6.3, va="center", ha="left",
                color=C["ink"])
    save(fig, "fig1_design")


# ------------------------------------------------------------------ Sekil 9
def fig_calibration():
    """Iki olcut, iki sentetik rejim: hangisi dogru cevap veriyor."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35), sharey=True)
    a, b = axes
    labs = list(CALIB)
    y = np.arange(len(labs))[::-1]
    truth = {"Geometry only": True, "Composition only": False}

    for ax, col, title, thr in ((a, 4, "(a) Retrained control arm", None),
                                (b, 5, "(b) Inference-time probe", 0.05)):
        for yi, k in zip(y, labs):
            v = CALIB[k][col]
            says = v > 0.05
            right = (says == truth[k])
            ax.barh(yi, v, 0.45, color=C["s1"] if right else C["s2"],
                    hatch="" if right else "///", edgecolor=C["ink2"], zorder=3)
            tx = max(v, 0.05) + 0.055       # sifira yakin sutunlarda etiket esigin sagina
            ax.text(tx, yi, f"{v:+.3f}", va="center", fontsize=7.5, fontweight="bold")
            ax.text(tx, yi - 0.30, ("correct" if right else "FALSE POSITIVE"),
                    va="center", fontsize=6.5, color=C["s1"] if right else C["s2"])
        ax.axvline(0.05, color=C["ink2"], lw=0.9, ls=(0, (4, 2)), zorder=4)
        ax.set_xlim(-0.08, 1.62)
        ax.set_ylim(-0.75, len(labs) - 0.35)
        ax.set_xlabel("Cost of removing geometry (pK)")
        ax.set_title(title, loc="left", fontsize=8)
        _grid(ax, "x")
    a.text(0.062, len(labs) - 0.52, "decision threshold", fontsize=6.5,
           color=C["ink2"])
    a.set_yticks(y)
    a.set_yticklabels([f"{k}\ndrives the target" for k in labs], fontsize=7.5)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig9_calibration")


# ------------------------------------------------------------------ TOC gorseli
def fig_toc():
    """ACS icindekiler gorseli. En fazla 3.25 x 1.75 inc, 300 dpi'dan yuksek.

    Artik ayrismayi degil pozitif kontrolu anlatiyor: ayni girdi, iki model,
    biri geometriyi kullaniyor digeri kullanmiyor, ve kullanmayan daha iyi degil.
    """
    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 97, "Same pocket, same distances.", fontsize=9.2,
            fontweight="bold", va="top", color=C["ink"])
    ax.text(0, 85, "Only one model uses them.", fontsize=9.2,
            fontweight="bold", va="top", color=C["ink"])

    rows = [("125 contact counts", 0.511, 1.462, C["s2"]),
            ("1.96M-parameter network", 0.004, 1.488, C["s1"])]
    scale = 56.0 / 0.511
    for k, (lab, cost, rmse, col) in enumerate(rows):
        yb = 48 - k * 26
        ax.text(0, yb + 13, lab, fontsize=6.9, va="center", color=C["ink"])
        ax.add_patch(plt.Rectangle((0, yb - 1), max(cost * scale, 0.8), 9,
                                   fc=col, ec=C["ink2"], lw=0.7, zorder=3))
        ax.text(max(cost * scale, 0.8) + 2.5, yb + 3.5, f"+{cost:.3f} pK",
                fontsize=7.4, fontweight="bold", va="center", color=col)
        ax.text(99, yb + 3.5, f"RMSE {rmse:.3f}", fontsize=6.6, va="center",
                ha="right", color=C["ink2"])
    ax.text(0, 8, "cost of setting every protein-ligand distance to 8 " + "\u00c5",
            fontsize=6.4, va="center", color=C["ink2"])
    ax.text(0, 0, "(retrained from scratch, the network loses nothing at all)",
            fontsize=6.4, va="center", color=C["ink2"])
    save(fig, "fig0_toc")


# ------------------------------------------------------------------ Sekil 10
def fig_poscontrol():
    """Sondalar kor degil: geometriyi kullanan modeli yikiyorlar."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
    names = list(POSCTRL)
    y = np.arange(len(names))[::-1]
    for ax, src, title, xlim in ((axes[0], POSCTRL, "(a) Test split (n = 4222)", (1.38, 2.45)),
                                 (axes[1], POSCTRL_BDB, "(b) BDB2020+ (n = 115)", (1.02, 2.45))):
        for yi, nm in zip(y, names):
            d = src[nm]
            cost = d["const"] - d["none"]
            col = C["s2"] if cost > 0.05 else C["s1"]
            ax.plot([d["none"], d["const"]], [yi, yi], color=col, lw=1.6, zorder=3,
                    solid_capstyle="butt")
            ax.plot(d["none"], yi, "o", ms=5.5, color="white", mec=C["ink2"],
                    mew=1.1, zorder=5)
            ax.plot(d["const"], yi, "s", ms=5.5, color=col, mec="white", mew=0.8,
                    zorder=5)
            ax.text(max(d["none"], d["const"]) + 0.035, yi, f"{cost:+.3f}",
                    va="center", fontsize=7.2, fontweight="bold", color=col)
        ax.set_xlim(*xlim)
        ax.set_ylim(-1.25, len(names) - 0.3)
        ax.set_xlabel("RMSE (pK)")
        ax.set_title(title, loc="left", fontsize=8)
        _grid(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7.5)
    axes[0].legend(handles=[Line2D([], [], ls="", marker="o", color="white",
                                   mec=C["ink2"], ms=5.5, label="real distances"),
                            Line2D([], [], ls="", marker="s", color=C["ink2"],
                                   ms=5.5, label="all distances 8 " + "\u00c5")],
                   loc="lower center", handlelength=1.0, ncol=2,
                   bbox_to_anchor=(0.52, -0.02), columnspacing=1.4)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig10_poscontrol")


# ------------------------------------------------------------------ Sekil 11
def fig_dissociation():
    """Iki alet, uc mimari: duyarlilik ile katki ayni sey degil."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True, sharex=True)
    y = np.arange(len(DISSOCIATION))[::-1]
    for ax, col, title in ((axes[0], 1, "(a) Removed at inference, already trained"),
                           (axes[1], 2, "(b) Removed from the training data, refitted")):
        for yi, row in zip(y, DISSOCIATION):
            v = row[col]
            if v is None:
                ax.text(0.04, yi, "not run", fontsize=7, va="center", color=C["mute"],
                        style="italic")
                continue
            big = abs(v) > 0.05
            c = C["s2"] if big else C["s1"]
            ax.barh(yi, v, 0.46, color=c, hatch="///" if big else "",
                    edgecolor=C["ink2"], zorder=3)
            ax.text(max(v, 0.0) + 0.05, yi, f"{v:+.4f}".rstrip("0").rstrip(".") + " pK",
                    va="center", fontsize=7.2, fontweight="bold", color=c)
        ax.axvline(0, color=C["ink2"], lw=0.9, zorder=4)
        ax.set_xlim(-0.12, 1.92)
        ax.set_ylim(-0.7, len(DISSOCIATION) - 0.35)
        ax.set_xlabel("Cost of removing the geometry (pK)")
        ax.set_title(title, loc="left", fontsize=8)
        _grid(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in DISSOCIATION], fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig11_dissociation")


# ------------------------------------------------------------------ Sekil 12
def fig_shells():
    """Temas sayiminda mesafe cozunurlugu ne eder, ve konformasyon bilgisi ne eder."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    a, b = axes

    ks = sorted(SHELLS)
    x = np.arange(len(ks))
    mu = np.array([SHELLS[k][2] for k in ks])
    sd = np.array([SHELLS[k][3] for k in ks])
    a.errorbar(x, mu, yerr=sd, fmt=MARKER["s1"], color=C["s1"], ms=5.5, mec="white",
               mew=0.8, lw=1.2, capsize=2.5, zorder=4)
    a.plot(x, mu, "-", color=C["s1"], lw=1.0, zorder=3)
    a.axhline(SHELLS[5][2], color=C["mute"], lw=0.8, ls=(0, (4, 2)), zorder=2)
    a.set_xticks(x)
    a.set_xticklabels([f"{k}\n({SHELLS[k][0]})" for k in ks], fontsize=7.5)
    a.set_xlabel("Distance shells (features)")
    a.set_ylabel("Test RMSE (pK)")
    a.set_ylim(1.455, 1.517)
    a.set_title("(a) Distance resolution has a cost", loc="left", fontsize=8)
    a.annotate("", xy=(0, SHELLS[0][2]), xytext=(0, SHELLS[5][2]),
               arrowprops=dict(arrowstyle="<->", color=C["s2"], lw=1.0))
    a.text(0.16, (SHELLS[0][2] + SHELLS[5][2]) / 2, "+0.024 pK\np = 4e-6",
           fontsize=6.8, color=C["s2"], va="center", fontweight="bold")
    _grid(a, "y")

    y = np.arange(len(DYN))[::-1]
    for yi, (lab, nf, d, lo, hi) in zip(y, DYN):
        b.plot([lo, hi], [yi, yi], color=C["s2"], lw=1.5, zorder=3, solid_capstyle="butt")
        for e in (lo, hi):
            b.plot([e, e], [yi - 0.11, yi + 0.11], color=C["s2"], lw=1.5, zorder=3)
        b.plot(d, yi, MARKER["s2"], color=C["s2"], ms=5.5, mec="white", mew=0.8, zorder=4)
        b.text(hi + 0.0016, yi, f"{d:+.4f}", va="center", fontsize=7, color=C["s2"])
        b.text(-0.0045, yi - 0.30, f"{nf} features", va="center", fontsize=6.3,
               color=C["mute"])
    b.axvline(0, color=C["ink2"], lw=0.9, zorder=2)
    b.set_yticks(y)
    b.set_yticklabels([d[0] for d in DYN], fontsize=7)
    b.set_xlim(-0.005, 0.030)
    b.set_ylim(-0.75, len(DYN) - 0.4)
    b.set_xlabel("Change in test RMSE (pK)")
    b.set_title("(b) Conformational and mobility information does not", loc="left",
                fontsize=8)
    b.text(0.029, -0.62, "worse", fontsize=6.5, color=C["s2"], ha="right")
    _grid(b, "x")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig12_shells")


# ------------------------------------------------------------------ Sekil 13
def fig_noise():
    """Geometrik sinyal, her modelin kendi tohum gurultusune karsi."""
    fig, ax = plt.subplots(figsize=(COL2 * 0.70, 2.2))
    y = np.arange(len(SEED_NOISE))[::-1]
    for yi, (lab, sd) in zip(y, SEED_NOISE):
        ratio = GEOM_SIGNAL / sd
        col = C["s1"] if ratio > 3 else C["s2"]
        ax.barh(yi, sd, 0.5, color=col, hatch="" if ratio > 3 else "///",
                edgecolor=C["ink2"], zorder=3)
        ax.text(sd + 0.0009, yi, f"{sd:.4f}   ({ratio:.1f}x)", va="center",
                fontsize=7.2, color=col, fontweight="bold")
    ax.axvline(GEOM_SIGNAL, color=C["ink2"], lw=1.1, ls=(0, (4, 2)), zorder=5)
    ax.text(GEOM_SIGNAL - 0.0008, len(SEED_NOISE) - 0.45,
            f"the whole geometric signal\n({GEOM_SIGNAL:.4f} pK)", fontsize=6.8,
            ha="right", color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in SEED_NOISE], fontsize=7.5)
    ax.set_xlim(0, 0.040)
    ax.set_ylim(-0.6, len(SEED_NOISE) - 0.15)
    ax.set_xlabel("Seed-to-seed standard deviation of test RMSE (pK)")
    _grid(ax, "x")
    save(fig, "fig13_noise")


if __name__ == "__main__":
    fig_design(); print("fig1 ok")
    fig_decomposition(); print("fig2 ok")
    fig_dose(); print("fig3 ok")
    fig_forcing(); print("fig4 ok")
    fig_controls(); print("fig5 ok")
    fig_pose(); print("fig6 ok")
    fig_resolution(); print("fig7 ok")
    fig_position(); print("fig8 ok")
    fig_calibration(); print("fig9 ok")
    fig_poscontrol(); print("fig10 ok")
    fig_dissociation(); print("fig11 ok")
    fig_shells(); print("fig12 ok")
    fig_noise(); print("fig13 ok")
    fig_toc(); print("TOC ok")
