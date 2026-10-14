"""
EnsAff makale sekilleri - ortak stil ve VERI blogu.


Kural 14: buradaki her sayi makaledeki bir tablodan gelir. Her girdinin
yanindaki yorum hangi tablodan geldigini soyler. Hicbir sayi uydurulmamistir,
hicbir sayi burada yeniden hesaplanmamistir.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --------------------------------------------------------------- stil
# ACS: tek sutun 3.33 in, cift sutun 7.0 in. Yazi tipi Arial/Helvetica.
# Liberation Sans, Arial ile ayni metriklere sahiptir.
COL1, COL2 = 3.33, 7.00

PALETTE = {          # dataviz skill referans paleti, validate_palette.js ile
    "s1": "#2a78d6", # slot 1 mavi   - degistirilmemis yapi modeli
    "s2": "#eb6834", # slot 2 turuncu- poz ayirt etmeye zorlanan kol
    "s3": "#1baf7a", # slot 3 su yesili - ucuncu seri (dogrudan etiketlenir)
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "mute": "#8a8982",
    "grid": "#dedcd4",
    "zero": "#b4b2a8",
}
# --pairs all ile her kapi gecti: CVD dE 9.2, normal gorus dE 24.0.
# Su yesili kontrast uyarisi aldigi icin kullanildigi her yerde dogrudan
# etiketlenir (rolyef kurali).

HATCH = {"s1": "", "s2": "///", "s3": "...."}   # gri tonlamada ayirt etmek icin
MARKER = {"s1": "o", "s2": "s", "s3": "^"}


def setup():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.7,
        "axes.edgecolor": PALETTE["ink2"],
        "xtick.color": PALETTE["ink2"],
        "ytick.color": PALETTE["ink2"],
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,      # gomulu TrueType, dergi istiyor
        "ps.fonttype": 42,
    })


def save(fig, name, outdir="/home/claude/paper/figs"):
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(outdir, f"{name}.png")


# --------------------------------------------------------------- VERI
# Tablo "decomp"
DECOMP = [("Per-residue representation\nand architecture", 0.063, "55%"),
          ("Pocket residue\nselection", 0.051, "45%"),
          ("Three-dimensional\ngeometry", -0.013, "no measurable\ncontribution")]
DECOMP_TOTAL = 0.114
DECOMP_BOOT = 0.115          # Tablo "primary", ayni buyuklugun yuvarlanmasi

# Tablo "probes": test bolmesi, degistirilmemis model, 5 tohum ortalamasi
#   ad -> (RMSE, Pearson, degisim d|mu| pK, r_base)
PROBES = {
    "none":            (1.532, 0.506, 0.0000, 1.0000),
    "translate 0.5 Å": (1.532, 0.506, 0.0044, 1.0000),
    "translate 1 Å":   (1.532, 0.506, 0.0076, 0.9999),
    "translate 2 Å":   (1.532, 0.506, 0.0129, 0.9998),
    "translate 4 Å":   (1.531, 0.506, 0.0206, 0.9997),
    "translate 8 Å":   (1.531, 0.506, 0.0264, 0.9996),
    "rotate":          (1.532, 0.506, 0.0113, 0.9999),
    "scale 1.5":       (1.531, 0.505, 0.0273, 0.9998),
    "permute residue": (1.532, 0.506, 0.0178, 0.9997),
    "permute atom":    (1.533, 0.506, 0.0080, 0.9999),
    "const 8 Å":       (1.536, 0.506, 0.0313, 0.9997),
}
PROBE_SD = 0.017   # Tablo "probes": tohumlar arasi RMSE standart sapmasi

# Tablo "forcing": degistirilmemis kol / zorlanan kol
FORCING_SENS = [
    ("Cost of constant distances (pK)",      0.004,  0.057, "14x"),
    ("Prediction change, constant (pK)",     0.031,  0.422, "13x"),
    ("Prediction change, 4 Å shift (pK)",    0.021,  0.093, "4.5x"),
    ("Prediction change, residues shuffled", 0.018,  0.074, "4.2x"),
]
FORCING_RBASE = (0.9997, 0.9855)
# Zorlanan kolun dogruluk farklari (eslesmis, degistirilmemis kola karsi)
FORCING_ACC = [
    ("Test (n = 4222)",      -0.013, -0.023, -0.004),
    ("Validation",            0.004, -0.013,  0.023),
    ("BDB2020+ (n = 115)",    0.050,  0.005,  0.091),
    ("mpro (n = 40)",         0.087,  0.030,  0.136),
]

# Tablo "controls": kol -> (toplulukRMSE, tohumRMSE, tohumSD, tohumSayisi)
CONTROLS = [
    ("Ligand-fingerprint baseline", 1.602, 1.610, 0.019, 5),
    ("Random pocket",               1.539, 1.582, 0.032, 5),
    ("Permuted distances",          1.476, 1.528, 0.054, 5),
    ("Full structure model",        1.488, 1.532, 0.017, 5),
    ("Constant distances",          1.475, 1.523, 0.050, 5),
]

# Tablo "position": kcal/mol
POSITION = [
    ("AutoDock Vina",        2.85, 2.56,  None,  False),
    ("DeepDTA",              1.34, 2.29,  0.04,  False),
    ("RF-Score",             1.89, 2.10,  0.003, False),
    ("IGN",                  1.82, 2.16,  0.13,  False),
    ("Ligand baseline",      None, 2.19,  None,  True),
    ("Structure model",      None, 2.03,  None,  True),
]

# Bolum 6.7: poz ayirt etme
# Sans duzeyi: olcut "her kristal poz her celdiriciden yuksek" oldugu icin
# 1/C(n_real + 4, n_real). Tek pozda 0.200, cok pozda daha dusuk; 0.200 UST SINIR.
POSE = {"chance": 0.200, "predicted": 0.80,
        "forced": 0.559, "wide": 0.324,
        "wide_seeds": [0.281, 0.351, 0.338]}

# Ana etki ve dis kumelerin cozunurlugu (Bolum 6.1)
#   ad -> (nokta tahmini, alt, ust, n)
RESOLUTION = [
    ("Test split",  -0.115, -0.136, -0.093, 4222),
    ("BDB2020+",    -0.129, -0.276,  0.022,  115),
    ("mpro",         0.055, -0.018,  0.128,   40),
]

# Tablo "primary"
PRIMARY = {"rmse": (1.602, 1.488, -0.115, -0.136, -0.093),
           "pearson": (0.432, 0.529, 0.097, 0.080, 0.114),
           "spearman": (0.415, 0.502, 0.086, 0.070, 0.102)}

# Sonda kalibrasyonu (sentetik, 2 tohum, sandbox). Bolum 6.8.
#   rejim -> (gercek kol RMSE, sd, karistirilmis kol RMSE, sd,
#             sifirdan egitim bedeli, cikarim sondasi bedeli, r_temel)
CALIB = {
    "Geometry only":    (0.576, 0.013, 0.990, 0.050, +0.415, +1.288, 0.099),
    "Composition only": (0.205, 0.001, 0.203, 0.001, -0.002, +0.473, 0.558),
}


# Pozitif kontrol (v14). Temas sayimi modelleri geometriyi KESIN kullanir.
#   model -> {sonda: test RMSE}
POSCTRL = {
    "Contact counts, forest":   {"none": 1.462, "const": 1.973, "t4": 1.865,
                                 "perm": 1.482, "rot": 1.502, "t1": 1.481},
    "Contact counts, boosting": {"none": 1.491, "const": 1.601, "t4": 1.916,
                                 "perm": 1.563, "rot": 1.559, "t1": 1.521},
    "Structure model":          {"none": 1.532, "const": 1.536, "t4": 1.531,
                                 "perm": 1.532, "rot": 1.532, "t1": 1.532},
}
# BDB2020+ ayni sira
POSCTRL_BDB = {
    "Contact counts, forest":   {"none": 1.130, "const": 2.349},
    "Contact counts, boosting": {"none": 1.184, "const": 1.531},
    "Structure model":          {"none": 1.168, "const": 1.172},
}

# Ikinci mimari (voksel 3B-CNN), v15.
VOXEL = {"vx": (1.5308, 1.5698, 0.0249, 5),      # topluluk, tohum ort, sd, n
         "vx_scr": (1.5311, 1.5571, 0.0179, 3),
         "vx_rand": (1.5290, 1.5526, 0.0113, 3)}
VOXEL_PROBES = {                                  # sonda -> (RMSE bedeli, d|mu|, r_temel, temiz mi)
    "scramble":      (+1.385, 2.477, 0.596, True),
    "rotate":        (-0.004, 0.366, 0.930, True),
    "permute_res":   (-0.004, 0.249, 0.979, True),
    "permute_atom":  (-0.008, 0.099, 0.994, True),
    "translate 8 Å": (+1.045, 1.874, 0.325, False),
    "scale 1.5":     (+1.480, 2.537, -0.103, False),
}
# Iki aletin uc mimaride verdigi cevap: (cikarim sondasi bedeli, sifirdan egitim bedeli)
# Temas sayiminin "sifirdan egitim" kolu v16'da kabuk sayisi sifira indirilerek kosuldu.
DISSOCIATION = [("Contact counts,\nforest",        +0.511, +0.0238),
                ("Voxel 3D-CNN",                   +1.385, +0.0004),
                ("Equivariant graph",              +0.052, -0.0017),
                ("Pair representation",            +0.004, -0.013)]

# v16: temas sayimi uzerinde kabuk merdiveni (sifirdan egitim, 5 tohum)
#   kabuk -> (ozellik, topluluk RMSE, tohum ort, tohum sd, 5 kabuga karsi fark, alt, ust)
SHELLS = {
    0: (25,  1.4844, 1.4854, 0.0011, +0.0237, +0.0098, +0.0373),
    1: (25,  1.4817, 1.4827, 0.0016, +0.0210, +0.0106, +0.0312),
    2: (50,  1.5070, 1.5080, 0.0016, +0.0464, +0.0376, +0.0559),
    3: (75,  1.4867, 1.4878, 0.0033, +0.0261, +0.0190, +0.0335),
    5: (125, 1.4606, 1.4618, 0.0026,  0.0,     0.0,     0.0),
}
# v16: konformasyon toplulugu ve hareketlilik (referansa karsi, pozitif = daha kotu)
DYN = [("Mean over sibling\nconformations", 125, +0.0073, +0.0032, +0.0113),
       ("Mean and spread over\nsibling conformations", 250, +0.0183, +0.0135, +0.0231),
       ("Counts weighted by\nensemble mobility", 250, +0.0126, +0.0093, +0.0159)]
DYN_K = {"mean_K": 2.72, "frac_multi": 0.55, "n_rmsf": 1852, "n_test": 4222}

# v17: E(n)-degismez graf agi (ucuncu mimari)
EQUIV = {"eq": (1.5536, 0.0183, 5), "eq_const": (1.5519, 0.0219, 3),
         "eq_scr": (1.5139, 0.0121, 3)}
EQUIV_CMP = {"eq_const": (-0.0017, 0.92), "eq_scr": (-0.0397, 0.011)}
EQUIV_PROBES = {"none": (1.5536, 0.4616, 0.0000, 1.0000),
                "scramble": (1.6054, 0.4333, 0.6453, 0.7150),
                "rotate": (1.5425, 0.4629, 0.4142, 0.8701),
                "permute_res": (1.5751, 0.4421, 0.3493, 0.8982),
                "permute_atom": (1.5518, 0.4610, 0.1322, 0.9837),
                "translate 8 Å": (2.6500, 0.2283, 2.0506, 0.3866),
                "scale 1.5": (2.2692, 0.3773, 1.6647, 0.7490)}
# Geometrik sinyalin buyuklugu, her modelin kendi tohum gurultusune karsi.
GEOM_SIGNAL = 0.0238          # temas sayiminda sifirdan egitim bedeli (v16)
SEED_NOISE = [("Contact counts, forest", 0.0026),
              ("Pair representation", 0.0170),
              ("Equivariant graph", 0.0183),
              ("Voxel 3D-CNN", 0.0249)]
