"""
================================================================================
EDA COMPLETO — HAM10000 Skin Cancer Dataset
TFM: Clasificación automática de enfermedades dermatológicas mediante IA
Máster en Análisis y Visualización de Datos Masivos — UNIR
================================================================================
Estructura del script:
  1.  Carga y auditoría inicial
  2.  Limpieza: estandarización, valores inconsistentes
  3.  Imputación: edad → mediana por clase (dx); resto sin imputar
  4.  Ingeniería de variables auxiliares
  5.  Estadística descriptiva univariante
  6.  Análisis bivariante y multivariante
  7.  Correlaciones (Cramér's V entre categóricas; Point-Biserial y ANOVA edad)
  8.  10 figuras profesionales guardadas en D:/claude/Resultados/
  9.  Exportación de todos los resultados a Excel (múltiples hojas)
================================================================================
Dependencias:  pandas, numpy, matplotlib, seaborn, scipy, openpyxl
Instalación:   pip install pandas numpy matplotlib seaborn scipy openpyxl
================================================================================
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # sin pantalla gráfica; cambia a "TkAgg" si quieres ventanas

# ── 1. RUTAS ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\TFM\HAM10000"
OUT_DIR    = r"D:\TFM\Resultados"
DATA_PATH  = os.path.join(BASE_DIR, "HAM10000_metadata_balanceado.csv")
EXCEL_PATH = os.path.join(OUT_DIR,  "eda_resultados_HAM10000.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 2. SISTEMA DE DISEÑO GLOBAL ───────────────────────────────────────────────
DX_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

DX_LABELS = {
    "nv":    "Nevus melanocítico",
    "mel":   "Melanoma",
    "bkl":   "Queratosis benigna",
    "bcc":   "Carcinoma basocelular",
    "akiec": "Queratosis actínica",
    "vasc":  "Lesión vascular",
    "df":    "Dermatofibroma",
}

# Paleta accesible y consistente en todos los gráficos
PALETTE = {
    "nv":    "#4C9BE8",
    "mel":   "#D94F4F",
    "bkl":   "#5BAD72",
    "bcc":   "#E8933A",
    "akiec": "#9B6DC9",
    "vasc":  "#E86FAC",
    "df":    "#6BBFBF",
}

MALIGNOS = {"mel", "bcc", "akiec"}

DTYPE_LABELS = {
    "histo":     "Histopatología",
    "follow_up": "Seguimiento clínico",
    "consensus": "Consenso experto",
    "confocal":  "Microscopía confocal",
}

# Estilo matplotlib global
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
})
GRID_KWARGS = dict(alpha=0.35, linewidth=0.6, linestyle="--", color="#aaaaaa")

FIG_DPI = 150

def savefig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Figura guardada: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA Y AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 1 — CARGA Y AUDITORÍA INICIAL")
print("═"*62)

df_raw = pd.read_csv(DATA_PATH)

print(f"\n  Dimensiones brutas : {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
print(f"  Columnas           : {list(df_raw.columns)}")
print(f"\n  Tipos de datos:\n{df_raw.dtypes.to_string()}")
print(f"\n  Primeras 5 filas:\n{df_raw.head().to_string()}")
print(f"\n  Nulos por columna (bruto):\n{df_raw.isnull().sum().to_string()}")
print(f"\n  Duplicados exactos (todas las columnas): {df_raw.duplicated().sum()}")
print(f"  Lesiones únicas (lesion_id) : {df_raw['lesion_id'].nunique():,}")
print(f"  Imágenes únicas (image_id)  : {df_raw['image_id'].nunique():,}")

imgs_per_lesion = df_raw.groupby("lesion_id")["image_id"].count()
print(f"  Lesiones con >1 imagen      : {(imgs_per_lesion > 1).sum():,}")
print(f"\n  Valores únicos por columna categórica:")
for col in ["dx", "dx_type", "sex", "localization"]:
    print(f"    {col:15s}: {sorted(df_raw[col].dropna().unique())}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — LIMPIEZA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 2 — LIMPIEZA")
print("═"*62)

df = df_raw.copy()

# Estandarizar cadenas (strip + lower)
for col in ["dx", "dx_type", "sex", "localization"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Reemplazar cadenas vacías o literales "nan"/"unknown" por NaN real
df.replace({"nan": np.nan, "unknown": np.nan, "": np.nan}, inplace=True)

# Registrar nulos tras estandarización
nulos_post_clean = df.isnull().sum()
print(f"\n  Nulos tras estandarización:\n{nulos_post_clean.to_string()}")

# Auditar valores de edad fuera de rango clínico plausible
n_edad_neg  = (df["age"] < 0).sum()
n_edad_alto = (df["age"] > 110).sum()
print(f"\n  Edades negativas    : {n_edad_neg}")
print(f"  Edades > 110 años   : {n_edad_alto}")
if n_edad_neg > 0 or n_edad_alto > 0:
    df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan
    print("  → Convertidas a NaN las edades fuera de rango")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — IMPUTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 3 — IMPUTACIÓN DE EDAD (mediana por clase dx)")
print("═"*62)

age_nulos_antes = df["age"].isnull().sum()
median_por_dx = df.groupby("dx")["age"].median()

print(f"\n  Nulos en 'age' antes de imputar : {age_nulos_antes}")
print(f"\n  Mediana de edad por clase dx:")
for dx in DX_ORDER:
    med = median_por_dx.get(dx, np.nan)
    print(f"    {dx:8s} ({DX_LABELS[dx]:30s}): {med:.1f} años")

df["age"] = df.apply(
    lambda row: median_por_dx[row["dx"]] if pd.isnull(row["age"]) else row["age"],
    axis=1,
)

age_nulos_despues = df["age"].isnull().sum()
print(f"\n  Nulos en 'age' después de imputar: {age_nulos_despues}")
print(f"\n  Nulos finales por columna (sex, localization NO imputadas):")
print(df.isnull().sum().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — VARIABLES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 4 — VARIABLES AUXILIARES")
print("═"*62)

# Malignidad binaria
df["malignidad"] = df["dx"].apply(
    lambda x: "Maligno" if x in MALIGNOS else "Benigno"
)

# Grupos de edad (quintiles clínicos)
bins_edad   = [0, 20, 35, 50, 65, 80, 110]
labels_edad = ["<20", "20–35", "35–50", "50–65", "65–80", ">80"]
df["age_group"] = pd.cut(df["age"], bins=bins_edad,
                          labels=labels_edad, right=True)

# Nº imágenes por lesión (contexto de duplicados)
df = df.merge(
    imgs_per_lesion.rename("imgs_x_lesion"),
    on="lesion_id", how="left"
)

print(f"  Distribución malignidad:")
print(df["malignidad"].value_counts().to_string())
print(f"\n  Distribución grupos de edad:")
print(df["age_group"].value_counts().sort_index().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — ESTADÍSTICA DESCRIPTIVA UNIVARIANTE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 5 — DESCRIPTIVA UNIVARIANTE")
print("═"*62)

print(f"\n  Edad (describe completo):")
print(df["age"].describe().round(2).to_string())

print(f"\n  Distribución dx:")
print(df["dx"].value_counts().to_string())

print(f"\n  Distribución dx_type:")
print(df["dx_type"].value_counts().to_string())

print(f"\n  Distribución sex:")
print(df["sex"].value_counts().to_string())

print(f"\n  Distribución localization (top 10):")
print(df["localization"].value_counts().head(10).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 6 — CORRELACIONES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 6 — CORRELACIONES")
print("═"*62)

def cramers_v(x, y):
    """Cramér's V corregido (bias-corrected) entre dos series categóricas."""
    ct   = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n    = ct.values.sum()
    r, k = ct.shape
    phi2c = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
    rc    = r - (r - 1) ** 2 / (n - 1)
    kc    = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return np.sqrt(phi2c / denom) if denom > 0 else 0.0

cat_cols = ["dx", "dx_type", "sex", "localization", "malignidad", "age_group"]
cv_matrix = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))),
                          index=cat_cols, columns=cat_cols)

for c1, c2 in itertools.combinations(cat_cols, 2):
    sub = df[[c1, c2]].dropna()
    v   = cramers_v(sub[c1], sub[c2])
    cv_matrix.loc[c1, c2] = v
    cv_matrix.loc[c2, c1] = v
np.fill_diagonal(cv_matrix.values, 1.0)

print("\n  Cramér's V entre variables categóricas:")
print(cv_matrix.round(3).to_string())

# ANOVA / Kruskal-Wallis: edad ~ dx
grupos_edad = [df[df["dx"] == d]["age"].dropna().values for d in DX_ORDER]
stat_kw, p_kw = kruskal(*grupos_edad)
print(f"\n  Kruskal-Wallis (edad ~ dx):  H = {stat_kw:.2f}, p = {p_kw:.4e}")

# Point-Biserial: edad ~ malignidad
mal_mask  = df["malignidad"] == "Maligno"
bpb, ppb  = stats.pointbiserialr(mal_mask.astype(int), df.loc[mal_mask.index, "age"])
print(f"  Point-Biserial (edad ~ malignidad):  r = {bpb:.4f}, p = {ppb:.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 7 — FIGURAS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 7 — GENERACIÓN DE FIGURAS")
print("═"*62)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 01 — Distribución de clases: barras + pie
# ─────────────────────────────────────────────────────────────────────────────
counts_dx = df["dx"].value_counts().reindex(DX_ORDER)

fig, axes = plt.subplots(2, 1, figsize=(8, 11))
fig.patch.set_facecolor("white")

# Barras
bars = axes[0].bar(
    [DX_LABELS[d] for d in DX_ORDER],
    counts_dx.values,
    color=[PALETTE[d] for d in DX_ORDER],
    edgecolor="white", linewidth=0.9, zorder=3,
)
axes[0].yaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
for bar, val in zip(bars, counts_dx.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 40,
                 f"{val:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
axes[0].set_title("Distribución absoluta de clases diagnósticas")
axes[0].set_ylabel("Número de imágenes")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=40)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Pie
pcts = counts_dx / counts_dx.sum() * 100
wedges, texts, autotexts = axes[1].pie(
    pcts.values,
    labels=None,
    colors=[PALETTE[d] for d in DX_ORDER],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.3},
    pctdistance=0.78,
)
for at in autotexts:
    at.set_fontsize(8)
axes[1].legend(
    handles=[mpatches.Patch(color=PALETTE[d], label=DX_LABELS[d]) for d in DX_ORDER],
    loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5,
)
axes[1].set_title("Distribución proporcional de clases diagnósticas")
fig.suptitle("HAM10000 — Desbalance de clases (N = {:,})".format(len(df)),
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig01_distribucion_clases.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 02 — Distribución de edad: histograma + KDE + boxplots por clase
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 15))
fig.patch.set_facecolor("white")

# Histograma global
axes[0].hist(df["age"], bins=30, color="#4C9BE8", edgecolor="white",
             linewidth=0.7, alpha=0.80, density=True, zorder=3)
df["age"].plot.kde(ax=axes[0], color="#D94F4F", linewidth=2.2, zorder=4)
axes[0].axvline(df["age"].mean(),   color="#E8933A", linestyle="--",
                linewidth=1.7, label=f"Media: {df['age'].mean():.1f} a")
axes[0].axvline(df["age"].median(), color="#5BAD72", linestyle=":",
                linewidth=1.7, label=f"Mediana: {df['age'].median():.1f} a")
axes[0].yaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
axes[0].set_title("Distribución global de edad")
axes[0].set_xlabel("Edad (años)")
axes[0].set_ylabel("Densidad")
axes[0].legend()

# KDE por clase
for dx in DX_ORDER:
    subset = df[df["dx"] == dx]["age"].dropna()
    subset.plot.kde(ax=axes[1], label=DX_LABELS[dx],
                    color=PALETTE[dx], linewidth=2.0)
axes[1].yaxis.grid(**GRID_KWARGS)
axes[1].set_axisbelow(True)
axes[1].set_title("Densidad de edad por clase dx")
axes[1].set_xlabel("Edad (años)")
axes[1].set_ylabel("Densidad")
axes[1].set_xlim(0, 110)
axes[1].legend(fontsize=7.5)

# Boxplot por clase
bp = axes[2].boxplot(
    [df[df["dx"] == d]["age"].dropna().values for d in DX_ORDER],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.2),
    flierprops=dict(marker="o", markersize=2.5, alpha=0.35),
    whiskerprops=dict(linewidth=1.1),
    capprops=dict(linewidth=1.1),
)
for patch, dx in zip(bp["boxes"], DX_ORDER):
    patch.set_facecolor(PALETTE[dx])
    patch.set_alpha(0.85)
axes[2].set_xticks(range(1, len(DX_ORDER) + 1))
axes[2].set_xticklabels(DX_ORDER, rotation=30)
axes[2].yaxis.grid(**GRID_KWARGS)
axes[2].set_axisbelow(True)
axes[2].set_title("Boxplot de edad por clase dx")
axes[2].set_ylabel("Edad (años)")

fig.suptitle("HAM10000 — Análisis de la variable Edad", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig02_distribucion_edad.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 03 — Sexo: distribución global y por clase
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 10))
fig.patch.set_facecolor("white")

sex_colors = {"male": "#4C9BE8", "female": "#E86FAC"}
sex_counts = df["sex"].value_counts()

# Barras sexo global
b = axes[0].bar(
    sex_counts.index.str.capitalize(),
    sex_counts.values,
    color=[sex_colors.get(s, "#aaa") for s in sex_counts.index],
    edgecolor="white", linewidth=0.9, width=0.5, zorder=3,
)
axes[0].yaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
for bar, val in zip(b, sex_counts.values):
    pct = val / sex_counts.sum() * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[0].set_title("Distribución global de sexo")
axes[0].set_ylabel("Número de imágenes")

# Barras agrupadas sexo por clase
sex_dx = df.groupby(["dx", "sex"]).size().unstack(fill_value=0).reindex(DX_ORDER)
x      = np.arange(len(DX_ORDER))
width  = 0.35
sexes  = [s for s in ["male", "female"] if s in sex_dx.columns]
for i, sex in enumerate(sexes):
    axes[1].bar(x + i * width, sex_dx[sex].values, width=width,
                label=sex.capitalize(), color=sex_colors[sex],
                edgecolor="white", linewidth=0.7, zorder=3)
axes[1].set_xticks(x + width * (len(sexes) - 1) / 2)
axes[1].set_xticklabels(DX_ORDER, rotation=30)
axes[1].yaxis.grid(**GRID_KWARGS)
axes[1].set_axisbelow(True)
axes[1].set_title("Distribución de sexo por clase dx")
axes[1].set_ylabel("Número de imágenes")
axes[1].legend(title="Sexo")

fig.suptitle("HAM10000 — Variable Sexo", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig03_distribucion_sexo.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 04 — Localización anatómica (top 12) y por clase
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 13))
fig.patch.set_facecolor("white")

# Top 12 localizaciones
loc_counts = df["localization"].value_counts().head(12)
palette_loc = sns.color_palette("Blues_r", len(loc_counts))
axes[0].barh(loc_counts.index[::-1], loc_counts.values[::-1],
             color=palette_loc[::-1], edgecolor="white", linewidth=0.7, zorder=3)
axes[0].xaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
for i, val in enumerate(loc_counts.values[::-1]):
    axes[0].text(val + 20, i, f"{val:,}", va="center", fontsize=8)
axes[0].set_title("Top 12 localizaciones anatómicas")
axes[0].set_xlabel("Número de imágenes")

# Heatmap localización × dx
top8_locs = df["localization"].value_counts().head(8).index
ct_loc_dx = pd.crosstab(
    df[df["localization"].isin(top8_locs)]["localization"],
    df[df["localization"].isin(top8_locs)]["dx"]
).reindex(columns=DX_ORDER, fill_value=0)

sns.heatmap(ct_loc_dx, ax=axes[1], annot=True, fmt="d",
            cmap="YlOrRd", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Conteo"})
axes[1].set_title("Top 8 localizaciones × Clase dx (conteos)")
axes[1].set_xlabel("Clase dx")
axes[1].set_ylabel("Localización anatómica")
axes[1].tick_params(axis="y", rotation=0)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

fig.suptitle("HAM10000 — Localización anatómica de las lesiones",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig04_localizacion.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 05 — Método diagnóstico (dx_type)
# ─────────────────────────────────────────────────────────────────────────────
dtype_counts = df["dx_type"].value_counts()
ct_dx_dtype  = pd.crosstab(df["dx"], df["dx_type"]).reindex(DX_ORDER)
colors_dtype = ["#4C9BE8", "#E8933A", "#5BAD72", "#9B6DC9"]

fig, axes = plt.subplots(2, 1, figsize=(8, 10))
fig.patch.set_facecolor("white")

# Barras globales
b = axes[0].bar(
    [DTYPE_LABELS.get(d, d) for d in dtype_counts.index],
    dtype_counts.values,
    color=colors_dtype[:len(dtype_counts)],
    edgecolor="white", linewidth=0.9, zorder=3,
)
axes[0].yaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
for bar, val in zip(b, dtype_counts.values):
    pct = val / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")
axes[0].set_title("Método de diagnóstico (dx_type)")
axes[0].set_ylabel("Número de imágenes")
axes[0].tick_params(axis="x", rotation=15)

# Heatmap dx × dx_type
sns.heatmap(ct_dx_dtype, ax=axes[1], annot=True, fmt="d",
            cmap="Blues", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Conteo"})
axes[1].set_title("Clase dx × Método diagnóstico")
axes[1].set_xlabel("Método diagnóstico")
axes[1].set_ylabel("Clase dx")
axes[1].tick_params(axis="y", rotation=0)
axes[1].tick_params(axis="x", rotation=15)

fig.suptitle("HAM10000 — Calidad y método del diagnóstico",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig05_dx_type.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 06 — Malignidad: distribución y edad
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 15))
fig.patch.set_facecolor("white")
mal_colors = {"Benigno": "#5BAD72", "Maligno": "#D94F4F"}

# Barras malignidad
mal_counts = df["malignidad"].value_counts()
b = axes[0].bar(mal_counts.index, mal_counts.values,
                color=[mal_colors[m] for m in mal_counts.index],
                edgecolor="white", width=0.45, zorder=3)
axes[0].yaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
for bar, val in zip(b, mal_counts.values):
    pct = val / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[0].set_title("Proporción maligno vs benigno")
axes[0].set_ylabel("Número de imágenes")

# Boxplot edad ~ malignidad
data_mal = [df[df["malignidad"] == m]["age"].dropna().values
            for m in ["Benigno", "Maligno"]]
bp = axes[1].boxplot(data_mal, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2.2),
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.35))
for patch, m in zip(bp["boxes"], ["Benigno", "Maligno"]):
    patch.set_facecolor(mal_colors[m])
    patch.set_alpha(0.85)
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["Benigno", "Maligno"])
axes[1].yaxis.grid(**GRID_KWARGS)
axes[1].set_axisbelow(True)
axes[1].set_title("Distribución de edad: maligno vs benigno")
axes[1].set_ylabel("Edad (años)")

# Violín edad ~ clase
vp_data = [df[df["dx"] == d]["age"].dropna().values for d in DX_ORDER]
parts   = axes[2].violinplot(vp_data, positions=range(1, len(DX_ORDER) + 1),
                              showmedians=True, showextrema=True)
for i, (body, dx) in enumerate(zip(parts["bodies"], DX_ORDER)):
    body.set_facecolor(PALETTE[dx])
    body.set_alpha(0.75)
parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)
axes[2].set_xticks(range(1, len(DX_ORDER) + 1))
axes[2].set_xticklabels(DX_ORDER, rotation=30)
axes[2].yaxis.grid(**GRID_KWARGS)
axes[2].set_axisbelow(True)
axes[2].set_title("Violín de edad por clase dx")
axes[2].set_ylabel("Edad (años)")

fig.suptitle("HAM10000 — Edad y malignidad", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig06_malignidad_edad.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 07 — Grupos de edad × clase dx (barras agrupadas)
# ─────────────────────────────────────────────────────────────────────────────
age_dx_ct = (df.groupby(["age_group", "dx"]).size()
               .unstack(fill_value=0)
               .reindex(columns=DX_ORDER))

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("white")
x     = np.arange(len(age_dx_ct.index))
w     = 0.11
for i, dx in enumerate(DX_ORDER):
    ax.bar(x + i * w, age_dx_ct[dx].values, width=w,
           label=DX_LABELS[dx], color=PALETTE[dx],
           edgecolor="white", linewidth=0.6, zorder=3)
ax.set_xticks(x + w * (len(DX_ORDER) - 1) / 2)
ax.set_xticklabels(age_dx_ct.index, rotation=0)
ax.yaxis.grid(**GRID_KWARGS)
ax.set_axisbelow(True)
ax.set_title("Distribución de diagnósticos por grupo de edad")
ax.set_ylabel("Número de imágenes")
ax.set_xlabel("Grupo de edad (años)")
ax.legend(title="Diagnóstico", bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=8)
plt.tight_layout()
savefig(fig, "fig07_grupos_edad_dx.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 08 — Correlación Cramér's V (triángulo inferior)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("white")

mask_upper = np.triu(np.ones_like(cv_matrix, dtype=bool), k=1)
sns.heatmap(cv_matrix, ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            mask=mask_upper,
            cbar_kws={"label": "Cramér's V (0 = sin asociación, 1 = asociación perfecta)"},
            annot_kws={"size": 10})
ax.set_title("Matriz de correlación — Cramér's V entre variables categóricas",
             fontsize=12)
ax.tick_params(axis="both", labelsize=9)
plt.tight_layout()
savefig(fig, "fig08_correlacion_cramers_v.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 09 — Multiplicidad de imágenes por lesión
# ─────────────────────────────────────────────────────────────────────────────
vc_imgs = imgs_per_lesion.value_counts().sort_index()

fig, axes = plt.subplots(2, 1, figsize=(7, 10))
fig.patch.set_facecolor("white")

# Histograma
axes[0].bar(vc_imgs.index, vc_imgs.values, color="#4C9BE8",
            edgecolor="white", linewidth=0.8, zorder=3)
axes[0].yaxis.grid(**GRID_KWARGS)
axes[0].set_axisbelow(True)
for x_val, y_val in zip(vc_imgs.index, vc_imgs.values):
    axes[0].text(x_val, y_val + 15, f"{y_val:,}", ha="center",
                 fontsize=9, fontweight="bold")
axes[0].set_title("Distribución de imágenes por lesión")
axes[0].set_xlabel("Nº imágenes / lesión")
axes[0].set_ylabel("Nº lesiones")

# Por clase
imgs_dx = (df.groupby(["lesion_id", "dx"])["image_id"]
             .count()
             .reset_index()
             .rename(columns={"image_id": "n_imgs"}))
bp = axes[1].boxplot(
    [imgs_dx[imgs_dx["dx"] == d]["n_imgs"].values for d in DX_ORDER],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.2),
    flierprops=dict(marker="o", markersize=2.5, alpha=0.35),
)
for patch, dx in zip(bp["boxes"], DX_ORDER):
    patch.set_facecolor(PALETTE[dx])
    patch.set_alpha(0.85)
axes[1].set_xticks(range(1, len(DX_ORDER) + 1))
axes[1].set_xticklabels(DX_ORDER, rotation=30)
axes[1].yaxis.grid(**GRID_KWARGS)
axes[1].set_axisbelow(True)
axes[1].set_title("Imágenes por lesión, por clase dx")
axes[1].set_ylabel("Nº imágenes / lesión")

fig.suptitle("HAM10000 — Multiplicidad de imágenes por lesión",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig09_imagenes_por_lesion.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Sexo × Localización (top 8) — barras apiladas 100 %
# ─────────────────────────────────────────────────────────────────────────────
top8_locs2 = df["localization"].value_counts().head(8).index
ct_sex_loc = pd.crosstab(
    df[df["localization"].isin(top8_locs2)]["localization"],
    df[df["localization"].isin(top8_locs2)]["sex"]
)
ct_sex_loc_pct = ct_sex_loc.div(ct_sex_loc.sum(axis=1), axis=0) * 100
sexes_in = [s for s in ["male", "female"] if s in ct_sex_loc_pct.columns]

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("white")
bottom = np.zeros(len(ct_sex_loc_pct))
for sex in sexes_in:
    ax.bar(ct_sex_loc_pct.index, ct_sex_loc_pct[sex].values,
           bottom=bottom, label=sex.capitalize(),
           color=sex_colors[sex], edgecolor="white", linewidth=0.7, zorder=3)
    bottom += ct_sex_loc_pct[sex].values

ax.axhline(50, color="#555", linestyle="--", linewidth=1.0, alpha=0.6)
ax.yaxis.grid(**GRID_KWARGS)
ax.set_axisbelow(True)
ax.set_title("Distribución de sexo por localización anatómica (top 8) — 100 %")
ax.set_ylabel("Porcentaje (%)")
ax.set_xlabel("Localización anatómica")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Sexo")
ax.set_ylim(0, 115)
plt.tight_layout()
savefig(fig, "fig10_sexo_localizacion_apilado.png")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 8 — EXPORTACIÓN EXCEL (12 hojas)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  BLOQUE 8 — EXPORTACIÓN A EXCEL")
print("═"*62)

with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:

    # Hoja 1: Resumen general
    pd.DataFrame({
        "Métrica": [
            "Total imágenes", "Total lesiones únicas",
            "Lesiones con >1 imagen", "Columnas originales",
            "Nulos age (antes imputación)", "Nulos age (después imputación)",
            "Nulos sex (sin imputar)", "Nulos localization (sin imputar)",
            "Clases diagnósticas (dx)",
        ],
        "Valor": [
            df.shape[0], df["lesion_id"].nunique(),
            int((imgs_per_lesion > 1).sum()), df_raw.shape[1],
            age_nulos_antes, age_nulos_despues,
            int(df["sex"].isnull().sum()),
            int(df["localization"].isnull().sum()),
            df["dx"].nunique(),
        ],
    }).to_excel(writer, sheet_name="01_Resumen_General", index=False)

    # Hoja 2: Distribución clases
    dist_dx = df["dx"].value_counts().reset_index()
    dist_dx.columns = ["dx", "n_imagenes"]
    dist_dx["porcentaje_%"] = (dist_dx["n_imagenes"] / len(df) * 100).round(2)
    dist_dx["nombre_largo"] = dist_dx["dx"].map(DX_LABELS)
    dist_dx["malignidad"]   = dist_dx["dx"].apply(
        lambda x: "Maligno" if x in MALIGNOS else "Benigno")
    dist_dx.to_excel(writer, sheet_name="02_Distribucion_Clases", index=False)

    # Hoja 3: Estadística descriptiva edad
    desc_global = df["age"].describe().rename("Global")
    desc_por_dx = df.groupby("dx")["age"].describe()
    pd.concat([desc_global.to_frame().T,
               desc_por_dx]).round(2).to_excel(writer, sheet_name="03_Estadist_Edad")

    # Hoja 4: Mediana edad por clase (imputación)
    med_df = median_por_dx.reset_index()
    med_df.columns = ["dx", "mediana_edad_imputada"]
    med_df["nombre_largo"] = med_df["dx"].map(DX_LABELS)
    med_df.to_excel(writer, sheet_name="04_Mediana_Edad_Imputac", index=False)

    # Hoja 5: Distribución sexo
    dist_sex = df["sex"].value_counts(dropna=False).reset_index()
    dist_sex.columns = ["sexo", "conteo"]
    dist_sex["porcentaje_%"] = (dist_sex["conteo"] / len(df) * 100).round(2)
    dist_sex.to_excel(writer, sheet_name="05_Distribucion_Sexo", index=False)

    # Hoja 6: Distribución localización
    dist_loc_full = df["localization"].value_counts(dropna=False).reset_index()
    dist_loc_full.columns = ["localizacion", "conteo"]
    dist_loc_full["porcentaje_%"] = (dist_loc_full["conteo"] / len(df) * 100).round(2)
    dist_loc_full.to_excel(writer, sheet_name="06_Distribucion_Localizac", index=False)

    # Hoja 7: Distribución dx_type
    dist_dt = df["dx_type"].value_counts(dropna=False).reset_index()
    dist_dt.columns = ["dx_type", "conteo"]
    dist_dt["porcentaje_%"] = (dist_dt["conteo"] / len(df) * 100).round(2)
    dist_dt["nombre_largo"] = dist_dt["dx_type"].map(DTYPE_LABELS)
    dist_dt.to_excel(writer, sheet_name="07_Distribucion_DxType", index=False)

    # Hoja 8: Tabla cruzada dx × dx_type
    ct_dx_dtype.to_excel(writer, sheet_name="08_CrossTab_Dx_DxType")

    # Hoja 9: Tabla cruzada localización × dx (top 12)
    top12_locs = df["localization"].value_counts().head(12).index
    ct_loc_dx_full = pd.crosstab(
        df[df["localization"].isin(top12_locs)]["localization"],
        df[df["localization"].isin(top12_locs)]["dx"]
    ).reindex(columns=DX_ORDER, fill_value=0)
    ct_loc_dx_full.to_excel(writer, sheet_name="09_CrossTab_Loc_Dx")

    # Hoja 10: Cramér's V
    cv_matrix.round(4).to_excel(writer, sheet_name="10_Cramers_V")

    # Hoja 11: Grupos de edad × dx
    age_dx_ct.to_excel(writer, sheet_name="11_GruposEdad_Dx")

    # Hoja 12: Dataset limpio completo
    df.to_excel(writer, sheet_name="12_Dataset_Limpio", index=False)

print(f"\n  Excel guardado en: {EXCEL_PATH}")
print(f"  Hojas generadas   : 12")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  RESUMEN FINAL DEL EDA")
print("═"*62)
print(f"  Total imágenes analizadas : {df.shape[0]:,}")
print(f"  Lesiones únicas           : {df['lesion_id'].nunique():,}")
print(f"  Variables tras limpieza   : {df.shape[1]}")
print(f"  Nulos restantes (edad)    : {df['age'].isnull().sum()}")
print(f"  Nulos restantes (sex)     : {df['sex'].isnull().sum()}")
print(f"  Nulos restantes (local.)  : {df['localization'].isnull().sum()}")
print()
print(f"  {'Clase':<8} {'N':>6}  {'%':>6}  Malignidad")
print(f"  {'-'*40}")
for dx in DX_ORDER:
    n   = int((df["dx"] == dx).sum())
    pct = n / len(df) * 100
    mal = "Maligno" if dx in MALIGNOS else "Benigno"
    print(f"  {dx:<8} {n:>6,}  {pct:>5.1f}%  {mal}")
print()
print(f"  Malignas : {(df['malignidad']=='Maligno').sum():,} "
      f"({(df['malignidad']=='Maligno').mean()*100:.1f}%)")
print(f"  Benignas : {(df['malignidad']=='Benigno').sum():,} "
      f"({(df['malignidad']=='Benigno').mean()*100:.1f}%)")
print()
print(f"  Cramér's V (dx ~ malignidad)  : {cv_matrix.loc['dx','malignidad']:.3f}")
print(f"  Cramér's V (dx ~ dx_type)     : {cv_matrix.loc['dx','dx_type']:.3f}")
print(f"  Cramér's V (dx ~ localizacion): {cv_matrix.loc['dx','localization']:.3f}")
print(f"  Cramér's V (dx ~ sex)         : {cv_matrix.loc['dx','sex']:.3f}")
print(f"  Cramér's V (dx ~ age_group)   : {cv_matrix.loc['dx','age_group']:.3f}")
print(f"  Kruskal-Wallis (edad ~ dx)    : H={stat_kw:.2f}, p={p_kw:.2e}")
print()
print(f"  Figuras generadas : 10  →  {OUT_DIR}")
print(f"  Excel generado    : eda_resultados_HAM10000.xlsx")
print("═"*62)
print("  EDA completado sin errores.")
print("═"*62)
