"""
Generación del dataset balanceado HAM10000 para entrenamiento de DenseNet.

Criterios aplicados (ver informe metodológico del TFM):
  1. Igualación del número de instancias por clase: se toman las 1.954
     imágenes malignas (totalidad disponible) y se submuestrean 1.954
     benignas de las 8.061 originales.
  2. Selección a nivel de lesión (lesion_id), no de imagen: de cada
     lesión benigna seleccionada se toma una sola fotografía al azar,
     para no incluir dos fotos casi idénticas de la misma lesión.
  3. Muestreo estratificado proporcional por subtipo clínico benigno
     (nv, bkl, vasc, df), respetando su proporción real en el dataset.
  4. Semilla aleatoria fija para que el resultado sea reproducible.

Entrada : HAM10000_metadata.csv
Salida  : HAM10000_metadata_balanceo.csv
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────
SEED = 42                                   # semilla fija -> resultado reproducible
RUTA_CSV_ENTRADA = 'D:\\TFM\\HAM10000\\HAM10000_metadata.csv'
RUTA_CSV_SALIDA = 'D:\\TFM\\HAM10000\\HAM10000_metadata_balanceo.csv'

# Mapeo de las 7 clases dx a la variable binaria benigno/maligno
MAPA_CATEGORIA = {
    'akiec': 'Maligno',
    'bcc':   'Maligno',
    'mel':   'Maligno',
    'bkl':   'Benigno',
    'df':    'Benigno',
    'nv':    'Benigno',
    'vasc':  'Benigno',
}

rng = np.random.default_rng(SEED)

# ──────────────────────────────────────────────────────────
# 1. Cargar metadatos y clasificar benigno/maligno
# ──────────────────────────────────────────────────────────
df = pd.read_csv(RUTA_CSV_ENTRADA, dtype={'age': 'Int64'})
df['categoria_dx'] = df['dx'].map(MAPA_CATEGORIA)
assert df['categoria_dx'].isna().sum() == 0, "Hay valores de dx sin mapear"

mal_df = df[df['categoria_dx'] == 'Maligno'].copy()
ben_df = df[df['categoria_dx'] == 'Benigno'].copy()

N_TARGET = len(mal_df)  # 1.954 → número de imágenes malignas disponibles

# ──────────────────────────────────────────────────────────
# 2. Reparto proporcional de lesiones benignas por subtipo clínico
#    (método de mayores restos para que la suma sea exactamente N_TARGET)
# ──────────────────────────────────────────────────────────
lesion_dx = ben_df.drop_duplicates('lesion_id')[['lesion_id', 'dx']]
counts_subtipo = lesion_dx['dx'].value_counts()
proporciones = counts_subtipo / counts_subtipo.sum()

raw = proporciones * N_TARGET
base = np.floor(raw).astype(int)
resto = (raw - base).sort_values(ascending=False)
faltan = N_TARGET - base.sum()
for subtipo in resto.index[:faltan]:
    base[subtipo] += 1
objetivo_por_subtipo = base.to_dict()

# ──────────────────────────────────────────────────────────
# 3. Selección aleatoria de lesiones únicas por subtipo (sin reemplazo)
# ──────────────────────────────────────────────────────────
lesiones_seleccionadas = []
for subtipo, n in objetivo_por_subtipo.items():
    pool = lesion_dx.loc[lesion_dx['dx'] == subtipo, 'lesion_id'].to_numpy()
    elegidas = rng.choice(pool, size=n, replace=False)
    lesiones_seleccionadas.extend(elegidas)

# ──────────────────────────────────────────────────────────
# 4. Para cada lesión seleccionada, elegir aleatoriamente UNA sola
#    fotografía entre las disponibles (evita duplicar la misma lesión)
# ──────────────────────────────────────────────────────────
ben_filtrado = ben_df[ben_df['lesion_id'].isin(lesiones_seleccionadas)]

filas_elegidas = []
for lesion_id, grupo in ben_filtrado.groupby('lesion_id'):
    idx_elegido = rng.choice(grupo.index.to_numpy())
    filas_elegidas.append(idx_elegido)

ben_muestra = ben_df.loc[filas_elegidas].reset_index(drop=True)

assert ben_muestra['lesion_id'].is_unique, "Hay lesiones benignas repetidas"
assert len(ben_muestra) == N_TARGET, f"Tamaño incorrecto: {len(ben_muestra)}"

# ──────────────────────────────────────────────────────────
# 5. Unir malignas + benignas y mezclar el orden final de filas
# ──────────────────────────────────────────────────────────
balanceado = pd.concat([mal_df, ben_muestra], ignore_index=True)
balanceado = balanceado.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Dataset balanceado: {balanceado.shape[0]} filas")
print(balanceado['categoria_dx'].value_counts())
print("\nComposición benigna por subtipo:")
print(ben_muestra['dx'].value_counts())

# ──────────────────────────────────────────────────────────
# 6. Exportar a CSV
# ──────────────────────────────────────────────────────────
balanceado.drop(columns=['categoria_dx']).to_csv(RUTA_CSV_SALIDA, index=False, encoding='utf-8-sig', sep=';')
print(f"\nArchivo guardado en: {RUTA_CSV_SALIDA}")