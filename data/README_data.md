# Instrucciones para obtener los datos — HAM10000

Los datos **no están incluidos** en este repositorio por su tamaño (~2.5 GB).

## Descarga desde Kaggle (método recomendado)

### Requisitos previos
- Cuenta en https://www.kaggle.com (gratuita)
- Python con kaggle instalado: `pip install kaggle`

### Pasos

**1. Obtener credenciales Kaggle:**
- Inicia sesión en Kaggle → Avatar → Settings → API → "Create New API Token"
- Se descarga `kaggle.json`

**2. Ubicar el archivo de credenciales:**
- Windows: `C:\Users\<tu_usuario>\.kaggle\kaggle.json`
- Linux/Mac: `~/.kaggle/kaggle.json`

**3. Ejecutar desde la raíz del proyecto:**
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/
cd data
unzip skin-cancer-mnist-ham10000.zip -d HAM10000
```

## Descarga manual (alternativa)

1. Ve a: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Haz clic en "Download" (requiere cuenta Kaggle)
3. Descomprime en `data/HAM10000/`

## Estructura esperada tras la descarga

```
data/
└── HAM10000/
    ├── HAM10000_metadata.csv        ← Metadatos (requerido por el EDA)
    ├── HAM10000_images_part_1/      ← Imágenes (parte 1)
    ├── HAM10000_images_part_2/      ← Imágenes (parte 2)
    ├── hmnist_28_28_L.csv
    ├── hmnist_28_28_RGB.csv
    ├── hmnist_8_8_L.csv
    └── hmnist_8_8_RGB.csv
```

## Verificación

Puedes verificar la integridad del archivo de metadatos con Python:
```python
import pandas as pd
df = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')
print(f"Filas: {len(df)}")        # Debe mostrar 10015
print(f"Columnas: {list(df.columns)}")
```

## Referencia

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset,
a large collection of multi-source dermatoscopic images of common pigmented
skin lesions. Scientific Data, 5, 180161.
https://doi.org/10.1038/sdata.2018.161

**Licencia del dataset:** CC BY-NC-SA 4.0
