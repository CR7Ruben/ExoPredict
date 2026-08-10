import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "nasa_koi_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "graficas"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGAR DATOS
# ============================================================

print("=" * 60)
print("🔭 EXOPREDICT - ANÁLISIS EXPLORATORIO")
print("=" * 60)

datos = pd.read_csv(INPUT_FILE)

print(f"\n📊 Registros: {len(datos)}")
print(f"📋 Variables: {len(datos.columns)}")


# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

sns.set_theme(style="whitegrid")

ORDEN_CLASES = [
    "FALSE POSITIVE",
    "CANDIDATE",
    "CONFIRMED"
]


# ============================================================
# 1. DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n📊 Generando distribución de clases...")

plt.figure(figsize=(10, 6))

sns.countplot(
    data=datos,
    x="koi_disposition",
    order=ORDEN_CLASES
)

plt.title(
    "Distribución de objetos Kepler por clasificación NASA",
    fontsize=15
)

plt.xlabel("Clasificación")
plt.ylabel("Número de objetos")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_distribucion_clases.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. RADIO PLANETARIO
# ============================================================

print("🪐 Generando análisis de radio planetario...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=datos,
    x="koi_disposition",
    y="koi_prad",
    order=ORDEN_CLASES
)

plt.title(
    "Distribución del radio planetario según clasificación",
    fontsize=15
)

plt.xlabel("Clasificación")
plt.ylabel("Radio planetario (radio terrestre)")

plt.ylim(
    0,
    datos["koi_prad"].quantile(0.95)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_radio_planetario.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. TEMPERATURA DE EQUILIBRIO
# ============================================================

print("🌡️ Generando análisis de temperatura...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=datos,
    x="koi_disposition",
    y="koi_teq",
    order=ORDEN_CLASES
)

plt.title(
    "Temperatura de equilibrio según clasificación",
    fontsize=15
)

plt.xlabel("Clasificación")
plt.ylabel("Temperatura de equilibrio (K)")

plt.ylim(
    0,
    datos["koi_teq"].quantile(0.95)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_temperatura.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. PERIODO ORBITAL
# ============================================================

print("🌎 Generando análisis de periodo orbital...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=datos,
    x="koi_disposition",
    y="koi_period",
    order=ORDEN_CLASES
)

plt.title(
    "Periodo orbital según clasificación",
    fontsize=15
)

plt.xlabel("Clasificación")
plt.ylabel("Periodo orbital (días)")

plt.ylim(
    0,
    datos["koi_period"].quantile(0.95)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_periodo_orbital.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. RELACIÓN RADIO VS TEMPERATURA
# ============================================================

print("📈 Generando relación radio-temperatura...")

plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=datos,
    x="koi_teq",
    y="koi_prad",
    hue="koi_disposition",
    hue_order=ORDEN_CLASES,
    alpha=0.6
)

plt.title(
    "Relación entre temperatura y radio planetario",
    fontsize=15
)

plt.xlabel("Temperatura de equilibrio (K)")
plt.ylabel("Radio planetario")

plt.xlim(
    0,
    datos["koi_teq"].quantile(0.98)
)

plt.ylim(
    0,
    datos["koi_prad"].quantile(0.98)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_radio_vs_temperatura.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. RELACIÓN SEÑAL-RUIDO
# ============================================================

print("📡 Generando análisis de señal/ruido...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=datos,
    x="koi_disposition",
    y="koi_model_snr",
    order=ORDEN_CLASES
)

plt.title(
    "Relación señal/ruido según clasificación",
    fontsize=15
)

plt.xlabel("Clasificación")
plt.ylabel("Model SNR")

plt.ylim(
    0,
    datos["koi_model_snr"].quantile(0.95)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_senal_ruido.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. MATRIZ DE CORRELACIÓN
# ============================================================

print("🔗 Generando matriz de correlación...")

VARIABLES_CORRELACION = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_impact",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad"
]

correlacion = datos[VARIABLES_CORRELACION].corr()

plt.figure(figsize=(13, 10))

sns.heatmap(
    correlacion,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Matriz de correlación de variables astronómicas",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_correlacion.png",
    dpi=300
)

plt.close()


# ============================================================
# RESUMEN ESTADÍSTICO POR CLASE
# ============================================================

print("\n" + "=" * 60)
print("📐 RESUMEN POR CLASE")
print("=" * 60)

resumen = datos.groupby(
    "koi_disposition"
)[
    [
        "koi_period",
        "koi_prad",
        "koi_teq",
        "koi_model_snr"
    ]
].median()

print(resumen.round(2))


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("✅ ANÁLISIS EXPLORATORIO TERMINADO")
print("=" * 60)

print("\n📁 Gráficas guardadas en:")

print(OUTPUT_DIR)

for archivo in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"   ✓ {archivo.name}")