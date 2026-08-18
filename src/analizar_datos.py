import pandas as pd
from pathlib import Path

# CONFIGURACIÓN

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "nasa_koi_raw.csv"

# CARGAR DATOS
print("🔭 Cargando datos de NASA...")
datos = pd.read_csv(INPUT_FILE)
print("✅ Dataset cargado correctamente.")

# INFORMACIÓN GENERAL
print("\n" + "=" * 60)
print("📊 INFORMACIÓN GENERAL")
print("=" * 60)

print(f"Registros: {datos.shape[0]}")
print(f"Columnas: {datos.shape[1]}")

# TIPOS DE DATOS
print("\n" + "=" * 60)
print("🔤 TIPOS DE DATOS")
print("=" * 60)
print(datos.dtypes)

# VALORES NULOS
print("\n" + "=" * 60)
print("🧹 VALORES NULOS")
print("=" * 60)

nulos = datos.isnull().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)

print(nulos)

# PORCENTAJE DE NULOS
print("\n" + "=" * 60)
print("📈 PORCENTAJE DE NULOS")
print("=" * 60)

porcentaje_nulos = (
    datos.isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)
print(porcentaje_nulos[porcentaje_nulos > 0].round(2))

# DUPLICADOS
print("\n" + "=" * 60)
print("♻️ DUPLICADOS")
print("=" * 60)
duplicados = datos.duplicated().sum()

print(f"Registros duplicados: {duplicados}")

# VARIABLE OBJETIVO
print("\n" + "=" * 60)
print("🎯 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("=" * 60)

print(datos["koi_disposition"].value_counts())
print("\nPorcentaje:")
print(
    datos["koi_disposition"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ESTADÍSTICAS
print("\n" + "=" * 60)
print("📐 ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)
print(datos.describe().T)

# VALORES ÚNICOS
print("\n" + "=" * 60)
print("🔢 VALORES ÚNICOS")
print("=" * 60)

for columna in datos.columns:
    print(
        f"{columna:25} "
        f"{datos[columna].nunique():>6} valores únicos"
    )

# FINAL
print("\n" + "=" * 60)
print("✅ DIAGNÓSTICO TERMINADO")
print("=" * 60)