import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "nasa_koi_raw.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "nasa_koi_clean.csv"


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

VARIABLES_MODELO = [
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


# ============================================================
# EXTRACCIÓN
# ============================================================

def cargar_datos():

    print("📥 Cargando dataset RAW de NASA...")

    datos = pd.read_csv(INPUT_FILE)

    print(f"   Registros iniciales: {len(datos)}")

    return datos


# ============================================================
# ELIMINAR DUPLICADOS
# ============================================================

def eliminar_duplicados(datos):

    antes = len(datos)

    datos = datos.drop_duplicates()

    despues = len(datos)

    eliminados = antes - despues

    print(f"♻️ Duplicados eliminados: {eliminados}")

    return datos


# ============================================================
# LIMPIEZA DE NULOS
# ============================================================

def limpiar_nulos(datos):

    print("\n🧹 Tratamiento de valores nulos")

    # Variables utilizadas por el modelo
    columnas_numericas = VARIABLES_MODELO

    nulos_antes = datos[columnas_numericas].isnull().sum().sum()

    print(f"   Valores nulos detectados: {nulos_antes}")

    # Imputación mediante mediana
    # La mediana es robusta frente a valores extremos.
    for columna in columnas_numericas:

        if datos[columna].isnull().any():

            mediana = datos[columna].median()

            datos[columna] = datos[columna].fillna(mediana)

            print(
                f"   ✓ {columna}: "
                f"nulos reemplazados por mediana "
                f"({mediana:.4f})"
            )

    nulos_despues = datos[columnas_numericas].isnull().sum().sum()

    print(f"   Nulos restantes: {nulos_despues}")

    return datos


# ============================================================
# LIMPIEZA DE VALORES INVÁLIDOS
# ============================================================

def limpiar_valores_invalidos(datos):

    print("\n🔎 Validación de valores inválidos")

    columnas_numericas = VARIABLES_MODELO

    negativos_antes = 0

    for columna in columnas_numericas:

        negativos = (datos[columna] < 0).sum()

        if negativos > 0:

            negativos_antes += negativos

            print(
                f"   ⚠️ {columna}: "
                f"{negativos} valores negativos"
            )

    if negativos_antes == 0:
        print("   ✓ No se encontraron valores negativos.")

    return datos


# ============================================================
# VARIABLE OBJETIVO
# ============================================================

def preparar_objetivo(datos):

    print("\n🎯 Preparando variable objetivo")

    # Eliminamos registros sin clasificación
    datos = datos[
        datos["koi_disposition"].isin(
            ["FALSE POSITIVE", "CONFIRMED", "CANDIDATE"]
        )
    ].copy()

    # Codificación:
    # 0 = FALSE POSITIVE
    # 1 = CANDIDATE
    # 2 = CONFIRMED

    mapa = {
        "FALSE POSITIVE": 0,
        "CANDIDATE": 1,
        "CONFIRMED": 2
    }

    datos["target"] = datos["koi_disposition"].map(mapa)

    print("\n   Clases:")

    print(
        datos["koi_disposition"]
        .value_counts()
    )

    return datos


# ============================================================
# SELECCIÓN FINAL
# ============================================================

def seleccionar_variables(datos):

    columnas_finales = [
        "kepoi_name",
        "koi_disposition",
        "target"
    ] + VARIABLES_MODELO

    datos = datos[columnas_finales].copy()

    return datos


# ============================================================
# VALIDACIÓN FINAL
# ============================================================

def validar_dataset(datos):

    print("\n" + "=" * 60)
    print("🔬 VALIDACIÓN FINAL DEL DATASET")
    print("=" * 60)

    print(f"Registros finales: {len(datos)}")
    print(f"Columnas finales: {len(datos.columns)}")

    print(
        f"\nValores nulos totales: "
        f"{datos.isnull().sum().sum()}"
    )

    print(
        f"Duplicados: "
        f"{datos.duplicated().sum()}"
    )

    print("\nDistribución objetivo:")

    print(
        datos["koi_disposition"]
        .value_counts()
    )

    print("\n✓ Validación terminada.")


# ============================================================
# GUARDAR
# ============================================================

def guardar_datos(datos):

    try:
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        datos.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\n💾 Dataset procesado guardado en:")
        print(OUTPUT_FILE)

    except Exception as error:

        print("\n❌ Error al guardar el dataset:")
        print(error)


# ============================================================
# PIPELINE ETL
# ============================================================

def ejecutar_etl():

    print("=" * 60)
    print("🚀 EXOPREDICT - PROCESO ETL")
    print("=" * 60)

    datos = cargar_datos()

    datos = eliminar_duplicados(datos)

    datos = limpiar_nulos(datos)

    datos = limpiar_valores_invalidos(datos)

    datos = preparar_objetivo(datos)

    datos = seleccionar_variables(datos)

    validar_dataset(datos)

    guardar_datos(datos)

    print("\n" + "=" * 60)
    print("✅ PROCESO ETL COMPLETADO")
    print("=" * 60)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    ejecutar_etl()