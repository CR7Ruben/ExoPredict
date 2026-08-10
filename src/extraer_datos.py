import requests
import pandas as pd
from pathlib import Path


# ==========================================
# CONFIGURACIÓN
# ==========================================

URL_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "nasa_koi_raw.csv"


# ==========================================
# CONSULTA NASA
# ==========================================

QUERY = """
SELECT
    kepid,
    kepoi_name,
    koi_disposition,
    koi_period,
    koi_period_err1,
    koi_period_err2,
    koi_duration,
    koi_depth,
    koi_prad,
    koi_prad_err1,
    koi_prad_err2,
    koi_teq,
    koi_insol,
    koi_impact,
    koi_model_snr,
    koi_steff,
    koi_slogg,
    koi_srad,
    ra,
    dec
FROM q1_q17_dr25_koi
WHERE koi_disposition IS NOT NULL
"""

# ==========================================
# EXTRACCIÓN
# ==========================================

def extraer_datos():

    print("🚀 Conectando con NASA Exoplanet Archive...")

    parametros = {
        "query": QUERY,
        "format": "csv"
    }

    respuesta = requests.get(
        URL_TAP,
        params=parametros,
        timeout=60
    )

    respuesta.raise_for_status()

    print("✅ Datos recibidos correctamente.")

    with open(OUTPUT_FILE, "wb") as archivo:
        archivo.write(respuesta.content)

    datos = pd.read_csv(OUTPUT_FILE)

    print(f"📊 Registros obtenidos: {len(datos)}")
    print(f"📋 Columnas obtenidas: {len(datos.columns)}")

    return datos


# ==========================================
# EJECUCIÓN
# ==========================================

if __name__ == "__main__":

    datos = extraer_datos()

    print("\nPrimeros registros:")
    print(datos.head())

    print("\nDistribución de la variable objetivo:")
    print(datos["koi_disposition"].value_counts())

    print("\nArchivo guardado en:")
    print(OUTPUT_FILE)