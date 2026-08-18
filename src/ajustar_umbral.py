import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nasa_koi_clean.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "exopredict_model_optimizado.pkl"
)

OUTPUT_MODEL = (
    BASE_DIR
    / "models"
    / "exopredict_model_final.pkl"
)

OUTPUT_METRICS = (
    BASE_DIR
    / "models"
    / "metricas_modelo_final.pkl"
)

VARIABLES = [
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

OBJETIVO = "target"

# CARGAR DATOS Y MODELO
print("=" * 70)
print("🔭 EXOPREDICT - AJUSTE DEL UMBRAL DE DECISIÓN")
print("=" * 70)

print("\n📥 Cargando dataset...")

datos = pd.read_csv(DATA_FILE)

X = datos[VARIABLES]
y = datos[OBJETIVO]

print(f"✓ Registros: {len(datos)}")


print("\n🤖 Cargando modelo optimizado...")

modelo = joblib.load(MODEL_FILE)

print("✓ Modelo cargado correctamente")

# RECREAR MISMO TRAIN/TEST
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# PREDICCIONES
print("\n🔮 Generando probabilidades...")
probabilidades = modelo.predict_proba(X_test)
clases = modelo.classes_
print("\nClases detectadas:")
print(clases)

# FUNCIÓN PARA AJUSTAR CANDIDATE
def predecir_con_umbral(probabilidades, clases, umbral):

    predicciones = []

    indice_fp = list(clases).index(0)
    indice_candidate = list(clases).index(1)
    indice_confirmed = list(clases).index(2)

    for prob in probabilidades:
        prob_fp = prob[indice_fp]
        prob_candidate = prob[indice_candidate]
        prob_confirmed = prob[indice_confirmed]

        if prob_candidate >= umbral:
            predicciones.append(1)
        else:
            probabilidades_resto = [
                prob_fp,
                prob_confirmed
            ]
            if prob_fp >= prob_confirmed:
                predicciones.append(0)
            else:
                predicciones.append(2)
    return np.array(predicciones)

# PRUEBA DE UMBRALES
umbrales = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65
]

resultados = []

mejor_umbral = 0.50
mejor_f1_macro = -1
mejor_prediccion = None

print("\n" + "=" * 70)
print("🧪 PROBANDO UMBRALES PARA CANDIDATE")
print("=" * 70)

for umbral in umbrales:
    pred = predecir_con_umbral(
        probabilidades,
        clases,
        umbral
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    precision = precision_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    f1_clases = f1_score(
        y_test,
        pred,
        average=None,
        labels=[0, 1, 2],
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilidades,
        multi_class="ovr",
        average="weighted"
    )

    resultados.append({

        "umbral": umbral,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_false_positive": f1_clases[0],
        "f1_candidate": f1_clases[1],
        "f1_confirmed": f1_clases[2],
        "roc_auc": roc_auc
    })

    print(
        f"\nUmbral {umbral:.2f}"
    )
    print(
        f"Accuracy: {accuracy:.4f}"
    )
    print(
        f"F1 Macro: {f1_macro:.4f}"
    )
    print(
        f"F1 Weighted: {f1_weighted:.4f}"
    )
    print(
        f"F1 FALSE POSITIVE: {f1_clases[0]:.4f}"
    )
    print(
        f"F1 CANDIDATE: {f1_clases[1]:.4f}"
    )
    print(
        f"F1 CONFIRMED: {f1_clases[2]:.4f}"
    )

    if f1_macro > mejor_f1_macro:
        mejor_f1_macro = f1_macro
        mejor_umbral = umbral
        mejor_prediccion = pred

# TABLA COMPARATIVA
resultados_df = pd.DataFrame(
    resultados
)

print("\n" + "=" * 70)
print("📊 COMPARACIÓN DE UMBRALES")
print("=" * 70)

print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# MEJOR RESULTADO
print("\n" + "=" * 70)
print("🏆 MEJOR UMBRAL")
print("=" * 70)

print(
    f"\nUmbral seleccionado: {mejor_umbral:.2f}"
)

print(
    f"F1 Macro: {mejor_f1_macro:.4f}"
)

# REPORTE FINAL
print("\n📋 REPORTE DEL MODELO FINAL")

print(
    classification_report(
        y_test,
        mejor_prediccion,
        target_names=[
            "FALSE POSITIVE",
            "CANDIDATE",
            "CONFIRMED"
        ],
        zero_division=0
    )
)

# MÉTRICAS FINALES
accuracy_final = accuracy_score(
    y_test,
    mejor_prediccion
)

precision_final = precision_score(
    y_test,
    mejor_prediccion,
    average="weighted",
    zero_division=0
)

recall_final = recall_score(
    y_test,
    mejor_prediccion,
    average="weighted",
    zero_division=0
)

f1_macro_final = f1_score(
    y_test,
    mejor_prediccion,
    average="macro",
    zero_division=0
)

f1_weighted_final = f1_score(
    y_test,
    mejor_prediccion,
    average="weighted",
    zero_division=0
)

# DECISIÓN AUTOMÁTICA
f1_candidate_final = f1_score(
    y_test,
    mejor_prediccion,
    labels=[1],
    average="macro",
    zero_division=0
)


print("\n" + "=" * 70)
print("📌 DECISIÓN DEL MODELO")
print("=" * 70)

print(
    f"\nF1 CANDIDATE final: {f1_candidate_final:.4f}"
)

print(
    f"F1 Macro final: {f1_macro_final:.4f}"
)

# GUARDAR MÉTRICAS
metricas = {
    "modelo": "Random Forest Optimizado + Umbral",
    "umbral_candidate": mejor_umbral,
    "accuracy": accuracy_final,
    "precision": precision_final,
    "recall": recall_final,
    "f1_macro": f1_macro_final,
    "f1_weighted": f1_weighted_final,
    "f1_candidate": f1_candidate_final,
    "resultados_umbral": resultados_df
}

joblib.dump(
    metricas,
    OUTPUT_METRICS
)

print("\n📊 Métricas guardadas en:")
print(OUTPUT_METRICS)

# IMPORTANTE
print("\n" + "=" * 70)
print("ℹ️ IMPORTANTE")
print("=" * 70)

print(
    "\nEl modelo optimizado original NO fue modificado."
)

print(
    "Este experimento solamente ajusta la decisión de clasificación."
)

print(
    "\nSi el resultado no mejora significativamente,"
)

print(
    "conservaremos exopredict_model_optimizado.pkl como modelo final."
)

print("\n" + "=" * 70)
print("✅ PRUEBA TERMINADA")
print("=" * 70)