import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nasa_koi_clean.csv"
)

MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
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


# ============================================================
# CARGAR DATOS
# ============================================================

print("=" * 70)
print("🔭 EXOPREDICT - BALANCEO DE CLASES")
print("=" * 70)

print("\n📥 Cargando dataset...")

datos = pd.read_csv(INPUT_FILE)

X = datos[VARIABLES]
y = datos[OBJETIVO]

print(f"✓ Registros: {len(datos)}")
print(f"✓ Variables: {len(VARIABLES)}")


# ============================================================
# DIVISIÓN
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n✓ Entrenamiento: {len(X_train)}")
print(f"✓ Prueba: {len(X_test)}")


# ============================================================
# CONFIGURACIONES
# ============================================================

pesos = [
    1.2,
    1.5,
    1.8,
    2.0,
    2.5,
    3.0
]


resultados = []

mejor_modelo = None
mejor_f1_macro = -1
mejor_peso = None


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("🧪 PROBANDO DIFERENTES PESOS PARA CANDIDATE")
print("=" * 70)


for peso in pesos:

    print(
        f"\n🚀 Probando peso CANDIDATE = {peso}"
    )

    class_weight = {
        0: 1.0,
        1: peso,
        2: 1.0
    }

    modelo = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=15,
        min_samples_leaf=2,
        max_features="log2",
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    pred = modelo.predict(X_test)

    prob = modelo.predict_proba(X_test)

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

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

    f1_weighted = f1_score(
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

    f1_clases = f1_score(
        y_test,
        pred,
        average=None,
        labels=[
            "FALSE POSITIVE",
            "CANDIDATE",
            "CONFIRMED"
        ],
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        prob,
        multi_class="ovr",
        average="weighted"
    )

    f1_fp = f1_clases[0]
    f1_candidate = f1_clases[1]
    f1_confirmed = f1_clases[2]

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
        f"F1 FALSE POSITIVE: {f1_fp:.4f}"
    )

    print(
        f"F1 CANDIDATE: {f1_candidate:.4f}"
    )

    print(
        f"F1 CONFIRMED: {f1_confirmed:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    resultados.append({

        "peso_candidate": peso,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_macro": f1_macro,

        "f1_weighted": f1_weighted,

        "f1_false_positive": f1_fp,

        "f1_candidate": f1_candidate,

        "f1_confirmed": f1_confirmed,

        "roc_auc": roc_auc

    })


    # --------------------------------------------------------
    # SELECCIÓN
    # --------------------------------------------------------

    if f1_macro > mejor_f1_macro:

        mejor_f1_macro = f1_macro

        mejor_modelo = modelo

        mejor_peso = peso


# ============================================================
# TABLA FINAL
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)

print("\n" + "=" * 70)
print("📊 COMPARACIÓN DE MODELOS")
print("=" * 70)

print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# MEJOR MODELO
# ============================================================

print("\n" + "=" * 70)
print("🏆 MEJOR MODELO BALANCEADO")
print("=" * 70)

print(
    f"\nPeso utilizado para CANDIDATE: {mejor_peso}"
)

print(
    f"F1 Macro: {mejor_f1_macro:.4f}"
)


# ============================================================
# EVALUACIÓN FINAL
# ============================================================

pred_final = mejor_modelo.predict(
    X_test
)

prob_final = mejor_modelo.predict_proba(
    X_test
)

accuracy_final = accuracy_score(
    y_test,
    pred_final
)

precision_final = precision_score(
    y_test,
    pred_final,
    average="weighted",
    zero_division=0
)

recall_final = recall_score(
    y_test,
    pred_final,
    average="weighted",
    zero_division=0
)

f1_weighted_final = f1_score(
    y_test,
    pred_final,
    average="weighted",
    zero_division=0
)

f1_macro_final = f1_score(
    y_test,
    pred_final,
    average="macro",
    zero_division=0
)

roc_auc_final = roc_auc_score(
    y_test,
    prob_final,
    multi_class="ovr",
    average="weighted"
)


print("\n📋 REPORTE FINAL")

print(
    classification_report(
        y_test,
        pred_final,
        target_names=[
            "FALSE POSITIVE",
            "CANDIDATE",
            "CONFIRMED"
        ],
        zero_division=0
    )
)


# ============================================================
# GUARDAR SOLO COMO CANDIDATO
# ============================================================

modelo_path = (
    MODELS_DIR
    / "exopredict_model_balanceado.pkl"
)

joblib.dump(
    mejor_modelo,
    modelo_path
)


metricas = {

    "modelo": "Random Forest Balanceado",

    "peso_candidate": mejor_peso,

    "accuracy": accuracy_final,

    "precision": precision_final,

    "recall": recall_final,

    "f1_macro": f1_macro_final,

    "f1_weighted": f1_weighted_final,

    "roc_auc": roc_auc_final,

    "resultados_comparacion": resultados_df

}


metricas_path = (
    MODELS_DIR
    / "metricas_modelo_balanceado.pkl"
)

joblib.dump(
    metricas,
    metricas_path
)

print("\n💾 Modelo candidato guardado en:")
print(modelo_path)
print("\n📊 Métricas guardadas en:")
print(metricas_path)


print("\n" + "=" * 70)
print("✅ PRUEBA DE BALANCEO TERMINADA")
print("=" * 70)