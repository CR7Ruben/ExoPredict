import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
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


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

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
print("🚀 EXOPREDICT - OPTIMIZACIÓN DEL MODELO")
print("=" * 70)

print("\n📥 Cargando dataset...")

datos = pd.read_csv(INPUT_FILE)

X = datos[VARIABLES]
y = datos[OBJETIVO]

print(f"✓ Registros: {len(datos)}")
print(f"✓ Variables: {len(VARIABLES)}")


# ============================================================
# DIVISIÓN DE DATOS
# ============================================================

print("\n✂️ Separando datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"✓ Entrenamiento: {len(X_train)}")
print(f"✓ Prueba: {len(X_test)}")


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "modelo",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )
])


# ============================================================
# ESPACIO DE HIPERPARÁMETROS
# ============================================================

parametros = {

    "modelo__n_estimators": [
        200,
        300,
        400,
        500,
        600
    ],

    "modelo__max_depth": [
        5,
        8,
        10,
        12,
        15,
        20,
        None
    ],

    "modelo__min_samples_split": [
        2,
        5,
        10,
        15
    ],

    "modelo__min_samples_leaf": [
        1,
        2,
        4,
        6
    ],

    "modelo__max_features": [
        "sqrt",
        "log2",
        None
    ]
}


# ============================================================
# RANDOMIZED SEARCH
# ============================================================

print("\n🔎 Buscando los mejores hiperparámetros...")

print("⏳ Esto puede tardar algunos minutos.\n")

busqueda = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parametros,
    n_iter=25,
    scoring="f1_weighted",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


busqueda.fit(
    X_train,
    y_train
)


# ============================================================
# MEJORES PARÁMETROS
# ============================================================

print("\n" + "=" * 70)
print("🏆 MEJORES HIPERPARÁMETROS")
print("=" * 70)

for parametro, valor in busqueda.best_params_.items():

    print(
        f"{parametro}: {valor}"
    )


print(
    f"\n⭐ Mejor F1 durante validación cruzada: "
    f"{busqueda.best_score_:.4f}"
)


# ============================================================
# MODELO OPTIMIZADO
# ============================================================

modelo_optimizado = busqueda.best_estimator_

pred = modelo_optimizado.predict(X_test)

prob = modelo_optimizado.predict_proba(X_test)


# ============================================================
# MÉTRICAS
# ============================================================

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

f1 = f1_score(
    y_test,
    pred,
    average="weighted",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    prob,
    multi_class="ovr",
    average="weighted"
)


print("\n" + "=" * 70)
print("📊 RESULTADOS DEL MODELO OPTIMIZADO")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# REPORTE
# ============================================================

print("\n" + "=" * 70)
print("📋 REPORTE DE CLASIFICACIÓN")
print("=" * 70)

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "FALSE POSITIVE",
            "CANDIDATE",
            "CONFIRMED"
        ],
        zero_division=0
    )
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

matriz = confusion_matrix(
    y_test,
    pred
)

print("\n🔢 MATRIZ DE CONFUSIÓN:")

print(matriz)


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

modelo_rf = modelo_optimizado.named_steps["modelo"]

importancia = pd.DataFrame({

    "variable": VARIABLES,

    "importancia": modelo_rf.feature_importances_

})

importancia = importancia.sort_values(
    "importancia",
    ascending=False
)


print("\n" + "=" * 70)
print("🔬 IMPORTANCIA DE VARIABLES")
print("=" * 70)

print(
    importancia.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# GUARDAR MODELO
# ============================================================

modelo_path = (
    MODELS_DIR
    / "exopredict_model_optimizado.pkl"
)

joblib.dump(
    modelo_optimizado,
    modelo_path
)

print("\n💾 Modelo optimizado guardado en:")

print(modelo_path)


# ============================================================
# GUARDAR MÉTRICAS
# ============================================================

metricas = {

    "modelo": "Random Forest Optimizado",

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "roc_auc": roc_auc,

    "f1_validacion_cruzada": busqueda.best_score_,

    "confusion_matrix": matriz,

    "variables": VARIABLES,

    "mejores_parametros": busqueda.best_params_

}


metricas_path = (
    MODELS_DIR
    / "metricas_modelo_optimizado.pkl"
)

joblib.dump(
    metricas,
    metricas_path
)

print("\n📊 Métricas guardadas en:")

print(metricas_path)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("✅ OPTIMIZACIÓN COMPLETADA")
print("=" * 70)