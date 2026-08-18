import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# CONFIGURACIÓN
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

# VARIABLES
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

# CARGAR DATOS
print("🚀 EXOPREDICT - ENTRENAMIENTO DE MODELOS")
print("=" * 65)

print("\n📥 Cargando dataset procesado...")

datos = pd.read_csv(INPUT_FILE)

print(f"✓ Registros: {len(datos)}")
print(f"✓ Variables predictoras: {len(VARIABLES)}")

# X / Y
X = datos[VARIABLES]
y = datos[OBJETIVO]

print("\n🎯 Variable objetivo:")
print(
    datos["koi_disposition"]
    .value_counts()
)

# TRAIN / TEST
print("\n✂️ Separando datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"✓ Entrenamiento: {len(X_train)} registros")
print(f"✓ Prueba: {len(X_test)} registros")

# MODELO 1 - REGRESIÓN LOGÍSTICA
print("\n" + "=" * 65)
print("📈 MODELO 1 - REGRESIÓN LOGÍSTICA")
print("=" * 65)

modelo_logistico = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "modelo",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

modelo_logistico.fit(
    X_train,
    y_train
)

pred_logistico = modelo_logistico.predict(X_test)
prob_logistico = modelo_logistico.predict_proba(X_test)

# MÉTRICAS LOGÍSTICA
accuracy_logistico = accuracy_score(
    y_test,
    pred_logistico
)

precision_logistico = precision_score(
    y_test,
    pred_logistico,
    average="weighted",
    zero_division=0
)

recall_logistico = recall_score(
    y_test,
    pred_logistico,
    average="weighted",
    zero_division=0
)

f1_logistico = f1_score(
    y_test,
    pred_logistico,
    average="weighted",
    zero_division=0
)

auc_logistico = roc_auc_score(
    y_test,
    prob_logistico,
    multi_class="ovr",
    average="weighted"
)

print(f"\nAccuracy : {accuracy_logistico:.4f}")
print(f"Precision: {precision_logistico:.4f}")
print(f"Recall   : {recall_logistico:.4f}")
print(f"F1-Score : {f1_logistico:.4f}")
print(f"ROC-AUC  : {auc_logistico:.4f}")

# MODELO 2 - RANDOM FOREST
print("\n" + "=" * 65)
print("🌲 MODELO 2 - RANDOM FOREST")
print("=" * 65)

modelo_rf = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "modelo",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])
modelo_rf.fit(
    X_train,
    y_train
)

pred_rf = modelo_rf.predict(X_test)
prob_rf = modelo_rf.predict_proba(X_test)

# MÉTRICAS RANDOM FOREST
accuracy_rf = accuracy_score(
    y_test,
    pred_rf
)

precision_rf = precision_score(
    y_test,
    pred_rf,
    average="weighted",
    zero_division=0
)

recall_rf = recall_score(
    y_test,
    pred_rf,
    average="weighted",
    zero_division=0
)

f1_rf = f1_score(
    y_test,
    pred_rf,
    average="weighted",
    zero_division=0
)

auc_rf = roc_auc_score(
    y_test,
    prob_rf,
    multi_class="ovr",
    average="weighted"
)

print(f"\nAccuracy : {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall   : {recall_rf:.4f}")
print(f"F1-Score : {f1_rf:.4f}")
print(f"ROC-AUC  : {auc_rf:.4f}")

# COMPARACIÓN
print("\n" + "=" * 65)
print("🏆 COMPARACIÓN DE MODELOS")
print("=" * 65)
comparacion = pd.DataFrame({
    "Modelo": [
        "Regresión Logística",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_logistico,
        accuracy_rf
    ],
    "Precision": [
        precision_logistico,
        precision_rf
    ],
    "Recall": [
        recall_logistico,
        recall_rf
    ],
    "F1": [
        f1_logistico,
        f1_rf
    ],
    "ROC-AUC": [
        auc_logistico,
        auc_rf
    ]
})

print(
    comparacion.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ELEGIR MEJOR MODELO
if f1_rf >= f1_logistico:

    mejor_modelo = modelo_rf
    nombre_mejor = "Random Forest"
    mejor_f1 = f1_rf

    pred_mejor = pred_rf
    prob_mejor = prob_rf

else:
    mejor_modelo = modelo_logistico
    nombre_mejor = "Regresión Logística"
    mejor_f1 = f1_logistico

    pred_mejor = pred_logistico
    prob_mejor = prob_logistico

print("\n🏆 Mejor modelo:", nombre_mejor)
print(f"F1-Score: {mejor_f1:.4f}")

# REPORTE DE CLASIFICACIÓN
print("\n" + "=" * 65)
print("📋 REPORTE DE CLASIFICACIÓN")
print("=" * 65)
print(
    classification_report(
        y_test,
        pred_mejor,
        target_names=[
            "FALSE POSITIVE",
            "CANDIDATE",
            "CONFIRMED"
        ],
        zero_division=0
    )
)

# MATRIZ DE CONFUSIÓN
matriz = confusion_matrix(
    y_test,
    pred_mejor
)
print("\n🔢 MATRIZ DE CONFUSIÓN:")
print(matriz)

# IMPORTANCIA DE VARIABLES
if nombre_mejor == "Random Forest":
    modelo_final_rf = mejor_modelo.named_steps["modelo"]
    importancia = pd.DataFrame({
        "variable": VARIABLES,
        "importancia": modelo_final_rf.feature_importances_
    })

    importancia = importancia.sort_values(
        "importancia",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("🔬 IMPORTANCIA DE VARIABLES")
    print("=" * 65)

    print(
        importancia.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
else:

    importancia = None

# GUARDAR MODELO
modelo_path = MODELS_DIR / "exopredict_model.pkl"
joblib.dump(
    mejor_modelo,
    modelo_path
)
print("\n💾 Modelo guardado en:")
print(modelo_path)

# GUARDAR MÉTRICAS
metricas = {
    "modelo": nombre_mejor,
    "accuracy": accuracy_score(
        y_test,
        pred_mejor
    ),

    "precision": precision_score(
        y_test,
        pred_mejor,
        average="weighted",
        zero_division=0
    ),

    "recall": recall_score(
        y_test,
        pred_mejor,
        average="weighted",
        zero_division=0
    ),

    "f1": f1_score(
        y_test,
        pred_mejor,
        average="weighted",
        zero_division=0
    ),

    "roc_auc": roc_auc_score(
        y_test,
        prob_mejor,
        multi_class="ovr",
        average="weighted"
    ),
    "confusion_matrix": matriz,
    "variables": VARIABLES
}

metricas_path = MODELS_DIR / "metricas_modelo.pkl"
joblib.dump(
    metricas,
    metricas_path
)

print("\n📊 Métricas guardadas en:")
print(metricas_path)

# FINAL
print("\n" + "=" * 65)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 65)