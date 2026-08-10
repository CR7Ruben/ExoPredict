# ExoPredict

### Sistema de Machine Learning para la clasificación de candidatos a exoplanetas

ExoPredict es un proyecto de **extracción de conocimiento mediante Machine Learning** desarrollado a partir de datos reales del **NASA Exoplanet Archive**.

El sistema implementa un flujo completo de **ETL, análisis exploratorio, entrenamiento, evaluación y optimización de modelos de aprendizaje supervisado**, con el objetivo de clasificar objetos observados por la misión Kepler en tres categorías:

* **CONFIRMED** — Exoplaneta confirmado.
* **CANDIDATE** — Candidato a exoplaneta.
* **FALSE POSITIVE** — Falso positivo.

---

## Objetivo del proyecto

Desarrollar un sistema capaz de utilizar características astronómicas y de observación para **clasificar automáticamente candidatos a exoplanetas**, facilitando el análisis de grandes volúmenes de información astronómica.

El proyecto busca demostrar la aplicación de técnicas de:

* Extracción de datos.
* Limpieza y transformación.
* Análisis exploratorio.
* Aprendizaje supervisado.
* Optimización de modelos.
* Evaluación mediante métricas.
* Visualización de resultados.
* Apoyo a la toma de decisiones mediante datos.

---

# Fuente de datos

Los datos utilizados provienen del **NASA Exoplanet Archive**, repositorio público especializado en información sobre exoplanetas y candidatos detectados por diferentes misiones astronómicas.

La información utilizada corresponde principalmente a datos de candidatos observados por la misión **Kepler**.

---

# Arquitectura del proyecto

El proyecto está organizado siguiendo un flujo de procesamiento de datos:

```text
                    NASA Exoplanet Archive
                              │
                              ▼
                         EXTRACCIÓN
                              │
                              ▼
                        Dataset RAW
                              │
                              ▼
                             ETL
                    Limpieza y transformación
                              │
                              ▼
                       Dataset procesado
                              │
                              ▼
                     ANÁLISIS EXPLORATORIO
                              │
                              ▼
                      MACHINE LEARNING
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       Regresión Logística          Random Forest
                                             │
                                             ▼
                                       Optimización
                                             │
                                             ▼
                                       Modelo final
                                             │
                                             ▼
                                    Dashboard interactivo
```

---

# Estructura del proyecto

```text
ExoPredict/
│
├── data/
│   ├── raw/
│   │   └── nasa_koi_raw.csv
│   │
│   ├── processed/
│   │   └── nasa_koi_clean.csv
│   │
│   └── graficas/
│       ├── 01_distribucion_clases.png
│       ├── 02_radio_planetario.png
│       ├── 03_temperatura.png
│       ├── 04_periodo_orbital.png
│       ├── 05_radio_vs_temperatura.png
│       ├── 06_senal_ruido.png
│       └── 07_correlacion.png
│
├── models/
│   ├── exopredict_model.pkl
│   ├── exopredict_model_optimizado.pkl
│   ├── metricas_modelo.pkl
│   └── metricas_modelo_optimizado.pkl
│
├── src/
│   ├── extraer_datos.py
│   ├── analizar_datos.py
│   ├── etl.py
│   │
│   ├── explorar_datos.py
│   │
│   ├── entrenar_modelo.py
│   └── optimizar_modelo.py
│   └── ajustar_umbral.py
│   └── balancear_clases.py
│
├── README.md
└── requirements.txt
```

---

# Tecnologías utilizadas

| Tecnología   | Uso                          |
| ------------ | ---------------------------- |
| Python       | Lenguaje principal           |
| Pandas       | Manipulación de datos        |
| NumPy        | Operaciones numéricas        |
| Requests     | Extracción desde NASA        |
| Matplotlib   | Visualización                |
| Seaborn      | Análisis gráfico             |
| Scikit-learn | Machine Learning             |
| Joblib       | Persistencia de modelos      |
| Streamlit    | Dashboard                    |
| Plotly       | Visualizaciones interactivas |
| Git / GitHub | Control de versiones         |

**Transformando datos astronómicos en conocimiento mediante Machine Learning.**
