# 🔭 ExoPredict

### Sistema de Machine Learning para la clasificación de candidatos a exoplanetas

ExoPredict es un proyecto de **extracción de conocimiento mediante Machine Learning** desarrollado a partir de datos reales del **NASA Exoplanet Archive**.

El sistema implementa un flujo completo de **ETL, análisis exploratorio, entrenamiento, evaluación y optimización de modelos de aprendizaje supervisado**, con el objetivo de clasificar objetos observados por la misión Kepler en tres categorías:

* 🪐 **CONFIRMED** — Exoplaneta confirmado.
* 🔎 **CANDIDATE** — Candidato a exoplaneta.
* ❌ **FALSE POSITIVE** — Falso positivo.

---

## 🎯 Objetivo del proyecto

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

# 🛰️ Fuente de datos

Los datos utilizados provienen del **NASA Exoplanet Archive**, repositorio público especializado en información sobre exoplanetas y candidatos detectados por diferentes misiones astronómicas.

La información utilizada corresponde principalmente a datos de candidatos observados por la misión **Kepler**.

---

# 🏗️ Arquitectura del proyecto

El proyecto está organizado siguiendo un flujo de procesamiento de datos:

```text
                    NASA Exoplanet Archive
                              │
                              ▼
                       📥 EXTRACCIÓN
                              │
                              ▼
                    📄 Dataset RAW
                              │
                              ▼
                         🧹 ETL
                    Limpieza y transformación
                              │
                              ▼
                    📊 Dataset procesado
                              │
                              ▼
                   🔎 ANÁLISIS EXPLORATORIO
                              │
                              ▼
                     🤖 MACHINE LEARNING
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       Regresión Logística          Random Forest
                                             │
                                             ▼
                                   ⚙️ Optimización
                                             │
                                             ▼
                                      🏆 Modelo final
                                             │
                                             ▼
                                  📊 Dashboard interactivo
```

---

# 📂 Estructura del proyecto

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
│
├── README.md
└── requirements.txt
```

---

# 📥 Fase 1 — Extracción de datos

Los datos son obtenidos directamente desde el **NASA Exoplanet Archive** mediante una solicitud HTTP utilizando Python.

El proceso de extracción genera un dataset RAW que conserva la información original antes de aplicar transformaciones.

### Resultado actual

```text
Registros obtenidos: 8054
Variables iniciales: 20
```

La variable objetivo utilizada es:

```text
koi_disposition
```

Con las siguientes categorías:

```text
FALSE POSITIVE
CONFIRMED
CANDIDATE
```

---

# 🧹 Fase 2 — ETL

El proceso ETL realiza las siguientes operaciones:

### Extracción

Obtención de información desde la fuente de datos de NASA.

### Transformación

Se realizan diferentes procesos de preparación:

* Eliminación de registros duplicados.
* Identificación de valores nulos.
* Imputación de valores numéricos mediante la **mediana**.
* Validación de valores inválidos.
* Selección de variables relevantes.
* Preparación de la variable objetivo.
* Validación final del dataset.

### Carga

El dataset limpio se almacena en:

```text
data/processed/nasa_koi_clean.csv
```

### Resultado

```text
Registros iniciales: 8054
Registros finales:   8054

Duplicados:          0
Valores nulos:       0
Variables finales:   14
```

Durante el ETL se detectaron y trataron **531 valores nulos**, utilizando la mediana de cada variable numérica correspondiente.

---

# 🔎 Fase 2.1 — Análisis exploratorio

Se realizó un análisis exploratorio para identificar patrones y relaciones entre las variables astronómicas.

Se generaron visualizaciones relacionadas con:

* Distribución de las clases.
* Radio planetario.
* Temperatura de equilibrio.
* Periodo orbital.
* Relación entre radio y temperatura.
* Relación señal/ruido.
* Correlación entre variables.

Algunos de los factores con mayor relevancia posteriormente para el modelo fueron:

```text
koi_model_snr
koi_prad
koi_depth
koi_impact
koi_period
koi_duration
```

---

# 🤖 Fase 3 — Machine Learning

Se implementaron dos algoritmos supervisados para comparar su rendimiento:

### 1. Regresión Logística

Modelo utilizado como referencia para establecer una línea base.

### 2. Random Forest

Modelo basado en múltiples árboles de decisión, seleccionado debido a su capacidad para modelar relaciones no lineales entre las características astronómicas.

---

# 📊 Resultados iniciales

| Modelo              | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Regresión Logística |   69.77% |    62.61% | 69.77% | 64.33% |  83.96% |
| Random Forest       |   79.21% |    79.83% | 79.21% | 79.35% |  92.19% |

El **Random Forest** presentó un mejor desempeño general.

---

# ⚙️ Optimización del modelo

Posteriormente se realizó una búsqueda de hiperparámetros utilizando:

```text
RandomizedSearchCV
```

con:

```text
5-fold Cross Validation
```

Se evaluaron diferentes configuraciones de:

* Número de árboles.
* Profundidad máxima.
* Número mínimo de muestras para dividir un nodo.
* Número mínimo de muestras por hoja.
* Número de características utilizadas en cada división.

### Mejores hiperparámetros encontrados

```text
n_estimators      = 400
min_samples_split = 15
min_samples_leaf  = 2
max_features      = log2
max_depth         = 20
```

---

# 🏆 Resultado del modelo optimizado

| Métrica   |  Resultado |
| --------- | ---------: |
| Accuracy  | **79.83%** |
| Precision | **80.25%** |
| Recall    | **79.83%** |
| F1-Score  | **79.95%** |
| ROC-AUC   | **92.70%** |

El modelo optimizado presentó una mejora respecto al Random Forest inicial.

### Comparación

```text
F1-Score

Random Forest base       79.35%
Random Forest optimizado 79.95%

Mejora                   +0.60 puntos porcentuales
```

---

# 🔬 Importancia de variables

El modelo optimizado identificó las siguientes variables como las más relevantes:

| Variable        | Importancia |
| --------------- | ----------: |
| `koi_model_snr` |      24.55% |
| `koi_prad`      |      16.05% |
| `koi_depth`     |      10.41% |
| `koi_impact`    |       8.62% |
| `koi_period`    |       8.41% |
| `koi_duration`  |       8.33% |
| `koi_insol`     |       6.22% |
| `koi_teq`       |       6.13% |
| `koi_steff`     |       4.19% |
| `koi_srad`      |       3.58% |
| `koi_slogg`     |       3.52% |

La variable con mayor importancia fue:

```text
koi_model_snr
```

con aproximadamente **24.55% de importancia** dentro del modelo.

---

# 🎯 Desempeño por clase

El modelo optimizado obtuvo:

| Clase          | Precision | Recall | F1-Score |
| -------------- | --------: | -----: | -------: |
| FALSE POSITIVE |       89% |    83% |  **86%** |
| CANDIDATE      |       54% |    57% |  **55%** |
| CONFIRMED      |       81% |    87% |  **84%** |

Se observa que la categoría **CANDIDATE** representa el principal reto de clasificación.

Como siguiente etapa se analizarán estrategias de ponderación de clases con el objetivo de mejorar el equilibrio entre las tres categorías.

---

# 🔢 Matriz de confusión

El modelo optimizado obtuvo:

```text
                 Predicción
                 FP   CAND  CONF

Real FP          658   79    56

Real CAND         62  154    56

Real CONF         22   50   474
```

---

# 📊 Fase 4 — Dashboard

Como etapa final se desarrollará un dashboard interactivo utilizando:

* Python
* Streamlit
* Plotly
* Pandas
* Scikit-learn

El dashboard permitirá visualizar:

* Cantidad de objetos analizados.
* Distribución de las categorías.
* Métricas del modelo.
* Matriz de confusión.
* Importancia de variables.
* Principales características de los candidatos.
* Predicciones realizadas por el modelo.

> 🚧 **Estado:** Dashboard en desarrollo.

---

# 🛠️ Tecnologías utilizadas

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

---

# ▶️ Instalación

Clonar el repositorio:

```bash
git clone TU_REPOSITORIO
```

Entrar al proyecto:

```bash
cd ExoPredict
```

Instalar dependencias:

```bash
py -m pip install -r requirements.txt
```

---

# ▶️ Ejecución

### 1. Extraer datos

```bash
py src/extraer_datos.py
```

### 2. Ejecutar ETL

```bash
py src/etl.py
```

### 3. Ejecutar análisis exploratorio

```bash
py src/explorar_datos.py
```

### 4. Entrenar modelos

```bash
py src/entrenar_modelo.py
```

### 5. Optimizar Random Forest

```bash
py src/optimizar_modelo.py
```

### 6. Ejecutar dashboard

```bash
py -m streamlit run app.py
```

---

# 📌 Estado actual

```text
[████████████████████] 100%  Extracción
[████████████████████] 100%  ETL
[████████████████████] 100%  Análisis exploratorio
[████████████████████] 100%  Modelos iniciales
[████████████████████] 100%  Optimización
[██████████----------]  50%  Balanceo de clases
[--------------------]   0%  Dashboard final
```

---

# 👨‍💻 Equipo

**Proyecto académico — Desarrollo de Software**

Proyecto realizado como parte de la aplicación de técnicas de:

> Extracción de conocimiento, ETL, Machine Learning y visualización de datos.

---

# 🚀 Próximos pasos

* [ ] Optimizar el tratamiento de la clase `CANDIDATE`.
* [ ] Comparar diferentes estrategias de ponderación de clases.
* [ ] Seleccionar el modelo definitivo.
* [ ] Desarrollar dashboard interactivo.
* [ ] Integrar módulo de predicción.
* [ ] Documentar resultados finales.
* [ ] Preparar presentación del proyecto.

---

## 🔭 ExoPredict

**Transformando datos astronómicos en conocimiento mediante Machine Learning.**
