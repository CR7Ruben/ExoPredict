import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="ExoPredict | NASA Exoplanet Intelligence",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "exopredict_model_optimizado.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "nasa_koi_clean.csv"
)


# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(37, 99, 235, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 10%,
            rgba(124, 58, 237, 0.13),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(14, 116, 144, 0.10),
            transparent 30%
        ),
        #050914;

    color: #F8FAFC;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #070B16;
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}

section[data-testid="stSidebar"] p {
    color: #94A3B8;
}

/* ================= TITULOS ================= */

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: #FFFFFF;
    margin-bottom: 0;
}

.hero-title span {
    color: #60A5FA;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #94A3B8;
    max-width: 900px;
    line-height: 1.7;
}

.nasa-line {
    height: 3px;
    width: 130px;
    background: linear-gradient(
        90deg,
        #2563EB,
        #60A5FA,
        transparent
    );

    margin: 22px 0 28px 0;
}

/* ================= CARDS ================= */

.card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(96, 165, 250, 0.15);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);
}

.card h3 {
    color: #F8FAFC;
    margin-top: 0;
}

.card p {
    color: #94A3B8;
    line-height: 1.65;
}

/* ================= MÉTRICAS ================= */

.metric-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #60A5FA;
}

.metric-label {
    color: #94A3B8;
    font-size: 0.85rem;
    margin-top: 5px;
}

/* ================= RESULTADO ================= */

.result-card {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    margin: 25px 0;
}

.result-title {
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 10px;
}

.result-description {
    color: #94A3B8;
    font-size: 1rem;
}

/* ================= BOTONES ================= */

.stButton > button {
    width: 100%;
    border-radius: 10px;

    border: 1px solid rgba(96, 165, 250, 0.4);

    background:
        linear-gradient(
            135deg,
            #1D4ED8,
            #2563EB
        );

    color: white;
    font-weight: 700;
    padding: 13px;
}

.stButton > button:hover {
    border-color: #60A5FA;

    box-shadow:
        0 0 20px rgba(37, 99, 235, 0.35);
}

/* ================= INPUTS ================= */

.stNumberInput input {
    background-color: #0F172A !important;
    color: white !important;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 0.8rem;
    padding: 45px 0 20px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():

    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_datos():

    return pd.read_csv(DATA_PATH)


try:

    modelo = cargar_modelo()

    datos = cargar_datos()

    modelo_disponible = True

except Exception as error:

    modelo_disponible = False

    modelo = None
    datos = None

    error_modelo = error


# ============================================================
# NOMBRES DE CLASES
# ============================================================

def obtener_nombre_clase(clase):

    """
    Convierte la clase numérica del modelo (0, 1, 2) a texto.

    El modelo fue entrenado con la columna 'target':
    0 = FALSE POSITIVE, 1 = CANDIDATE, 2 = CONFIRMED
    """

    mapa = {
        0: "FALSE POSITIVE",
        1: "CANDIDATE",
        2: "CONFIRMED"
    }

    return mapa.get(
        int(clase),
        str(clase)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
# 🪐 ExoPredict

### NASA Exoplanet Intelligence

Sistema de Machine Learning para apoyar
la clasificación de objetos candidatos
a exoplanetas.
""")

st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación",
    [
        "🌌 Inicio",
        "🔭 Predicción",
        "📊 Dashboard",
        "🧠 Modelo"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "NASA Exoplanet Archive\n"
    "Machine Learning • Random Forest"
)


# ============================================================
# INICIO
# ============================================================

if pagina == "🌌 Inicio":

    st.markdown(
        '<div class="hero-title">'
        'EXO<span>PREDICT</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nasa-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">

        Inteligencia artificial aplicada a la exploración
        de exoplanetas.

        ExoPredict analiza características observacionales
        para estimar si una señal corresponde a un planeta
        confirmado, un candidato o un posible falso positivo.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    col1, col2 = st.columns([1.4, 1])

    # --------------------------------------------------------
    # PRESENTACIÓN
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>
            🚀 Explorando mundos más allá del Sistema Solar
            </h3>

            <p>
            ExoPredict utiliza Machine Learning para analizar
            datos astronómicos provenientes del
            NASA Exoplanet Archive.
            </p>

            <p>
            El sistema estudia características como el radio
            planetario, periodo orbital, temperatura,
            profundidad del tránsito y relación señal/ruido.
            </p>

            <p>
            Su objetivo es servir como herramienta de apoyo
            para analizar grandes cantidades de observaciones.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # MISIÓN
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🛰️ Misión ExoPredict</h3>

            <p>
            Convertir datos astronómicos complejos en
            información comprensible mediante inteligencia
            artificial.
            </p>

            <p>
            El sistema clasifica cada observación en tres
            categorías:
            </p>

            <p>
            🪐 <b>CONFIRMED</b><br>
            🔭 <b>CANDIDATE</b><br>
            ⚠️ <b>FALSE POSITIVE</b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # BOTÓN
    # --------------------------------------------------------

    if st.button("🔭 Iniciar Análisis"):

        st.info(
            "Selecciona 🔭 Predicción en el menú lateral "
            "para analizar un objeto."
        )

    # --------------------------------------------------------
    # INFORMACIÓN DEL DATASET
    # --------------------------------------------------------

    if datos is not None:

        st.write("")

        total = len(datos)

        confirmed = (
            datos["koi_disposition"]
            .eq("CONFIRMED")
            .sum()
        )

        candidate = (
            datos["koi_disposition"]
            .eq("CANDIDATE")
            .sum()
        )

        false_positive = (
            datos["koi_disposition"]
            .eq("FALSE POSITIVE")
            .sum()
        )

        st.markdown(
            "### 🌌 Datos de la misión",
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {total:,}
                </div>

                <div class="metric-label">
                Observaciones
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {confirmed:,}
                </div>

                <div class="metric-label">
                Confirmados
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {candidate:,}
                </div>

                <div class="metric-label">
                Candidatos
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-value">
                {false_positive:,}
                </div>

                <div class="metric-label">
                Falsos positivos
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PREDICCIÓN
# ============================================================

elif pagina == "🔭 Predicción":

    st.markdown(
        '<div class="hero-title">'
        '🔭 PREDICCIÓN'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nasa-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">

        Introduce las características observacionales
        del objeto que deseas analizar.

        El modelo generará una clasificación y mostrará
        las probabilidades estimadas.

        </div>
        """,
        unsafe_allow_html=True
    )

    if not modelo_disponible:

        st.error(
            "No se pudo cargar el modelo."
        )

        st.code(
            str(error_modelo)
        )

    else:

        st.write("")

        col1, col2 = st.columns(2)

        # ====================================================
        # CARACTERÍSTICAS PLANETARIAS
        # ====================================================

        with col1:

            st.markdown(
                """
                <div class="card">

                <h3>🪐 Características del objeto</h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            koi_period = st.number_input(
                "Periodo orbital (días)",
                min_value=0.01,
                value=10.0,
                step=0.1
            )

            koi_duration = st.number_input(
                "Duración del tránsito (horas)",
                min_value=0.01,
                value=5.0,
                step=0.1
            )

            koi_depth = st.number_input(
                "Profundidad del tránsito",
                min_value=0.0,
                value=450.0,
                step=10.0
            )

            koi_prad = st.number_input(
                "Radio planetario (R⊕)",
                min_value=0.01,
                value=2.5,
                step=0.1
            )

            koi_impact = st.number_input(
                "Parámetro de impacto",
                min_value=0.0,
                value=0.5,
                step=0.01
            )

            koi_model_snr = st.number_input(
                "Relación señal/ruido (SNR)",
                min_value=0.01,
                value=30.0,
                step=1.0
            )

        # ====================================================
        # CARACTERÍSTICAS ESTELARES
        # ====================================================

        with col2:

            st.markdown(
                """
                <div class="card">

                <h3>🌟 Características de la estrella</h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            koi_teq = st.number_input(
                "Temperatura de equilibrio (K)",
                min_value=0.0,
                value=900.0,
                step=10.0
            )

            koi_insol = st.number_input(
                "Insolación",
                min_value=0.0,
                value=180.0,
                step=10.0
            )

            koi_steff = st.number_input(
                "Temperatura estelar (K)",
                min_value=0.0,
                value=5760.0,
                step=50.0
            )

            koi_slogg = st.number_input(
                "Gravedad superficial estelar",
                min_value=0.0,
                value=4.4,
                step=0.1
            )

            koi_srad = st.number_input(
                "Radio estelar (R☉)",
                min_value=0.01,
                value=1.0,
                step=0.1
            )

        st.write("")

        # ====================================================
        # BOTÓN DE PREDICCIÓN
        # ====================================================

        if st.button("🚀 ANALIZAR OBJETO"):

            entrada = pd.DataFrame([
                {
                    "koi_period": koi_period,
                    "koi_duration": koi_duration,
                    "koi_depth": koi_depth,
                    "koi_prad": koi_prad,
                    "koi_teq": koi_teq,
                    "koi_insol": koi_insol,
                    "koi_impact": koi_impact,
                    "koi_model_snr": koi_model_snr,
                    "koi_steff": koi_steff,
                    "koi_slogg": koi_slogg,
                    "koi_srad": koi_srad
                }
            ])

            try:

                # --------------------------------------------
                # PREDICCIÓN
                # --------------------------------------------

                prediccion = modelo.predict(
                    entrada
                )[0]

                probabilidades = modelo.predict_proba(
                    entrada
                )[0]

                clases = modelo.classes_

                nombre_prediccion = (
                    obtener_nombre_clase(prediccion)
                )

                # --------------------------------------------
                # RESULTADO
                # --------------------------------------------

                if nombre_prediccion == "CONFIRMED":

                    emoji = "🪐"

                    titulo = (
                        "EXOPLANETA CONFIRMADO"
                    )

                    descripcion = (
                        "El modelo identifica características "
                        "compatibles con un exoplaneta confirmado."
                    )

                elif nombre_prediccion == "CANDIDATE":

                    emoji = "🔭"

                    titulo = (
                        "CANDIDATO A EXOPLANETA"
                    )

                    descripcion = (
                        "El objeto presenta características "
                        "compatibles con un posible exoplaneta, "
                        "pero requiere mayor análisis."
                    )

                else:

                    emoji = "⚠️"

                    titulo = (
                        "POSIBLE FALSO POSITIVO"
                    )

                    descripcion = (
                        "Las características observadas "
                        "son más compatibles con una señal "
                        "que no corresponde a un exoplaneta."
                    )

                # --------------------------------------------
                # TARJETA RESULTADO
                # --------------------------------------------

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-title">

                    {emoji} {titulo}

                    </div>

                    <div class="result-description">

                    {descripcion}

                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # --------------------------------------------
                # PROBABILIDADES
                # --------------------------------------------

                resultados = []

                for clase, probabilidad in zip(
                    clases,
                    probabilidades
                ):

                    resultados.append(
                        {
                            "Clasificación":
                                obtener_nombre_clase(clase),

                            "Probabilidad":
                                probabilidad
                        }
                    )

                prob_df = pd.DataFrame(
                    resultados
                )

                st.subheader(
                    "📊 Probabilidades de clasificación"
                )

                fig = px.bar(
                    prob_df,
                    x="Clasificación",
                    y="Probabilidad",
                    text_auto=".1%"
                )

                fig.update_layout(
                    template="plotly_dark",

                    yaxis=dict(
                        tickformat=".0%",
                        range=[0, 1]
                    ),

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # --------------------------------------------
                # DATOS UTILIZADOS
                # --------------------------------------------

                st.subheader(
                    "🔬 Datos utilizados para la predicción"
                )

                st.dataframe(
                    entrada,
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------------
                # INTERPRETACIÓN
                # --------------------------------------------

                st.subheader(
                    "🧠 Interpretación"
                )

                prob_maxima = max(
                    probabilidades
                )

                st.info(
                    f"""
                    ExoPredict clasificó el objeto como
                    **{nombre_prediccion}** con una probabilidad
                    estimada de **{prob_maxima:.1%}**.

                    Esta predicción representa el resultado
                    del modelo de Machine Learning y no sustituye
                    la validación astronómica científica.
                    """
                )

            except Exception as error:

                st.error(
                    "Ocurrió un error durante la predicción."
                )

                st.code(
                    str(error)
                )


# ============================================================
# DASHBOARD
# ============================================================

elif pagina == "📊 Dashboard":

    st.markdown(
        '<div class="hero-title">'
        '📊 DASHBOARD'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nasa-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">

        Esta sección muestra el comportamiento general
        del dataset utilizado para entrenar ExoPredict.

        Estas gráficas representan los datos históricos
        de NASA, no una predicción individual.

        </div>
        """,
        unsafe_allow_html=True
    )

    if datos is None:

        st.error(
            "No se pudo cargar el dataset."
        )

    else:

        st.write("")

        # --------------------------------------------
        # DISTRIBUCIÓN
        # --------------------------------------------

        distribucion = (
            datos["koi_disposition"]
            .value_counts()
            .reset_index()
        )

        distribucion.columns = [
            "Clasificación",
            "Cantidad"
        ]

        fig = px.bar(
            distribucion,
            x="Clasificación",
            y="Cantidad",
            text="Cantidad",
            title="Distribución de clasificaciones"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------
        # GRÁFICAS
        # --------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                datos,
                x="koi_prad",
                color="koi_disposition",
                title="Distribución del radio planetario",
                nbins=50
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.histogram(
                datos,
                x="koi_teq",
                color="koi_disposition",
                title="Distribución de temperatura",
                nbins=50
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# MODELO
# ============================================================

elif pagina == "🧠 Modelo":

    st.markdown(
        '<div class="hero-title">'
        '🧠 MODELO'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nasa-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">

        Rendimiento del Random Forest optimizado
        utilizado por ExoPredict.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------
    # MÉTRICAS
    # --------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-value">
            79.83%
            </div>

            <div class="metric-label">
            Accuracy
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-value">
            80.25%
            </div>

            <div class="metric-label">
            Precision
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-value">
            79.83%
            </div>

            <div class="metric-label">
            Recall
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-value">
            92.70%
            </div>

            <div class="metric-label">
            ROC-AUC
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------
    # INFORMACIÓN
    # --------------------------------------------

    st.markdown(
        """
        <div class="card">

        <h3>🏆 Random Forest Optimizado</h3>

        <p>
        El modelo fue optimizado mediante búsqueda
        de hiperparámetros y validación cruzada.
        </p>

        <p>
        Configuración seleccionada:
        </p>

        <p>
        • 400 árboles<br>
        • Profundidad máxima: 20<br>
        • Mínimo de muestras para dividir: 15<br>
        • Mínimo de muestras por hoja: 2<br>
        • max_features: log2
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------
    # IMPORTANCIA
    # --------------------------------------------

    importancia = pd.DataFrame({

        "Variable": [
            "koi_model_snr",
            "koi_prad",
            "koi_depth",
            "koi_impact",
            "koi_period",
            "koi_duration",
            "koi_insol",
            "koi_teq",
            "koi_steff",
            "koi_srad",
            "koi_slogg"
        ],

        "Importancia": [
            0.2455,
            0.1605,
            0.1041,
            0.0862,
            0.0841,
            0.0833,
            0.0622,
            0.0613,
            0.0419,
            0.0358,
            0.0352
        ]
    })

    fig = px.bar(
        importancia.sort_values(
            "Importancia"
        ),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de variables"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    ExoPredict • NASA Exoplanet Intelligence<br>

    Machine Learning aplicado a la clasificación
    de exoplanetas

    </div>
    """,
    unsafe_allow_html=True
)