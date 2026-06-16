import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Importamos las funciones desde utils.py
from utils import basic_stats, confidence_interval, conditional_prob, chi_square

# ──────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────
COL_MAP = {
    "Semestre que cursas actualmente": "Semestre",
    "Carrera": "Carrera",
    "¿Has usado ChatGPT o alguna IA generativa para actividades académicas?": "Uso_IA",
    "¿Con qué frecuencia usas ChatGPT o IA generativa para actividades académicas?": "Frecuencia",
    "Uso ChatGPT o IA generativa para resolver dudas académicas.": "Dudas",
    "Uso ChatGPT o IA generativa para comprender mejor temas difíciles.": "Comprension",
    "Uso ChatGPT o IA generativa para organizar tareas, trabajos o proyectos.": "Organizacion",
    "Uso ChatGPT o IA generativa para generar ideas, ejemplos o explicaciones relacionadas con mis materias.": "Ideas",
    "ChatGPT o la IA generativa me ayuda a mantenerme motivado cuando trabajo en actividades académicas.": "Motivado",
    "Usar ChatGPT o IA generativa me anima a seguir aprendiendo sobre temas que me interesan.": "Aprender",
    "ChatGPT o IA generativa me ayuda a sentir que puedo avanzar cuando una tarea me parece difícil.": "Avanzar",
    "Usar ChatGPT o IA generativa hace que algunas actividades académicas me resulten más interesantes.": "Interesante",
    "ChatGPT o IA generativa me ayuda a encontrar valor o utilidad personal en mis tareas académicas.": "Utilidad",
    "Cuando me siento bloqueado o desmotivado, ChatGPT o IA generativa me ayuda a retomar el trabajo académico.": "Retomar",
}

VAR_DESCRIPCIONES = {
    "Dudas":        "Resolución de dudas académicas con IA",
    "Comprension":  "Comprensión de temas difíciles con IA",
    "Organizacion": "Organización de tareas y proyectos con IA",
    "Ideas":        "Generación de ideas y ejemplos con IA",
    "Motivado":     "La IA ayuda a mantenerme motivado",
    "Aprender":     "La IA me anima a seguir aprendiendo",
    "Avanzar":      "La IA me ayuda a avanzar en tareas difíciles",
    "Interesante":  "La IA hace las actividades más interesantes",
    "Utilidad":     "La IA me ayuda a encontrar utilidad en mis tareas",
    "Retomar":      "La IA me ayuda a retomar el trabajo cuando estoy bloqueado",
}

EXPLICACIONES_TABLA = {
    "Media":               "Promedio aritmético de todas las respuestas (1–5). Indica el nivel general de acuerdo.",
    "Moda":                "Valor que más estudiantes eligieron. El más representativo de la opinión del grupo.",
    "Mediana":             "Valor central al ordenar todas las respuestas. Menos sensible a valores extremos.",
    "Varianza":            "Mide la dispersión al cuadrado. Valores altos = respuestas muy distintas entre sí.",
    "Desviación estándar": "Raíz de la varianza. Indica cuánto se alejan las respuestas del promedio.",
}

VARIABLES        = list(VAR_DESCRIPCIONES.keys())
VAR_LABELS       = list(VAR_DESCRIPCIONES.values())
VAR_LABEL_TO_KEY = {v: k for k, v in VAR_DESCRIPCIONES.items()}

SEMESTRES   = [f"{i} semestre" for i in range(1, 11)]
CARRERAS    = [
    "Ingeniería en Inteligencia Artificial",
    "Ingeniería en Sistemas Computacionales",
    "Ingeniería Matemática",
    "Ingeniería en Sistemas Ambientales",
    "Ingeniería en Ciencia de Datos",
    "Ingeniería Aeronáutica",
    "Mercadotecnia digital",
    "Medicina Veterinaria y Zootecnia",
    "Química",
    "Química farmacéutica",
    "QBP", "QFB", "LMD", "Mecánica", "Otra",
]
FRECUENCIAS = ["Nunca", "Raramente", "Algunas veces", "Frecuentemente", "Muy frecuentemente"]
LIKERT      = {1: "1 — Nunca", 2: "2 — Casi nunca", 3: "3 — A veces", 4: "4 — Casi siempre", 5: "5 — Siempre"}

PREGUNTAS = {
    "Dudas":        ("¿Con qué frecuencia usas la IA para resolver dudas académicas?", 0),
    "Comprension":  ("¿Usas la IA para comprender mejor temas que te parecen difíciles?", 0),
    "Organizacion": ("¿Usas la IA para organizar tareas, trabajos o proyectos?", 0),
    "Ideas":        ("¿Usas la IA para generar ideas o explicaciones de tus materias?", 0),
    "Motivado":     ("¿La IA te ayuda a mantenerte motivado en actividades académicas?", 1),
    "Aprender":     ("¿La IA te anima a seguir aprendiendo sobre temas que te interesan?", 1),
    "Avanzar":      ("¿La IA te ayuda a avanzar cuando una tarea te parece muy difícil?", 1),
    "Interesante":  ("¿La IA hace que algunas actividades académicas te resulten más interesantes?", 1),
    "Utilidad":     ("¿La IA te ayuda a encontrar valor o utilidad personal en tus tareas?", 1),
    "Retomar":      ("¿La IA te ayuda a retomar el trabajo cuando estás bloqueado o desmotivado?", 1),
}


# ──────────────────────────────────────────────────────────────────────
# PERFIL DE LA MUESTRA
# ──────────────────────────────────────────────────────────────────────
def render_perfil(df):
    n = len(df)
    if n == 0:
        st.warning("No hay datos para mostrar con los filtros actuales.")
        return

    st.markdown("### 👥 Perfil de la muestra")
    st.caption("Resumen de quiénes respondieron el cuestionario en esta selección.")

    c1, c2, c3, c4 = st.columns(4)
    usa_ia = (df["Uso_IA"].str.strip().str.lower() == "sí").sum() if "Uso_IA" in df.columns else 0
    pct_ia = usa_ia / n * 100 if n > 0 else 0
    n_sem  = df["Semestre"].nunique() if "Semestre" in df.columns else "—"
    n_car  = df["Carrera"].nunique()  if "Carrera"  in df.columns else "—"

    c1.metric("Total de respuestas", n)
    c2.metric("Usan IA generativa", f"{usa_ia} ({pct_ia:.0f}%)")
    c3.metric("Semestres representados", n_sem)
    c4.metric("Carreras representadas", n_car)

    col_a, col_b = st.columns(2)

    if "Semestre" in df.columns and not df["Semestre"].dropna().empty:
        with col_a:
            st.markdown("Respuestas por semestre")
            sem_counts = df["Semestre"].value_counts().sort_index()
            fig_s, ax_s = plt.subplots(figsize=(6, 3.5))
            ax_s.barh(sem_counts.index, sem_counts.values, color="#4C72B0")
            ax_s.set_xlabel("Número de respuestas")
            for i, v in enumerate(sem_counts.values):
                ax_s.text(v + 0.1, i, str(v), va="center", fontsize=9)
            st.pyplot(fig_s)
            plt.close(fig_s)

    if "Frecuencia" in df.columns and not df["Frecuencia"].dropna().empty:
        with col_b:
            st.markdown("Frecuencia de uso de IA")
            frec_counts = df["Frecuencia"].value_counts()
            orden = [f for f in FRECUENCIAS if f in frec_counts.index]
            frec_ord = frec_counts.reindex(orden)
            fig_f, ax_f = plt.subplots(figsize=(6, 3.5))
            colores = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c", "#2e7d32"]
            ax_f.bar(frec_ord.index, frec_ord.values, color=colores[:len(frec_ord)])
            ax_f.set_ylabel("Número de respuestas")
            plt.xticks(rotation=30, ha="right")
            for i, v in enumerate(frec_ord.values):
                ax_f.text(i, v + 0.1, str(v), ha="center", fontsize=9)
            st.pyplot(fig_f)
            plt.close(fig_f)

    if "Carrera" in df.columns:
        st.markdown("Respuestas por carrera")
        car_df = df["Carrera"].value_counts().reset_index()
        car_df.columns = ["Carrera", "Respuestas"]
        car_df["% del total"] = (car_df["Respuestas"] / n * 100).round(1).astype(str) + "%"
        
        # Centrar la tabla 
        col_espacio1, col_tabla, col_espacio2 = st.columns([1, 3, 1])
        with col_tabla:
            st.dataframe(car_df, hide_index=True, use_container_width=True)

    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# RESUMEN EJECUTIVO
# ──────────────────────────────────────────────────────────────────────
def render_resumen_ejecutivo(df):
    if len(df) == 0:
        return
        
    st.markdown("### 🗂️ Resumen ejecutivo — Comparación de variables")

    filas = []
    for var in VARIABLES:
        stats_data = basic_stats(df, var)
        ic = confidence_interval(df, var)
        filas.append({
            "Variable": VAR_DESCRIPCIONES[var],
            "Media": round(stats_data.get("Media", 0), 2),
            "Mediana": round(stats_data.get("Mediana", 0), 2),
            "Moda": round(stats_data.get("Moda", 0), 0),
            "Desv. estándar": round(stats_data.get("Desviación estándar", 0), 2),
            "IC 95% inferior": round(ic[0], 2),
            "IC 95% superior": round(ic[1], 2),
        })

    resumen_df = pd.DataFrame(filas)

    def color_media(val):
        if val >= 4:   return "background-color: #d4edda; color: #155724"
        elif val >= 3: return "background-color: #fff3cd; color: #856404"
        else:          return "background-color: #f8d7da; color: #721c24"

    col_espacio1, col_tabla, col_espacio2 = st.columns([1, 8, 1])
    with col_tabla:
        st.dataframe(resumen_df.style.map(color_media, subset=["Media"]), hide_index=True, use_container_width=True)
    st.caption("🟢 Media ≥ 4 (alta) · 🟡 Media entre 3 y 4 (moderada) · 🔴 Media < 3 (baja)")

    fig_r, ax_r = plt.subplots(figsize=(10, 5))
    etiquetas_cortas = [d[:35] + "…" if len(d) > 35 else d for d in resumen_df["Variable"]]
    colores_barras   = ["#2e7d32" if m >= 4 else "#f9a825" if m >= 3 else "#c62828" for m in resumen_df["Media"]]

    bars = ax_r.barh(etiquetas_cortas, resumen_df["Media"], color=colores_barras, edgecolor="white")
    ax_r.set_xlim(0, 5)
    ax_r.axvline(x=3, color="gray", linestyle="--", linewidth=1, label="Nivel neutro (3)")
    ax_r.set_xlabel("Media en escala Likert (1 = Nunca · 5 = Siempre)")
    ax_r.set_title("Comparación de medias por variable")

    for bar, val in zip(bars, resumen_df["Media"]):
        ax_r.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)

    leyenda = [mpatches.Patch(color="#2e7d32", label="Alta (≥ 4)"), mpatches.Patch(color="#f9a825", label="Moderada (3–4)"), mpatches.Patch(color="#c62828", label="Baja (< 3)")]
    ax_r.legend(handles=leyenda, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_r)
    plt.close(fig_r)

    var_mayor = resumen_df.loc[resumen_df["Media"].idxmax(), "Variable"]
    var_menor = resumen_df.loc[resumen_df["Media"].idxmin(), "Variable"]
    media_mayor = resumen_df["Media"].max()
    media_menor = resumen_df["Media"].min()
    
    st.info(
        f"Conclusión ejecutiva: La variable con mayor acuerdo fue {var_mayor} "
        f"(media = {media_mayor:.2f}) y la de menor acuerdo fue {var_menor} "
        f"(media = {media_menor:.2f})."
    )
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# ANÁLISIS POR VARIABLE
# ──────────────────────────────────────────────────────────────────────
def render_analysis(df, variable, key_suffix):
    desc = VAR_DESCRIPCIONES.get(variable, variable)
    data_num = pd.to_numeric(df[variable], errors="coerce").dropna()
    n = len(data_num)
    
    if n == 0:
        st.warning(f"No hay datos suficientes para analizar la variable '{desc}' con los filtros actuales.")
        return

    st.markdown(f"### 🔍 Análisis detallado: *{desc}*")

    estadisticas = basic_stats(df, variable)
    media  = estadisticas.get("Media", 0)
    moda   = estadisticas.get("Moda", 0)
    desv   = estadisticas.get("Desviación estándar", 0)

    filas_tabla = []
    for medida, valor in estadisticas.items():
        filas_tabla.append({
            "Medida estadística": medida,
            "Valor (escala 1–5)": round(valor, 4),
            "¿Qué significa?": EXPLICACIONES_TABLA.get(medida, ""),
        })
        
    col_espacio1, col_tabla, col_espacio2 = st.columns([1, 4, 1])
    with col_tabla:
        st.dataframe(pd.DataFrame(filas_tabla), hide_index=True, use_container_width=True)

    if media >= 4:
        nivel = "muy alto (≥ 4)"; interp = "la gran mayoría está muy de acuerdo con esta afirmación."
    elif media >= 3:
        nivel = "moderado-alto (3–4)"; interp = "los estudiantes tienden a estar de acuerdo, con cierta variabilidad."
    elif media >= 2:
        nivel = "bajo-moderado (2–3)"; interp = "las opiniones están divididas o la mayoría tiende a estar en desacuerdo."
    else:
        nivel = "bajo (< 2)"; interp = "la mayoría nunca experimenta esto o está en desacuerdo."

    dispersion = "poca variación" if desv < 1 else "alta variación"

    st.info(
        f"Conclusión estadística: La media de {media:.2f} sobre 5 indica un nivel de acuerdo {nivel}, "
        f"lo que sugiere que {interp} La moda fue {moda:.0f} y la desviación estándar de {desv:.2f} indica {dispersion}."
    )

    st.markdown("#### 📊 Histograma de respuestas")
    conteos  = data_num.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data_num, kde=True, ax=ax, bins=5, color="#4C72B0", edgecolor="white")
    ax.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Media = {media:.2f}")
    ax.set_xlabel("Valor en escala Likert (1 = Nunca · 5 = Siempre)")
    ax.set_ylabel("Número de respuestas")
    ax.set_title(f"Distribución de respuestas — {desc}")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\nNunca", "2\nCasi nunca", "3\nA veces", "4\nCasi siempre", "5\nSiempre"])
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    if "Semestre" in df.columns and df["Semestre"].nunique() > 1:
        st.markdown("#### 📅 Promedio por semestre")
        df2 = df.copy()
        df2[variable] = pd.to_numeric(df2[variable], errors="coerce")
        promedios = df2.groupby("Semestre")[variable].mean().sort_index()
        
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        orden = sorted(df2["Semestre"].dropna().unique())
        sns.barplot(x="Semestre", y=variable, data=df2, ax=ax3, order=orden, palette="Blues_d", errorbar="sd")
        ax3.axhline(media, color="red", linestyle="--", linewidth=1.2, label=f"Media global = {media:.2f}")
        ax3.set_xlabel("Semestre del estudiante")
        ax3.set_ylabel(f"Promedio de «{desc}» (1–5)")
        ax3.set_ylim(0, 5.5)
        ax3.legend()
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("---")
    if media >= 4:
        valoracion = f"muy positiva (media = {media:.2f})"
        cierre = "los estudiantes recurren activamente a la IA para esta función."
    elif media >= 3:
        valoracion = f"moderadamente positiva (media = {media:.2f})"
        cierre = "hay una percepción favorable aunque no generalizada."
    else:
        valoracion = f"baja (media = {media:.2f})"
        cierre = "los estudiantes no suelen usar la IA para esto."

    st.success(
        f"Conclusión general: La percepción de «{desc}» es {valoracion}. En conjunto, {cierre}"
    )
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# RESUMEN DE RESPUESTA DEL USUARIO
# ──────────────────────────────────────────────────────────────────────
def render_resumen_usuario(df_usuario):
    st.markdown("#### 📋 Resumen de tu respuesta")
    st.caption("Así quedaron registradas tus respuestas antes de analizarlas.")

    row = df_usuario.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semestre", row.get("Semestre", "—"))
    c2.metric("Usa IA", row.get("Uso_IA", "—"))
    c3.metric("Frecuencia", row.get("Frecuencia", "—"))
    c4.metric("Carrera", row.get("Carrera", "—")[:20] + ("…" if len(str(row.get("Carrera", ""))) > 20 else ""))

    filas = []
    for var in VARIABLES:
        val = row.get(var, "—")
        filas.append({
            "Variable": VAR_DESCRIPCIONES[var],
            "Tu respuesta (1–5)": val,
            "Significado": LIKERT.get(val, "—") if isinstance(val, int) else "—",
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────────────────────────────
tab_dataset, tab_form = st.tabs(["📂 Analizar dataset", "📝 Llenar formulario"])

# ── TAB 1: DATASET ──────────────────────────────────────────────────
with tab_dataset:
    st.markdown("Sube el archivo CSV exportado del cuestionario. Ahora puedes filtrar por semestre y seleccionar múltiples variables a la vez.")
    archivo = st.file_uploader("📁 Sube el archivo CSV del cuestionario", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)
        df = df.rename(columns=COL_MAP)

        st.markdown("### 🎛️ Filtros Globales")
        if "Semestre" in df.columns:
            semestres_disponibles = sorted(df["Semestre"].dropna().unique().tolist())
            semestres_seleccionados = st.multiselect(
                "Filtra la información por semestre (si dejas este espacio vacío, se analizará toda la población):",
                options=semestres_disponibles,
                default=[]
            )
            if semestres_seleccionados:
                df_filtrado = df[df["Semestre"].isin(semestres_seleccionados)]
            else:
                df_filtrado = df
        else:
            df_filtrado = df

        with st.expander(f"📄 Ver base de datos filtrada ({len(df_filtrado)} respuestas)", expanded=False):
            st.dataframe(df_filtrado, hide_index=True, use_container_width=True)

        render_perfil(df_filtrado)
        render_resumen_ejecutivo(df_filtrado)

        st.markdown("### 🔎 Análisis por variable(s)")
        labels_sel = st.multiselect(
            "Selecciona la variable o variables que quieres analizar en detalle:",
            options=VAR_LABELS,
            default=[VAR_LABELS[0]],
            key="var_dataset"
        )
        
        for label in labels_sel:
            variable = VAR_LABEL_TO_KEY[label]
            render_analysis(df_filtrado, variable, f"dataset_{variable}")

# ── TAB 2: FORMULARIO ───────────────────────────────────────────────
with tab_form:
    st.markdown("Completa el formulario con tu propia información para compararla con el resto.")
    
    with st.form("formulario_ia"):
        col1, col2 = st.columns(2)
        with col1:
            semestre = st.selectbox("¿En qué semestre estás?", SEMESTRES)
            carrera  = st.selectbox("¿Qué carrera estudias?", CARRERAS)
        with col2:
            uso_ia = st.radio("¿Has usado IA generativa?", ["Sí", "No"])
            frecuencia = st.selectbox("¿Con qué frecuencia usas IA generativa?", FRECUENCIAS)
        
        st.markdown("---")
        cols = st.columns(2)
        respuestas = {}
        for var, (texto, col_idx) in PREGUNTAS.items():
            with cols[col_idx]:
                respuestas[var] = st.select_slider(texto, options=list(LIKERT.keys()), format_func=lambda x: LIKERT[x], key=f"slider_{var}")
        
        submitted = st.form_submit_button("✅ Enviar mi respuesta")

    if submitted:
        fila = {"Semestre": semestre, "Carrera": carrera, "Uso_IA": uso_ia, "Frecuencia": frecuencia, **respuestas}
        st.session_state["respuesta_usuario"] = pd.DataFrame([fila])
        st.success("Respuesta registrada.")

    if "respuesta_usuario" in st.session_state:
        df_usuario = st.session_state["respuesta_usuario"]
        render_resumen_usuario(df_usuario)

        archivo2 = st.file_uploader(
            "📁 Opcional: sube el CSV del grupo para comparar tu respuesta con el resto",
            type=["csv"],
            key="csv_form"
        )

        fuente = st.radio(
            "¿Qué datos quieres analizar?",
            ["Solo mi respuesta", "Mi respuesta + el dataset del grupo"],
            horizontal=True
        )

        if fuente == "Mi respuesta + el dataset del grupo":
            if archivo2 is not None:
                df_grupo    = pd.read_csv(archivo2)
                df_grupo    = df_grupo.rename(columns=COL_MAP)
                df_analisis = pd.concat([df_grupo, df_usuario], ignore_index=True)
                st.info(f"Analizando {len(df_analisis)} respuestas en total ({len(df_grupo)} del dataset + la tuya).")
                render_perfil(df_analisis)
                render_resumen_ejecutivo(df_analisis)
            else:
                st.warning("Sube el dataset del grupo para poder combinarlo con tu respuesta.")
                df_analisis = df_usuario.copy()
        else:
            df_analisis = df_usuario.copy()
            st.info("Analizando solo tu respuesta.")

        st.markdown("### 🔎 Análisis por variable(s)")
        labels_sel2 = st.multiselect(
            "Selecciona la variable o variables que quieres analizar en detalle:",
            options=VAR_LABELS,
            default=[VAR_LABELS[0]],
            key="var_form"
        )
        
        for label in labels_sel2:
            variable2 = VAR_LABEL_TO_KEY[label]
            render_analysis(df_analisis, variable2, f"form_{variable2}")