import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from utils import basic_stats, confidence_interval, conditional_prob, chi_square

# ──────────────────────────────────────────────────────────────────────
# CONSTANTES Y DICCIONARIOS
# ──────────────────────────────────────────────────────────────────────
COL_MAP = {
    "Semestre que cursas actualmente": "Ciclo",
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
    "Dudas":        "Resolucion de dudas",
    "Comprension":  "Comprension de temas complejos",
    "Organizacion": "Organizacion de tareas",
    "Ideas":        "Generacion de ejemplos",
    "Motivado":     "Mantenimiento de motivacion",
    "Aprender":     "Estimulo para aprendizaje continuo",
    "Avanzar":      "Desbloqueo en tareas dificiles",
    "Interesante":  "Aumento de interes en la materia",
    "Utilidad":     "Busqueda de utilidad practica",
    "Retomar":      "Reactivacion tras bloqueo mental",
}

EXPLICACIONES_TABLA = {
    "Media":               "Promedio aritmetico. Indica la tendencia central del grupo.",
    "Moda":                "Valor mas repetido. Señala la respuesta mayoritaria.",
    "Mediana":             "Valor intermedio que divide la distribucion en dos mitades.",
    "Varianza":            "Dispersion de los datos. Valores altos indican opiniones divididas.",
    "Desviación estándar": "Distancia promedio respecto a la media. Evalua la consistencia del grupo.",
}

VARIABLES        = list(VAR_DESCRIPCIONES.keys())
VAR_LABELS       = list(VAR_DESCRIPCIONES.values())
VAR_LABEL_TO_KEY = {v: k for k, v in VAR_DESCRIPCIONES.items()}

NIVELES     = [f"{i} ciclo" for i in range(1, 11)]
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
LIKERT      = {1: "1 - Nunca", 2: "2 - Casi nunca", 3: "3 - A veces", 4: "4 - Casi siempre", 5: "5 - Siempre"}

PREGUNTAS = {
    "Dudas":        ("Con que frecuencia usas la IA para resolver dudas academicas?", 0),
    "Comprension":  ("Usas la IA para comprender mejor temas que te parecen dificiles?", 0),
    "Organizacion": ("Usas la IA para organizar tareas, trabajos o proyectos?", 0),
    "Ideas":        ("Usas la IA para generar ideas o explicaciones de tus materias?", 0),
    "Motivado":     ("La IA te ayuda a mantenerte motivado en actividades academicas?", 1),
    "Aprender":     ("La IA te anima a seguir aprendiendo sobre temas que te interesan?", 1),
    "Avanzar":      ("La IA te ayuda a avanzar cuando una tarea te parece muy dificil?", 1),
    "Interesante":  ("La IA hace que algunas actividades academicas te resulten mas interesantes?", 1),
    "Utilidad":     ("La IA te ayuda a encontrar valor o utilidad personal en tus tareas?", 1),
    "Retomar":      ("La IA te ayuda a retomar el trabajo cuando estas bloqueado o desmotivado?", 1),
}


# ──────────────────────────────────────────────────────────────────────
# FUNCIONES DE ANALISIS POBLACIONAL (DATASET GLOBAL)
# ──────────────────────────────────────────────────────────────────────
def render_perfil(df):
    n = len(df)
    if n == 0:
        st.warning("Datos insuficientes para los parametros actuales.")
        return

    with st.expander("Ver Base de Datos Completa", expanded=False):
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("### Demografia de la Poblacion")
    st.caption("Metricas base de los participantes.")
    
    c1, c2, c3, c4 = st.columns(4)
    usa_ia = df["Uso_IA"].apply(lambda x: f"{x}".strip().lower() == "si").sum() if "Uso_IA" in df.columns else 0
    pct_ia = usa_ia / n * 100 if n > 0 else 0
    n_ciclos = df["Ciclo"].nunique() if "Ciclo" in df.columns else "—"
    n_car  = df["Carrera"].nunique()  if "Carrera"  in df.columns else "—"

    c1.metric("Total de encuestas", n)
    c2.metric("Adopcion de IA", f"{usa_ia} ({pct_ia:.0f}%)")
    c3.metric("Ciclos distintos", n_ciclos)
    c4.metric("Carreras distintas", n_car)
    st.markdown("---")

def render_resumen_ejecutivo(df):
    if len(df) == 0:
        return
        
    st.markdown("### Resumen Directivo — Tendencias Generales")
    st.caption("Metricas consolidadas de las 10 variables de evaluacion.")

    filas = []
    for var in VARIABLES:
        metricas_calculadas = basic_stats(df, var)
        ic = confidence_interval(df, var)
        filas.append({
            "Metrica Evaluada": VAR_DESCRIPCIONES[var],
            "Media": round(metricas_calculadas.get("Media", 0), 2),
            "Mediana": round(metricas_calculadas.get("Mediana", 0), 2),
            "Moda": round(metricas_calculadas.get("Moda", 0), 0),
            "Varianza": round(metricas_calculadas.get("Varianza", 0), 2),
            "Desv. Est.": round(metricas_calculadas.get("Desviación estándar", 0), 2),
            "Limite Inf (95%)": round(ic[0], 2),
            "Limite Sup (95%)": round(ic[1], 2),
        })

    resumen_df = pd.DataFrame(filas)

    def color_media(val):
        if val >= 4:   return "background-color: #d4edda; color: #155724"
        elif val >= 3: return "background-color: #fff3cd; color: #856404"
        else:          return "background-color: #f8d7da; color: #721c24"

    st.dataframe(resumen_df.style.map(color_media, subset=["Media", "Mediana", "Moda"]), hide_index=True, use_container_width=True)
    st.caption("Integracion alta (>= 4) - Uso moderado (3-4) - Uso bajo (< 3)")
    st.info("Nota metodologica: El semaforo de colores se aplica unicamente a la Media, Mediana y Moda porque representan la escala Likert (1 a 5). La Varianza y Desviacion Estandar miden dispersion de probabilidad, por lo que mantienen un formato neutral.")
    st.markdown("---")

def render_prueba_hipotesis(df):
    st.markdown("### Inferencia de Probabilidad (Prueba de Hipotesis)")
    st.caption("Evaluacion matematica para confirmar si un factor determina el comportamiento de otro.")
    
    opciones_hipotesis = ["Ciclo", "Carrera", "Uso_IA", "Frecuencia"]
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        var_indep = st.selectbox("Variable A (Factor de influencia):", opciones_hipotesis, index=1)
    with col_sel2:
        var_dep = st.selectbox("Variable B (Factor afectado):", opciones_hipotesis, index=3)
        
    if var_indep == var_dep:
        st.warning("Seleccione variables categoricas distintas para ejecutar la matriz de contingencia.")
        st.markdown("---")
        return
        
    st.markdown(
        "**Planteamiento de Probabilidad:**\n"
        f"* **H0 (Hipotesis Nula):** No hay relacion. La variable **{var_indep}** NO afecta a la variable **{var_dep}**. Son eventos matematicamente independientes.\n"
        f"* **Ha (Hipotesis Alternativa):** Si hay relacion. La variable **{var_indep}** influye directamente en los resultados de **{var_dep}**."
    )
    
    if var_indep in df.columns and var_dep in df.columns:
        chi2_val, p_val, dof, _ = chi_square(df, var_indep, var_dep)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi-Cuadrada calculada", f"{chi2_val:.2f}")
        c2.metric("Valor p (p-value)", f"{p_val:.4f}")
        c3.metric("Grados de Libertad", f"{dof}")
        
        if dof == 0:
            st.warning("Error de Varianza: Los Grados de Libertad son 0. La poblacion es homogenea y carece de diversidad para calcular una correlacion.")
        elif p_val < 0.05:
            st.success(f"**Resultado:** Se RECHAZA la Hipotesis Nula (H0). Los datos prueban que **{var_indep}** SI afecta el comportamiento de **{var_dep}**.")
        else:
            st.info(f"**Resultado:** NO se rechaza la Hipotesis Nula (H0). No existe evidencia suficiente para afirmar que **{var_indep}** influye en **{var_dep}**.")
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# ANALISIS GRUPAL DETALLADO
# ──────────────────────────────────────────────────────────────────────
def render_analisis_grupo_variable(df, variable):
    desc = VAR_DESCRIPCIONES.get(variable, variable)
    data_num = pd.to_numeric(df[variable], errors="coerce").dropna()
    n = len(data_num)
    
    if n == 0: return

    st.markdown(f"### Exploracion Visual Poblacional: {desc}")

    metricas_calculadas = basic_stats(df, variable)
    media  = metricas_calculadas.get("Media", 0)
    moda   = metricas_calculadas.get("Moda", 0)
    desv   = metricas_calculadas.get("Desviación estándar", 0)

    filas_tabla = []
    for medida, valor in metricas_calculadas.items():
        filas_tabla.append({
            "Metrica calculada": medida,
            "Valor": round(valor, 4),
            "Interpretacion": EXPLICACIONES_TABLA.get(medida, ""),
        })
        
    st.dataframe(pd.DataFrame(filas_tabla), hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data_num, kde=True, ax=ax, bins=5, color="#4C72B0", edgecolor="white")
    ax.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Media = {media:.2f}")
    ax.set_xlabel("Escala Likert (1 = Nunca - 5 = Siempre)")
    ax.set_ylabel("Frecuencia de respuestas")
    ax.set_title(f"Distribucion del Grupo — {desc}")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\nNunca", "2\nCasi nunca", "3\nA veces", "4\nCasi siempre", "5\nSiempre"])
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**Conclusiones de la Distribucion:**\n"
        f"La concentracion de respuestas en la opcion {moda:.0f} y la media de {media:.2f} marcan la tendencia central de la poblacion analizada. "
        f"Una dispersion de {desv:.2f} define el nivel de consenso: desviaciones bajas reflejan acuerdos uniformes, mientras que desviaciones altas evidencian una marcada polarizacion de posturas dentro del grupo."
    )

    if "Ciclo" in df.columns and df["Ciclo"].nunique() > 1:
        st.markdown("#### Comportamiento Historico por Ciclo")
        df2 = df.copy()
        df2[variable] = pd.to_numeric(df2[variable], errors="coerce")
        
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        orden = sorted(df2["Ciclo"].dropna().unique())
        sns.barplot(x="Ciclo", y=variable, data=df2, ax=ax3, order=orden, palette="Blues_d", errorbar="sd")
        ax3.axhline(media, color="red", linestyle="--", linewidth=1.2, label=f"Promedio global = {media:.2f}")
        ax3.set_xlabel("Ciclo")
        ax3.set_ylabel(f"Promedio (1-5)")
        ax3.set_ylim(0, 5.5)
        ax3.legend()
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig3)
        plt.close(fig3)
        
        st.info(
            "**Conclusiones del Historico:**\n"
            "Esta visualizacion permite aislar el factor de la experiencia academica. Las fluctuaciones o estabilidades entre los ciclos de formacion inicial y terminal revelan si el nivel de exigencia tecnica altera sistemicamente la adopcion de estas herramientas operativas."
        )
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────
# ANALISIS INDIVIDUAL: COMPARATIVAS Y AUDITORIA DE NIVEL TEXTUAL
# ──────────────────────────────────────────────────────────────────────
def render_resumen_usuario(df_usuario, df_grupo=None):
    st.markdown("#### Analisis Comparativo de tu Perfil")
    st.caption("Evaluacion de tus respuestas en contraste con las metricas acumuladas del grupo.")
    
    row = df_usuario.iloc[0]
    filas_tabla = []
    
    for var in VARIABLES:
        val_num = int(row.get(var, 3))
        significado_likert = LIKERT.get(val_num, "—")
        
        if df_grupo is not None and not df_grupo.empty:
            respuestas_grupo = pd.to_numeric(df_grupo[var], errors="coerce").dropna()
            media_g = respuestas_grupo.mean()
            diff = val_num - media_g
            
            coincidencias = (respuestas_grupo == val_num).sum()
            porcentaje = (coincidencias / len(respuestas_grupo)) * 100
            if porcentaje > 50:
                contexto = "Mayoria"
                por_que_contexto = f"El {porcentaje:.1f}% comparte tu postura. Estas alineado a la norma operativa."
            elif porcentaje > 15:
                contexto = "Minoria"
                por_que_contexto = f"Solo el {porcentaje:.1f}% coincide contigo. Ejecutas un enfoque alternativo."
            else:
                contexto = "Perfil Unico"
                por_que_contexto = f"Unicamente el {porcentaje:.1f}% respondio igual. Postura aislada matematicamente."
            
            if var in ["Dudas", "Comprension", "Organizacion", "Ideas", "Avanzar"]:
                if val_num >= 4:
                    nivel_eval = "Buen Nivel"
                    por_que_nivel = "Delegacion eficiente que maximiza el procesamiento tecnico."
                elif val_num == 3:
                    nivel_eval = "Nivel Moderado"
                    por_que_nivel = "Balance neutro. No explotas del todo la automatizacion."
                else:
                    nivel_eval = "Nivel Suboptimo"
                    por_que_nivel = "Baja optimizacion. Gastas tiempo operativo valioso en procesos manuales."
            else:
                if val_num >= 4:
                    nivel_eval = "Nivel de Riesgo"
                    por_que_nivel = "Condicionas la retencion teorica al rendimiento de la plataforma."
                elif val_num == 3:
                    nivel_eval = "Nivel Neutro"
                    por_que_nivel = "Uso auxiliar. La herramienta opera de forma periferica."
                else:
                    nivel_eval = "Buen Nivel"
                    por_que_nivel = "Autonomia tecnica pura libre de dependencias externas."
            
            filas_tabla.append({
                "Variable Evaluada": VAR_DESCRIPCIONES[var],
                "Tu Puntaje": val_num,
                "Media Grupo": round(media_g, 2),
                "Diferencia": round(diff, 2),
                "Contexto Social": contexto,
                "Motivo Contextual": por_que_contexto,
                "Evaluacion Tecnica": nivel_eval,
                "Justificacion Operativa": por_que_nivel
            })
        else:
            filas_tabla.append({
                "Variable Evaluada": VAR_DESCRIPCIONES[var],
                "Tu Puntaje": val_num,
                "Significado": significado_likert,
                "Contexto": "Requiere dataset del grupo",
                "Evaluacion": "Requiere dataset del grupo"
            })
        
    st.dataframe(pd.DataFrame(filas_tabla), hide_index=True, use_container_width=True)
    st.markdown("---")

def render_comparativa_visual_individual(df_grupo, df_usuario, variable):
    desc = VAR_DESCRIPCIONES.get(variable, variable)
    data_grupo = pd.to_numeric(df_grupo[variable], errors="coerce").dropna()
    val_usuario = pd.to_numeric(df_usuario[variable], errors="coerce").iloc[0]
    
    if len(data_grupo) == 0: return

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data_grupo, discrete=True, ax=ax, color="#4C72B0", alpha=0.6)
    ax.axvline(val_usuario, color="#d9534f", linewidth=4, linestyle="-", label="Tu Respuesta")
    ax.set_xlabel("Escala Likert (1 = Nunca - 5 = Siempre)")
    ax.set_ylabel("Frecuencia en el grupo")
    ax.set_title(f"Tu Posicion vs El Salon: {desc}")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    coincidencias = (data_grupo == val_usuario).sum()
    pct_coincidencia = (coincidencias / len(data_grupo)) * 100

    st.info(
        f"**Conclusiones de tu Posicion Intergrupal:**\n"
        f"Seleccionaste la metrica {val_usuario:.0f}. Matematicamente, el {pct_coincidencia:.1f}% de los encuestados comparte exactamente tu misma posicion. "
        "Alinearse con el centro de la distribucion denota una adopcion estandarizada de la tecnologia, mientras que situarse en los extremos señala el uso de metodologias operativas no convencionales o altamente especializadas frente a tus pares."
    )
    st.markdown("---")


def render_evaluacion_individual_y_tacticas(df_usuario):
    st.markdown("### Diagnostico y Tactica Academica")

    row = df_usuario.iloc[0]
    valores = [int(row.get(var, 3)) for var in VARIABLES]
    media_usuario = sum(valores) / len(valores)
    carrera_val = row.get("Carrera", "Otra")
    ciclo_val_texto = row.get("Ciclo", "1 ciclo")

    try:
        num_ciclo = int(f"{ciclo_val_texto}".split()[0])
    except Exception:
        num_ciclo = 1

    st.markdown("#### Tu Perfil Visual de Respuestas")
    fig_ind, ax_ind = plt.subplots(figsize=(8, 4))
    etiquetas_cortas = [VAR_DESCRIPCIONES[v][:25] + "..." for v in VARIABLES]
    colores = ["#2e7d32" if v >= 4 else "#f9a825" if v >= 3 else "#c62828" for v in valores]

    bars = ax_ind.barh(etiquetas_cortas, valores, color=colores, edgecolor="white")
    ax_ind.set_xlim(0, 5)
    ax_ind.axvline(x=3, color="gray", linestyle="--", linewidth=1)
    ax_ind.set_xlabel("Nivel de adopcion (1 = Nunca - 5 = Siempre)")
    ax_ind.set_title("Autoevaluacion Operativa")

    for bar, val in zip(bars, valores):
        ax_ind.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val}", va="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_ind)
    plt.close(fig_ind)
    
    st.info(
        "**Conclusiones de tu Perfil Individual:**\n"
        "La longitud de las barras expone los procesos donde concedes mayor delegacion tecnologica frente a las areas donde preservas un control manual directo. "
        "Un grafico marcadamente asimetrico evidencia un uso puramente tactico, mientras que un esquema uniforme indica una adopcion sistematica integral a lo largo de tu flujo de trabajo."
    )
    st.markdown("---")

    st.markdown(f"#### Habitos de Estudio y Uso de IA (Promedio: {media_usuario:.2f} / 5.00)")
    if media_usuario >= 4.0:
        st.write("Diagnostico: Alta Integracion Operativa. Tu dependencia de la automatizacion sugiere eficiencia en tareas pesadas, con el riesgo inherente de atrofiar la agilidad analitica pura si se omite la validacion teorica.")
        st.markdown(
            "**Tacticas de Optimizacion:**\n"
            "* Modo de Auditoria: Dedica 20 minutos a diseñar la logica o seudocodigo analogico antes de habilitar la asistencia generativa.\n"
            "* Validacion Inversa: Configura el modelo para que detecte vulnerabilidades o ineficiencias en tus planteamientos manuales, en lugar de generar la arquitectura desde cero.\n"
            "* Aislamiento Programado: Ejecuta bloques de desarrollo de 2 horas en desconexion total para preservar y ejercitar el razonamiento critico individual."
        )
    elif media_usuario >= 2.5:
        st.write("Diagnostico: Adopcion Pragmatica. Reflejas un balance tecnico adecuado, empleando los modelos para acelerar procesos intermedios mientras retienes el dominio sobre la integracion final.")
        st.markdown(
            "**Tacticas de Optimizacion:**\n"
            "* Interaccion Socratica: Ordena al sistema formular preguntas secuenciales y analiticas que conduzcan a la solucion, evitando la entrega de bloques operativos completos.\n"
            "* Delegacion Selectiva: Limita la automatizacion al formateo, configuracion inicial o generacion de datos de prueba, blindando el desarrollo del nucleo del proyecto.\n"
            "* Refactorizacion Obligatoria: Toda linea de codigo o conclusion generada debe ser reescrita e interpretada bajo tus propios esquemas de sintaxis antes de implementarse."
        )
    else:
        st.write("Diagnostico: Adopcion Conservadora. Tus procesos priorizan la traccion teorica y el metodo formativo tradicional. Aunque esto solidifica bases tecnicas, sacrifica metricas de eficiencia operativa valiosas en el mercado laboral.")
        st.markdown(
            "**Tacticas de Optimizacion:**\n"
            "* Micro-Delegacion: Asigna al modelo tareas exclusivamente logisticas, como planificacion de diagramas o extraccion automatizada de datos bibliograficos.\n"
            "* Consultoria Semantica: Utiliza el motor como un glosario analitico para abstraer y simplificar conceptos de alta densidad tecnica sin afectar la resolucion central del problema.\n"
            "* Arquitecturas Base: Mitiga la friccion de arranque permitiendo que la plataforma genere el esqueleto inicial del documento o script, acelerando el inicio operativo."
        )

    st.markdown(f"#### Perspectiva Disciplinaria: {carrera_val}")
    
    if carrera_val in ["Ingeniería en Inteligencia Artificial", "Ingeniería en Ciencia de Datos"]:
        mensaje_disciplina = "Tu disciplina exige el diseño y ajuste profundo de los modelos, no su mero consumo. El mercado laboral penaliza la incapacidad para justificar las decisiones algoritmicas. Al desarrollar soluciones complejas, tu valor real radica en el analisis de variables. Delega la sintaxis repetitiva a la automatizacion, pero manten el dominio absoluto sobre la formulacion matematica y el ajuste de la Tolerancia de tus implementaciones."
    elif carrera_val in ["Ingeniería en Sistemas Computacionales", "Ingeniería Matemática"]:
        mensaje_disciplina = "La redaccion de codigo basico y el calculo de operaciones primarias se han devaluado. Tu ventaja competitiva reside en el diseño de arquitecturas escalables, seguridad de bases de datos y la comprobacion formal de algoritmos. Permite que la IA asuma el codigo estandar repetitivo, pero reten la jurisdiccion tecnica completa sobre el despliegue y la integracion del sistema general."
    elif carrera_val in ["Química", "Química farmacéutica", "QBP", "QFB", "Medicina Veterinaria y Zootecnia", "LMD"]:
        mensaje_disciplina = "Las ciencias biologicas y de la salud operan bajo un principio ineludible de experimentacion fisica y rigor bioetico. La computacion avanzada debe limitar su rol a la correlacion de probabilidad y procesamiento de datos masivos. La formulacion de diagnosticos in vivo, el control de laboratorio y la validacion clinica son mandatos que requieren invariablemente tu supervision y etica profesional."
    elif carrera_val in ["Ingeniería Aeronáutica", "Mecánica", "Ingeniería en Sistemas Ambientales"]:
        mensaje_disciplina = "El diseño generativo asistido reduce significativamente los tiempos de iteracion inicial, pero la aprobacion sobre integridad termodinamica, impacto ambiental regulatorio y coeficientes de fatiga de materiales derivan en responsabilidades civiles. Todo resultado emitido por IA requiere auditoria determinista humana de tolerancia cero."
    else:
        mensaje_disciplina = "Frente a la automatizacion masiva, la destreza operativa pura ha perdido traccion competitiva. Para afianzar el rendimiento profesional a largo plazo, la prioridad absoluta debe concentrarse en la gestion de tacticas direccionales, la mitigacion de crisis atipicas y el pensamiento sistemico."
    
    st.info(mensaje_disciplina)

    st.markdown(f"#### Proyeccion Operativa de Nivel ({ciclo_val_texto})")
    if num_ciclo <= 3:
        if media_usuario >= 4.0:
            texto_nivel = f"Fase de Fundamentos: Te encuentras cursando el tronco comun de {carrera_val}. La prioridad es consolidar el razonamiento matematico y deductivo base. El uso prematuro y excesivo de plataformas generativas para resolver ejercicios de tronco comun bloqueara irreversiblemente tu capacidad de desarrollar logica avanzada. Reduce de inmediato el uso automatizado; emplea la herramienta exclusivamente como glosario analitico."
        elif media_usuario >= 2.5:
            texto_nivel = f"Fase de Fundamentos: Atraviesas las materias de tronco comun en {carrera_val}. Mantienes un uso moderado. Asegura no cruzar la linea hacia la automatizacion de tus ejercicios base. Emplea la herramienta rigurosamente como un tutor de apoyo para clarificar conceptos teoricos, garantizando que el esfuerzo analitico fundacional siga siendo tuyo."
        else:
            texto_nivel = f"Fase de Fundamentos: Tu enfoque tradicional es altamente beneficioso para el tronco comun de {carrera_val}. Al resolver los ejercicios de forma analogica aseguras el desarrollo de las conexiones logicas requeridas para las asignaturas de especialidad. Manten este rigor y utiliza la tecnologia solo para optimizar labores organizativas menores."
    elif num_ciclo <= 7:
        if media_usuario >= 4.0:
            texto_nivel = f"Fase de Integracion: Las materias de tronco comun quedaron atras; enfrentas la carga volumetrica y pesada de la especialidad en {carrera_val}. Tu dependencia actual de la IA representa un riesgo frente al escrutinio docente, ya que evaluaran la viabilidad y autoria intelectual de tus diseños complejos. Debes trasladar el peso de la resolucion a tu propio analisis y usar la IA solo para depurar logs de errores (debugging). Simultaneamente, se abren ventanas hacia oportunidades laborales; si omites el aprendizaje empirico y permites que la automatizacion absorba la resolucion, llegaras al cierre academico sin la capacidad analitica que la industria exige."
        elif media_usuario >= 2.5:
            texto_nivel = f"Fase de Integracion: Enfrentas la carga pesada de las materias de especialidad en {carrera_val}. Mantienes un buen equilibrio. Se requiere una adopcion tactica para optimizar tiempos operativos, depurar errores y gestionar repositorios, garantizando la autoria intelectual de tus desarrollos. Los docentes evaluaran la eficiencia algoritmica y tecnica de tus soluciones. Ante la apertura de oportunidades laborales, tu capacidad de combinar aprendizaje autodidacta con asistencia tecnologica te permitira edificar un portafolio competitivo."
        else:
            texto_nivel = f"Fase de Integracion: La complejidad y el volumen de las materias de especialidad en {carrera_val} exigen eficiencia. Tu bajo nivel de adopcion tecnologica puede generar cuellos de botella criticos. Los docentes evaluaran la eficiencia de tus soluciones complejas. Debes comenzar a integrar la IA para optimizar tiempos operativos y depurar errores. Las primeras oportunidades laborales requieren agilidad; si te mantienes aislado de la automatizacion, estaras en desventaja competitiva frente al sector corporativo."
    else:
        if media_usuario >= 4.0:
            texto_nivel = f"Fase Terminal: Orientacion hacia la insercion laboral en {carrera_val}. Tu perfil exhibe una sobre-utilizacion de recursos automatizados que sera penalizada en procesos de reclutamiento tecnico. Las plataformas deben utilizarse para depurar logica, auditar eficiencias y generar documentacion tecnica, no para suplir el diseño nucleo. Debes evidenciar autonomia de inmediato mediante el desarrollo de proyectos independientes sin asistencia para validar tu competencia profesional ante empleadores."
        elif media_usuario >= 2.5:
            texto_nivel = f"Fase Terminal: Orientacion hacia la insercion laboral en {carrera_val}. Tu uso equilibrado te posiciona favorablemente. Utiliza las plataformas tecnologicas para depurar logica, auditar eficiencias, generar documentacion y simular escenarios de entrevistas corporativas, exhibiendo madurez tecnologica y autonomia tecnica ante futuros empleadores en tu area."
        else:
            texto_nivel = f"Fase Terminal: Orientacion hacia la insercion laboral en {carrera_val}. Es mandatorio que aceleres tu dominio sobre herramientas de asistencia para equiparar los estandares de productividad corporativa. Utiliza las plataformas para auditar eficiencias y comprender flujos de trabajo automatizados. El mercado laboral exige precision y velocidad, y la adopcion de estas tecnologias es indispensable para destacar en procesos de reclutamiento."
    
    st.write(texto_nivel)


# ──────────────────────────────────────────────────────────────────────
# ESTRUCTURA PRINCIPAL DE LA APLICACION (TABS UI)
# ──────────────────────────────────────────────────────────────────────
st.title("Monitor de Adopcion Tecnologica")
tab_dataset, tab_form = st.tabs(["Analisis Poblacional", "Evaluacion Individual"])

with tab_dataset:
    st.markdown("Procesamiento del conjunto de datos grupales para extraccion de metricas descriptivas e inferencia de probabilidad.")
    archivo = st.file_uploader("Cargar matriz de datos poblacional (CSV)", type=["csv"], key="csv_dataset")

    if archivo is not None:
        df = pd.read_csv(archivo)
        df = df.rename(columns=COL_MAP)
        
        render_perfil(df)
        render_resumen_ejecutivo(df)
        render_prueba_hipotesis(df)
        
        st.markdown("### Visualizacion del Comportamiento Grupal")
        st.caption("Selecciona las metricas que deseas auditar visualmente para entender el comportamiento de la mayoria.")
        labels_sel = st.multiselect(
            "Metricas a graficar:",
            options=VAR_LABELS,
            default=[VAR_LABELS[0]],
            key="var_dataset"
        )
        for label in labels_sel:
            variable = VAR_LABEL_TO_KEY[label]
            render_analisis_grupo_variable(df, variable)

with tab_form:
    st.markdown("Ingreso de parametros personales para el calculo y proyeccion de perfiles de uso.")
    
    with st.form("formulario_ia"):
        col1, col2 = st.columns(2)
        with col1:
            ciclo_val = st.selectbox("Nivel Curricular:", NIVELES)
            carrera_val  = st.selectbox("Especialidad Academica:", CARRERAS)
        with col2:
            uso_ia = st.radio("Integra IA generativa en procesos academicos?", ["Sí", "No"])
            frecuencia = st.selectbox("Frecuencia global declarada:", FRECUENCIAS)
        
        st.markdown("---")
        st.markdown("**Evaluacion de Patrones de Uso (Escala 1 a 5):**")
        cols = st.columns(2)
        respuestas = {}
        for var, (texto, col_idx) in PREGUNTAS.items():
            with cols[col_idx]:
                respuestas[var] = st.select_slider(texto, options=list(LIKERT.keys()), format_func=lambda x: f"{LIKERT[x]}", key=f"sl_{var}")
        
        submitted = st.form_submit_button("Ejecutar Diagnostico")

    if submitted:
        fila = {"Ciclo": ciclo_val, "Carrera": carrera_val, "Uso_IA": uso_ia, "Frecuencia": frecuencia, **respuestas}
        st.session_state["respuesta_usuario"] = pd.DataFrame([fila])
        st.success("Variables capturadas correctamente. Sistema listo.")

    if "respuesta_usuario" in st.session_state:
        df_usuario = st.session_state["respuesta_usuario"]
        
        st.markdown("---")
        st.markdown("#### Activacion de Modelos Comparativos")
        archivo2 = st.file_uploader(
            "Para habilitar el mapeo visual e inferencial contra el grupo, cargue el dataset base:",
            type=["csv"],
            key="csv_form"
        )
        
        if archivo2 is not None:
            df_grupo_contexto = pd.read_csv(archivo2)
            df_grupo_contexto = df_grupo_contexto.rename(columns=COL_MAP)
            
            render_resumen_usuario(df_usuario, df_grupo=df_grupo_contexto)
            render_evaluacion_individual_y_tacticas(df_usuario)
            
            st.markdown("---")
            st.markdown("### Mapeo Visual: Tu vs. La Poblacion")
            st.caption("Verifica en tiempo real como se contrasta tu respuesta individual contra la densidad general del salon.")
            labels_sel2 = st.multiselect(
                "Seleccionar metricas a evaluar:",
                options=VAR_LABELS,
                default=[VAR_LABELS[0], VAR_LABELS[1]],
                key="var_form_context"
            )
            for label in labels_sel2:
                variable2 = VAR_LABEL_TO_KEY[label]
                render_comparativa_visual_individual(df_grupo_contexto, df_usuario, variable2)
                
        else:
            render_resumen_usuario(df_usuario)
            render_evaluacion_individual_y_tacticas(df_usuario)
