import numpy as np
import pandas as pd
from scipy import stats

def basic_stats(df, col):
    data = pd.to_numeric(df[col], errors="coerce").dropna()

    if len(data) == 0:
        return {
            "Media": 0,
            "Moda": 0,
            "Mediana": 0,
            "Varianza": 0,
            "Desviación estándar": 0
        }

    moda = stats.mode(data, keepdims=True)

    return {
        "Media": float(np.mean(data)),
        "Moda": float(moda.mode[0]),
        "Mediana": float(np.median(data)),
        "Varianza": float(np.var(data, ddof=1)) if len(data) > 1 else 0,
        "Desviación estándar": float(np.std(data, ddof=1)) if len(data) > 1 else 0
    }

def confidence_interval(df, col, conf=0.95):
    data = pd.to_numeric(df[col], errors="coerce").dropna()

    if len(data) <= 1:
        return (0, 0)

    media = np.mean(data)
    error = stats.sem(data)
    h = error * stats.t.ppf((1 + conf) / 2, len(data) - 1)

    return (media - h, media + h)

def conditional_prob(df, colA, colB):
    return pd.crosstab(df[colA], df[colB], normalize="index")

def chi_square(df, colA, colB):
    tabla = pd.crosstab(df[colA], df[colB])
    
    # Prevenir error si la tabla es muy pequeña por los filtros
    if tabla.empty or tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return 0, 1.0, 0, None
        
    return stats.chi2_contingency(tabla)

def correlacion_spearman(df, col_x, col_y):
    data_x = pd.to_numeric(df[col_x], errors="coerce")
    data_y = pd.to_numeric(df[col_y], errors="coerce")
    
    temp_df = pd.DataFrame({"X": data_x, "Y": data_y}).dropna()
    
    if len(temp_df) < 2:
        return 0.0, 1.0
        
    rho, p_val = stats.spearmanr(temp_df["X"], temp_df["Y"])
    return rho, p_val
