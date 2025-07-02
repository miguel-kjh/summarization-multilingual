import pandas as pd
import os

def resaltar_maximos(df):
    df_format = df.copy()
    for col in df.columns:
        if col != "model" and pd.api.types.is_numeric_dtype(df[col]):
            max_val = df[col].max()
            df_format[col] = df[col].apply(lambda x: f"\\textbf{{{x:.2f}}}" if x == max_val else f"{x:.2f}")
        elif col != "model":
            df_format[col] = df[col]  # sin formato especial
    return df_format

def generar_tablas_latex_por_hoja(ruta_excel, carpeta_salida="metrics/tablas_latex"):
    name_file = os.path.basename(ruta_excel).split(".")[0]
    carpeta_salida = os.path.join(carpeta_salida, name_file)
    os.makedirs(carpeta_salida, exist_ok=True)

    columnas_a_eliminar = [
        "times(sec)", "origen", "bertscore_precision", "bertscore_recall"
    ]

    renombrado_columnas = {
        "model": "\\textbf{Model}",
        "rouge1": "\\textbf{R-1}",
        "rouge2": "\\textbf{R-2}",
        "rougeL": "\\textbf{R-L}",
        "rougeLsum": "\\textbf{R-LS}",
        "bertscore_f1": "\\textbf{BS}",
        "coherence": "\\textbf{Coh.}",
        "consistency": "\\textbf{Cons.}",
        "fluency": "\\textbf{Flu.}",
        "relevance": "\\textbf{Rel.}",
        "average": "\\textbf{Avg.}"
    }

    excel = pd.ExcelFile(ruta_excel)

    for hoja in excel.sheet_names:
        df = excel.parse(hoja)
        df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], errors='ignore')

        if "model" in df.columns:
            df = df.sort_values(by="model", ascending=True)
            df["model"] = [
                model.split(hoja)[0] if isinstance(model, str) else model for model in df["model"]
            ]

        # Aplicar formato de máximos
        df_format = resaltar_maximos(df)

        # Renombrar columnas para formato LaTeX
        df_format = df_format.rename(columns=renombrado_columnas)

        # Exportar
        name_file_formatted = name_file.replace("_", " ")
        latex_tabla = df_format.to_latex(index=False, escape=False)
        tabla_completa = (
            f"\FloatBarrier\n"
            f"\\begin{{table}}[ht]\n"
            f"\\caption{{Resultados para {hoja} {name_file_formatted}}}\n"
            f"\\label{{tab:{hoja}}}\n"
            f"\\centering\n"
            f"{latex_tabla}\n"
            f"\\end{{table}}\n"
            f"\\FloatBarrier\n"
        )

        nombre_archivo = os.path.join(carpeta_salida, f"{hoja}.tex")
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(tabla_completa)

    print(f"Tablas LaTeX generadas en: {carpeta_salida}")

if __name__ == "__main__":
    ruta_excel = ["data/metrics_summary_baseline.xlsx", "data/metrics_summary_ft.xlsx", "data/metrics_summary_not_ft.xlsx"]
    for ruta in ruta_excel:
        generar_tablas_latex_por_hoja(ruta)

