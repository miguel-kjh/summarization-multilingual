#main
import os
import pandas as pd


def save_merged_by_sheet(results, output_file):
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sheet_name, dfs in results.items():
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"Archivo combinado guardado en: {output_file}")

if __name__ == "__main__":
    list_data = [
        "metrics/data/BSC-LT_merged_by_language.xlsx",
        "metrics/data/Qwen_merged_by_language.xlsx",
        "metrics/data/unsloth_merged_by_language.xlsx",
    ]

    results = {}
    for file_path in list_data:
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                if sheet_name not in results:
                    results[sheet_name] = []
                df["model"] = df["modelo"] + ' (ft)'
                df.drop(columns=["modelo"], inplace=True, errors='ignore')
                results[sheet_name].append(df)
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")

    output_file = "metrics/data/metrics_summary_ft.xlsx"
    save_merged_by_sheet(results, output_file)



    