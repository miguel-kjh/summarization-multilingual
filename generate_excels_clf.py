import os
import json

import pandas as pd

def find_and_read_data(base_dir, max_depth):
    results = {}
    
    for root, dirs, files in os.walk(base_dir):
        # Calcular la profundidad actual
        depth = root[len(base_dir):].count(os.sep)
        
        if depth == max_depth:
            if "chunks-sentence-transformers" in root:
                print(f"Leyendo {root}")
                df = pd.read_csv(os.path.join(root, "clf_models.csv"), sep=";")
                lang = root.replace("-chunks-sentence-transformers", "").split(os.sep)[-1]
                results[lang] = df
        
        # No bajar más allá del nivel especificado
        if depth >= max_depth:
            dirs.clear()
    
    return results

def save_to_excel(results, output_file):
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    for language, data in results.items():
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=language, index=False)
    
    writer.close()

# Ejemplo de uso
if __name__ == "__main__":
    base_directory = "data/04-clustering"  # Cambia esto por la ruta base
    model = base_directory.split(os.sep)[-1]
    max_search_depth = 1  # Cambia esto al nivel deseado
    output_excel = os.path.join(base_directory, f"metrics_summary_{model}.xlsx")
    
    results = find_and_read_data(base_directory, max_search_depth)
    print(results.keys())
    print(f"Resultados encontrados: {results}")
    save_to_excel(results, output_excel)
    print(f"Resultados guardados en {output_excel}")
