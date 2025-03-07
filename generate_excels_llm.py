import os
import json

import pandas as pd

def find_and_read_json(base_dir, max_depth):
    results = {}
    
    for root, dirs, files in os.walk(base_dir):
        # Calcular la profundidad actual
        depth = root[len(base_dir):].count(os.sep)
        
        if depth == max_depth:
            if "result_metrics.json" in files:
                model = root.split(os.sep)[-1]
                json_path = os.path.join(root, "result_metrics.json")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        results[model] = data
                except Exception as e:
                    print(f"Error leyendo {json_path}: {e}")
        
        # No bajar más allá del nivel especificado
        if depth >= max_depth:
            dirs.clear()
    
    return results

def save_to_excel(results, output_file):
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    language_data = {}
    for path, content in results.items():
        base_model_name = os.path.basename(path)
        model_name = base_model_name.split("-")[:4]
        
        for language, metrics in content.items():
            if language not in language_data:
                language_data[language] = []
            
            row = {"model": base_model_name}
            row.update(metrics["rouge"])
            row.update(metrics["bertscore"])
            row.update({
                "coherence": metrics["coherence"],
                "consistency": metrics["consistency"],
                "fluency": metrics["fluency"],
                "relevance": metrics["relevance"],
                "average": metrics["average"]
            })
            row["times(sec)"] = metrics["times(sec)"]
            
            language_data[language].append(row)
    
    for language, data in language_data.items():
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=language, index=False)
    
    writer.close()

# Ejemplo de uso
if __name__ == "__main__":
    base_directory = "models/BSC-LT/salamandra-7b"  # Cambia esto por la ruta base
    model = base_directory.split(os.sep)[-1]
    max_search_depth = 3  # Cambia esto al nivel deseado
    output_excel = os.path.join(base_directory, f"metrics_summary_{model}.xlsx")
    
    results = find_and_read_json(base_directory, max_search_depth)
    print(f"Resultados encontrados: {results}")
    save_to_excel(results, output_excel)
    print(f"Resultados guardados en {output_excel}")
