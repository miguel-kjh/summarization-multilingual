import os
import json

import pandas as pd

def find_and_read_json(base_dir, max_depth):
    results = {}
    
    for root, dirs, files in os.walk(base_dir):
        # Calcular la profundidad actual
        depth = root[len(base_dir):].count(os.sep)
        
        if depth == max_depth:
            if "truncate_result_metrics.json" in files:
                model = root.split(os.sep)[-1]
                print(f"\"{root}\",")
                json_path = os.path.join(root, "truncate_result_metrics.json")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        results[model] = data
                except Exception as e:
                    print(f"Error leyendo {json_path}: {e}")
            else:
                if "result_metrics.json" in files:
                    print(f"\"{model}\",")
            if "result_metrics.json" in files:
                model = root.split(os.sep)[-1]
                language = root.split(os.sep)[3]  # Asumiendo que el idioma está en la segunda posición
                json_path = os.path.join(root, "result_metrics.json")
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                try:
                    if model not in results:
                        results[model] = data
                    else:
                        # si results esta vacío, inicializarlo
                        results[model][language]["coherence"] = data[language]["coherence"]
                        results[model][language]["consistency"] = data[language]["consistency"]
                        results[model][language]["fluency"] = data[language]["fluency"]
                        results[model][language]["relevance"] = data[language]["relevance"]
                        results[model][language]["average"] = data[language]["average"]
                except Exception as e:
                    print(f"Error procesando {json_path}: {e}")
        
        # No bajar más allá del nivel especificado
        if depth >= max_depth:
            dirs.clear()
    
    return results

def scale_and_round_metrics(metrics_dict):
    return {k: round(v * 100, 2) for k, v in metrics_dict.items()}

def round_metrics(metrics_dict):
    return {k: round(v, 2) for k, v in metrics_dict.items()}

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
            row.update(scale_and_round_metrics(metrics["rouge"]))
            row.update(scale_and_round_metrics(metrics["bertscore"]))
            try:
                row.update({
                    "coherence": round(metrics["coherence"], 2),
                    "consistency": round(metrics["consistency"], 2),
                    "fluency": round(metrics["fluency"], 2),
                    "relevance": round(metrics["relevance"], 2),
                    "average": round(metrics["average"], 2)
                })
            except Exception as e:
                row.update({
                    "coherence": None,
                    "consistency": None,
                    "fluency": None,
                    "relevance": None,
                    "average": None
                })
            row["times(sec)"] = metrics["times(sec)"]
            
            language_data[language].append(row)
    
    for language, data in language_data.items():
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=language, index=False)
    
    writer.close()

# Ejemplo de uso
if __name__ == "__main__":
    base_directories = [
        "models/BSC-LT/salamandra-2b",
        "models/BSC-LT/salamandra-2b-instruct",
        "models/Qwen/Qwen2.5-0.5B",
        "models/Qwen/Qwen2.5-0.5B-Instruct",
        "models/Qwen/Qwen2.5-1.5B",
        "models/Qwen/Qwen2.5-1.5B-Instruct",
        "models/Qwen/Qwen2.5-3B",
        "models/Qwen/Qwen2.5-3B-Instruct",
        "models/Qwen/Qwen3-0.6B",
        "models/Qwen/Qwen3-0.6B-Base",
        "models/Qwen/Qwen3-1.7B",
        "models/Qwen/Qwen3-1.7B-Base",
        "models/Qwen/Qwen3-4B",
        "models/Qwen/Qwen3-4B-Base",
        "models/unsloth/Llama-3.2-1B",
        "models/unsloth/Llama-3.2-3B",
        "models/unsloth/Llama-3.2-1B-Instruct",
        "models/unsloth/Llama-3.2-3B-Instruct",
    ]

    for directory in base_directories:
        model = directory.split(os.sep)[-1]
        max_search_depth = 3  # Cambia esto al nivel deseado
        output_excel = os.path.join(directory, f"metrics_summary_{model}.xlsx")
        
        try:
            results = find_and_read_json(directory, max_search_depth)
        except Exception as e:
            print(f"Error procesando {directory}: {e}")
            continue
        print(f"Resultados encontrados: {results}")
        #save_to_excel(results, output_excel)
        #print(f"Resultados guardados en {output_excel}")
