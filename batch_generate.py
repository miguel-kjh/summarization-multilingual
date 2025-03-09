import os
import subprocess

base_paths = [
    "models/Qwen/Qwen2.5-0.5B",
    "models/Qwen/Qwen2.5-0.5B-Instruct",
    "models/Qwen/Qwen2.5-1.5B",
    "models/Qwen/Qwen2.5-1.5B-Instruct",
    "models/Qwen/Qwen2.5-3B",
    "models/Qwen/Qwen2.5-3B-instruct",
    "models/BSC-LT/salamandra-2b",
    "models/BSC-LT/salamandra-2b-instruct",
    "models/meta-llama/Llama-3.2-3B",
    "models/meta-llama/Llama-3.2-3B-instruct",
    "models/meta-llama/Llama-3.2-1B",
    "models/meta-llama/Llama-3.2-1B-instruct",
]
target_depth = 4
chunks = "chunks"
datasets = ["data/04-clustering"]

for base_path in base_paths:
    for root, dirs, files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep) + 1  # +1 para incluir el base_path como nivel 1
        if depth == target_depth or "openai" in root:
            if "FacebookAI" in root:
                root = root + "/xlm-roberta-large"
            print(f"Directorio en profundidad {target_depth}: {root}")
            using_clustering = chunks in root
            data = os.path.join(datasets[0], root.split(os.sep)[-3])

            print(f"Dataset: {data}")
            print(f"Usando clustering: {using_clustering}")
            command = [
                "python", "generate.py",
                "--model_name_or_path", root,
                "--dataset", data,
                "--using_clustering", "True" if using_clustering else "False",
                "--data_sample", "10" if using_clustering else "5",
                "--quantization", "False",
            ]

            # Ejecuta el script y muestra la salida en tiempo real
            try:
                subprocess.run(command, check=True)
            except:
                print("Error al ejecutar el script")
            print("Generación finalizada")
            print("#"*50)
            print("Evaluar modelo")
            command_evaluate = [
                "python", "model_evaluate.py",
                "--model_name_or_path", root,
                "--wandb", "False",
                "--method", "normal" if not using_clustering else "clustering",
                "--use_openai", "True",
                "--up", "True",
            ]
            try:
                subprocess.run(command_evaluate, check=True)
            except:
                print("Error al ejecutar el script de evaluación")

        elif depth > target_depth:
            dirs.clear()  # Detiene la exploración de niveles más profundos