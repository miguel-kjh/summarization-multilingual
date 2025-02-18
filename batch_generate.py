import os
import subprocess

base_path = "models/Qwen"
target_depth = 5
chunks = "chunks"
datasets = ["data/02-processed", "data/04-clustering"]

for root, dirs, files in os.walk(base_path):
    depth = root[len(base_path):].count(os.sep) + 1  # +1 para incluir el base_path como nivel 1
    if depth == target_depth:
        print(f"Directorio en profundidad {target_depth}: {root}")
        using_clustering = chunks in root
        if using_clustering:
            data = os.path.join(datasets[1], root.split(os.sep)[-3])
        else:
            data = os.path.join(datasets[0], root.split(os.sep)[-3])
        
        print(f"Dataset: {data}")
        print(f"Usando clustering: {using_clustering}")
        command = [
            "python", "generate.py",
            "--model_name_or_path", root,
            "--dataset", data,
            "--using_clustering", "True" if using_clustering else "False",
            "--data_sample", "10" if "german" not in data else "5",
        ]

        # Ejecuta el script y muestra la salida en tiempo real
        subprocess.run(command, check=True)
        #print("Generación finalizada")
        #print("#"*50)
        #print("Evaluar modelo")
        #command_evaluate = [
        #    "python", "model_evaluate.py",
        #    "--model_name_or_path", root,
        #    "--wandb", "True",
        #    "--method", "normal" if not using_clustering else "clustering",
        #    "--use_openai", "False",
        #]
        #subprocess.run(command_evaluate, check=True)

    elif depth > target_depth:
        dirs.clear()  # Detiene la exploración de niveles más profundos