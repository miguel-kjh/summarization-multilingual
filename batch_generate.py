import os
import subprocess

base_path = "models/meta-llama"
target_depth = 5
chunks = "chunks"
datasets = ["data/04-clustering"]

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
        """print("#"*50)
        print("Evaluar modelo")
        command_evaluate = [
            "python", "model_evaluate.py",
            "--model_name_or_path", root,
            "--wandb", "False",
            "--method", "normal" if not using_clustering else "clustering",
            "--use_openai", "True",
            "--up", "True",
        ]
        subprocess.run(command_evaluate, check=True)"""

    elif depth > target_depth:
        dirs.clear()  # Detiene la exploración de niveles más profundos