import os
from datasets import load_dataset
import shutil

from utils import DATASET

def download_dataset(language, folder):
    
    # Configuración para otros idiomas
    lang_folder = os.path.join(folder, language)
    
    print(f"Downloading dataset for language {language}")
    dataset = load_dataset(DATASET, language, trust_remote_code=True)

    dataset = dataset.rename_column('reference', 'text')

    if os.path.exists(lang_folder):
        shutil.rmtree(lang_folder)

    # Guardar el dataset procesado
    dataset.save_to_disk(os.path.join(folder, language))
    print(f"Dataset for language {language} downloaded and saved")
