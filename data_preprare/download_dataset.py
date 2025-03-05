import os
from datasets import load_dataset
from datasets import DatasetDict
import shutil

from utils import DATASET, DATASET_PAR_CAN

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


def download_canary_parlament(folder):
    print(f"Downloading dataset for Canary Parlament")
    ds = load_dataset(DATASET_PAR_CAN, trust_remote_code=True)
    # split the dataset
    # Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    train_test_split = ds["train"].train_test_split(test_size=0.1)

    # Dividir la parte de prueba en validación y prueba (50% validación, 50% prueba)
    val_test_split = train_test_split["test"].train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    new_ds = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    new_ds.save_to_disk(os.path.join(folder, "canario"))
    print(f"Dataset for Canary Parlament downloaded and saved")
