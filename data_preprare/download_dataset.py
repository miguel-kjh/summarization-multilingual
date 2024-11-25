import os
from datasets import load_dataset
import shutil

def download_dataset(language, folder):
    
    # Configuración para otros idiomas
    lang_folder = os.path.join(folder, language)
    print(f"Downloading dataset for language {language}")

    if not os.path.exists(lang_folder):  
        if language == 'en':
            
            dataset = load_dataset(
                "ccdv/cnn_dailymail",
                '3.0.0',
                trust_remote_code=True,
            )
            
            dataset = dataset.rename_column('article', 'text')
            dataset = dataset.rename_column('highlights', 'summary')
            
            if os.path.exists(lang_folder):
                shutil.rmtree(lang_folder)
            
            # Guardar el dataset procesado
            dataset.save_to_disk(os.path.join(folder, 'en'))
            print(f"Dataset for language {language} downloaded and saved")
        
        else:
            if not os.path.exists(lang_folder):
                try:
                    # Cargar dataset MLSum
                    dataset = load_dataset(
                        "reciTAL/mlsum",
                        language,
                        trust_remote_code=True,
                    )
                    
                    # delete the folder if it exists
                    if os.path.exists(lang_folder):
                        shutil.rmtree(lang_folder)

                    # Guardar el dataset procesado
                    dataset.save_to_disk(lang_folder)

                    print(f"Dataset for language {language} downloaded and saved")

                except KeyboardInterrupt:
                    return

                except Exception as e:
                    print(f"Error to load dataset for language {language}")
