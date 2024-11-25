import os

# Folders
DATA_FOLDER           = "data"
RAW_DATA_FOLDER       = os.path.join(DATA_FOLDER, "01-raw")
PROCESS_DATA_FOLDER   = os.path.join(DATA_FOLDER, "02-processed")
COMBINED_DATA_FOLDER  = os.path.join(DATA_FOLDER, "03-combined")

# Statistics
FILE_STATS = os.path.join(RAW_DATA_FOLDER, "stats.csv")

# Languages to download
LANGUAGES = ['de', 'es', 'fr', 'ru', 'tu', 'en']

SEED = 3407