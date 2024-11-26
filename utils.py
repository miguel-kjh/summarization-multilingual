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

INSTRUCTION_TEMPLATE = {
    "en": "Please summarize the following text in a few sentences, highlighting the most important points.",
    "es": "Por favor, resuma el siguiente texto en unas pocas frases, destacando los puntos más importantes.",
    "fr": "Veuillez résumer le texte suivant en quelques phrases, en mettant en évidence les points les plus importants.",
    "de": "Bitte fassen Sie den folgenden Text in ein paar Sätzen zusammen und heben Sie die wichtigsten Punkte hervor.",
    "ru": "Пожалуйста, подытожите следующий текст несколькими предложениями, выделив наиболее важные моменты.",
    "tu": "Lütfen aşağıdaki metni birkaç cümlede özetleyin ve en önemli noktaları vurgulayın."
}

if __name__ == '__main__':
    print("This is a utils file")