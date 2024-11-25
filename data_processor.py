import os
import argparse
from datasets import load_from_disk

from utils import LANGUAGES, RAW_DATA_FOLDER, FILE_STATS
from data_preprare.download_dataset import download_dataset
from data_preprare.generate_data_stats import StatsGenerator
from tqdm import tqdm

def download():
    for lang in LANGUAGES:
        download_dataset(lang, RAW_DATA_FOLDER)
    print("All datasets downloaded and saved")

def stats():
    stats_gen = StatsGenerator()
    for lang in tqdm(LANGUAGES, desc="Processing languages"):
        dataset_name = os.path.join(RAW_DATA_FOLDER, lang)
        assert os.path.exists(dataset_name), f"Dataset {dataset_name} does not exist"
        dataset = load_from_disk(dataset_name)
        stats_gen.add_lang(dataset, lang)

    df = stats_gen.get_stats()
    df.to_csv(FILE_STATS, index=False)

def process():
    pass

def parse():
    pass

OPERATIONS = {
    "download": download,
    "stats": stats,
    "process": process,
}


if __name__ == '__main__':
    #download()
    stats()