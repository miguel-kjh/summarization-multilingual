import os
import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk

from data_preprare.transform_data import TransformData
from data_preprare.generate_data_stats import StatsGenerator
from data_preprare.download_dataset import download_dataset
from utils import LANGUAGES, RAW_DATA_FOLDER,  FILE_STATS, PROCESS_DATA_FOLDER

def download():
    for lang in LANGUAGES:
        download_dataset(lang, RAW_DATA_FOLDER)
    print("All datasets downloaded and saved")

def stats():
    print("Generating stats")
    stats_gen = StatsGenerator()
    for lang in tqdm(LANGUAGES, desc="Processing languages"):
        dataset_name = os.path.join(RAW_DATA_FOLDER, lang)
        assert os.path.exists(dataset_name), f"Dataset {dataset_name} does not exist"
        dataset = load_from_disk(dataset_name)
        stats_gen.add_lang(dataset, lang)

    df = stats_gen.get_stats()
    df.to_csv(FILE_STATS, index=False)

def process():
    print("Processing data")
    transform = TransformData()
    for lang in LANGUAGES:
        dataset_name = os.path.join(RAW_DATA_FOLDER, lang)
        dataset = load_from_disk(dataset_name)
        dataset_it_name = os.path.join(PROCESS_DATA_FOLDER, lang)
        os.makedirs(dataset_it_name, exist_ok=True)
        for split in dataset.keys():
            print(f"Processing {split} split for {lang}")
            instructions = transform.generate_instructions(dataset[split], lang)
            file = os.path.join(dataset_it_name, f"{split}.json")
            with open(file, 'w') as f:
                json.dump(instructions, f)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--operation",
        type=str,
        choices=OPERATIONS.keys(),
        help="Operation to perform",
    )
    return parse.parse_args()

OPERATIONS = {
    "download": download,
    "stats": stats,
    "process": process,
}


if __name__ == '__main__':
    args = parse_args()
    operation = args.operation
    assert operation in OPERATIONS, f"Operation {operation} not found"
    OPERATIONS[operation]()