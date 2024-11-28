import os
import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict

from data_preprare.transform_data import TransformData
from data_preprare.generate_data_stats import StatsGenerator
from data_preprare.download_dataset import download_dataset
from utils import LANGUAGES, RAW_DATA_FOLDER,  FILE_STATS, PROCESS_DATA_FOLDER, COMBINED_DATA_FOLDER

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
        dataset_dict = DatasetDict()
        for split in dataset.keys():
            print(f"Processing {split} split for {lang}")
            instructions = transform.generate_instructions(dataset[split], lang)
            print("Saving dataset 1" )
            dataset_dict[split] = instructions
            print("Saving dataset 2")
        dataset_dict.save_to_disk(dataset_it_name)

def combine():
    # generate a tiny dataset for testing using en
    print("Combining data")
    lang = "en"
    dataset_tiny_name = os.path.join(COMBINED_DATA_FOLDER, "tiny")
    os.makedirs(dataset_tiny_name, exist_ok=True)
    dataset = load_from_disk(os.path.join(PROCESS_DATA_FOLDER, lang))
    # take 0.1% of the data
    dataset_small = DatasetDict()
    for split in dataset.keys():
        dataset_small[split] = dataset[split].shuffle(seed=42).select(range(len(dataset[split]) // 1000))
    dataset_small.save_to_disk(dataset_tiny_name)


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
    "combine": combine,
}


if __name__ == '__main__':
    args = parse_args()
    operation = args.operation
    assert operation in OPERATIONS, f"Operation {operation} not found"
    OPERATIONS[operation]()