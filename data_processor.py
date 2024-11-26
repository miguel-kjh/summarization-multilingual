import os
import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk

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
        for split in dataset.keys():
            print(f"Processing {split} split for {lang}")
            instructions = transform.generate_instructions(dataset[split], lang)
            file = os.path.join(dataset_it_name, f"{split}.json")
            with open(file, 'w') as f:
                json.dump(instructions, f)

def combine():
    # generate a tiny dataset for testing using en
    print("Combining data")
    lang = "en"
    dataset_tiny_name = os.path.join(COMBINED_DATA_FOLDER, "tiny")
    os.makedirs(dataset_tiny_name, exist_ok=True)
    dataset = json.load(open(os.path.join(PROCESS_DATA_FOLDER, lang, "train.json")))
    dataset_train = dataset[:100]
    dataset = json.load(open(os.path.join(PROCESS_DATA_FOLDER, lang, "validation.json")))
    dataset_val = dataset[200:210]
    dataset = json.load(open(os.path.join(PROCESS_DATA_FOLDER, lang, "test.json")))
    dataset_test = dataset[300:310]
    # save the tiny dataset
    with open(os.path.join(dataset_tiny_name, "train.json"), 'w') as f:
        json.dump(dataset_train, f)
    with open(os.path.join(dataset_tiny_name, "validation.json"), 'w') as f:
        json.dump(dataset_val, f)
    with open(os.path.join(dataset_tiny_name, "test.json"), 'w') as f:
        json.dump(dataset_test, f)


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