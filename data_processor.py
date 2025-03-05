import os
import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, concatenate_datasets

from data_preprare.transform_data import TransformData, TransformDataReduce, TransformDataCanario
from data_preprare.generate_data_stats import StatsGenerator
from data_preprare.download_dataset import download_dataset, download_canary_parlament
from utils import LANGUAGES, RAW_DATA_FOLDER,  FILE_STATS, PROCESS_DATA_FOLDER, COMBINED_DATA_FOLDER

def download():
    for lang in LANGUAGES:
        download_dataset(lang, RAW_DATA_FOLDER)
    print("All datasets downloaded and saved")

def download_canary():
    download_canary_parlament(RAW_DATA_FOLDER)

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
            dataset_dict[split] = instructions
        dataset_dict.save_to_disk(dataset_it_name)

def process_canary():
    print("Processing data")
    transform = TransformDataCanario()
    dataset_name = os.path.join(RAW_DATA_FOLDER, "canario")
    dataset = load_from_disk(dataset_name)

    dataset_it_name = os.path.join(PROCESS_DATA_FOLDER, "canario")
    os.makedirs(dataset_it_name, exist_ok=True)
    dataset_dict = DatasetDict()
    for split in dataset.keys():
        print(f"Processing {split} split for canario")
        instructions = transform.generate_instructions(dataset[split])
        dataset_dict[split] = instructions
    dataset_dict.save_to_disk(dataset_it_name)

def process_reduce():
    print("Processing data")
    transform = TransformDataReduce()
    for lang in LANGUAGES:
        dataset_name = os.path.join(RAW_DATA_FOLDER, lang)
        dataset = load_from_disk(dataset_name)
        dataset_it_name = os.path.join(PROCESS_DATA_FOLDER, f"{lang}_reduce")
        os.makedirs(dataset_it_name, exist_ok=True)
        dataset_dict = DatasetDict()
        for split in dataset.keys():
            print(f"Processing {split} split for {lang}")
            instructions = transform.generate_instructions(dataset[split], lang)
            dataset_dict[split] = instructions
        dataset_dict.save_to_disk(dataset_it_name)

def combine():
    # generate a tiny dataset for testing using en
    print("Combining data")
    dataset2combine = [
        # romance
        ("data/02-processed/spanish", "data/02-processed/portuguese"),
        ("data/02-processed/spanish", "data/02-processed/french"),
        ("data/02-processed/spanish", "data/02-processed/italian"),
        
        # spanish - non romance
        ("data/02-processed/spanish", "data/02-processed/german"),
        ("data/02-processed/spanish", "data/02-processed/english"),

        # non romance
        ("data/02-processed/english", "data/02-processed/german"),
        
    ]

    for lang1, lang2 in dataset2combine:

        dataset1 = load_from_disk(lang1)
        dataset2 = load_from_disk(lang2)
        new_dataset = DatasetDict()

        dataset1_train = dataset1["train"]
        dataset2_train = dataset2["train"]

        combined_dataset = concatenate_datasets([dataset1_train, dataset2_train])
        new_dataset["train"] = combined_dataset
        # shuffle
        new_dataset["train"] = new_dataset["train"].shuffle()

        dataset1_val = dataset1["validation"]
        dataset2_val = dataset2["validation"]

        combined_dataset = concatenate_datasets([dataset1_val, dataset2_val])
        new_dataset["validation"] = combined_dataset
        new_dataset["validation"] = new_dataset["validation"].shuffle()

        dataset1_test = dataset1["test"]
        dataset2_test = dataset2["test"]

        combined_dataset = concatenate_datasets([dataset1_test, dataset2_test])
        new_dataset["test"] = combined_dataset
        new_dataset["test"] = new_dataset["test"].shuffle()


        name = f"{os.path.basename(lang1)}-{os.path.basename(lang2)}"
        combine_dataset_name = os.path.join(COMBINED_DATA_FOLDER, name)

        # PRINT STATS
        print("Combined dataset stats")

        for split in new_dataset.keys():

            print(f"Split: {split}")
            print("Number of samples", len(new_dataset[split]))

        new_dataset.save_to_disk(combine_dataset_name)


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
    "download_canary": download_canary,
    "stats": stats,
    "process": process,
    "process_reduce": process_reduce,
    "process_canary": process_canary,
    "combine": combine,
}


if __name__ == '__main__':
    args = parse_args()
    operation = args.operation
    assert operation in OPERATIONS, f"Operation {operation} not found"
    OPERATIONS[operation]()