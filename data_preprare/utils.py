import os
from typing import Tuple
import pandas as pd

FILE = "/home/miguel/data/01-raw/finetuning/contradictory_my_dear_watson"
TRAIN = os.path.join(FILE, "train.csv")
TEST = os.path.join(FILE, "test.csv")

def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)

    return train_df, test_df