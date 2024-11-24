import os
import pandas as pd
from typing import Dict
from datasets import load_dataset

FILDE_TO_SAVE = "std"



if __name__ == '__main__':
    dataset = "data/01-raw/en"
    data = load_dataset(dataset, split='train')
    print(data)