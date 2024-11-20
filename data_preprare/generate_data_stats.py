import os
import pandas as pd
from typing import Dict

from utils import load_dataset

FILDE_TO_SAVE = "std"
TRAIN_NAME_FILE = os.path.join(FILDE_TO_SAVE, "train_stats.xlsx")
TEST_NAME_FILE = os.path.join(FILDE_TO_SAVE, "test_stats.xlsx")

def generate_data_stats(df: pd.DataFrame) -> Dict:
    
    lan = df['language'].value_counts()
    class_ = df['label'].value_counts()
    class_lan = df.groupby(['language', 'label']).size().unstack()

    return {
        'lan': lan,
        'class': class_,
        'class_lan': class_lan
    }


def save_data_stats(dict_std: Dict, name_file: str) -> None:
    with pd.ExcelWriter(name_file) as writer:
        dict_std['lan'].to_excel(writer, sheet_name='lan')
        dict_std['class'].to_excel(writer, sheet_name='class')
        dict_std['class_lan'].to_excel(writer, sheet_name='class_lan')


if __name__ == '__main__':
    train_df, _ = load_dataset()
    dict_std = generate_data_stats(train_df)
    save_data_stats(dict_std, TRAIN_NAME_FILE)
    print('Data saved in excel!')