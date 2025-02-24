import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import os
from lightning import seed_everything
from utils import SEED
import numpy as np
from lazypredict.Supervised import LazyClassifier

NAME_DATASET = [
    "data/04-clustering/english-chunks-sentence-transformers",
    "data/04-clustering/spanish-chunks-sentence-transformers",
    "data/04-clustering/french-chunks-sentence-transformers",
    "data/04-clustering/german-chunks-sentence-transformers",
    "data/04-clustering/italian-chunks-sentence-transformers",
    "data/04-clustering/portuguese-chunks-sentence-transformers",
    "data/04-clustering/spanish-english-chunks-sentence-transformers",
    "data/04-clustering/spanish-french-chunks-sentence-transformers",
    "data/04-clustering/spanish-german-chunks-sentence-transformers",
    "data/04-clustering/spanish-italian-chunks-sentence-transformers",
    "data/04-clustering/spanish-portuguese-chunks-sentence-transformers",
    "data/04-clustering/english-german-chunks-sentence-transformers",
]
TINY = False

TRAIN_DATASET = "clustring_embedding_train.pkl"
TEST_DATASET = "clustring_embedding_test.pkl"
VALIDATION_DATASET = "clustring_embedding_validation.pkl"

def read_pickle(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as f:
        return pd.DataFrame(pickle.load(f))
    
def balance_dataset(dataset_train):
    class_majority = dataset_train[dataset_train['label'] == 0]
    class_minority = dataset_train[dataset_train['label'] == 1]
    class_majority_downsampled = class_majority.sample(n=len(class_minority), random_state=SEED)
    df_balanced = pd.concat([class_majority_downsampled, class_minority])
    return df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
def lazy_classification(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=SEED)
    models, _ = clf.fit(X_train, X_test, y_train, y_test)
    return models

def save_predictions_to_csv(predictions, folder=None, filename="clf_models.csv"):
    assert folder is not None, "Please provide a folder to save the predictions."
    predictions_df = pd.DataFrame(predictions)
    file_path = os.path.join(folder, filename)
    predictions_df.to_csv(file_path, index=True, sep=';')
    print(f"Predictions saved to {file_path}")
    
def train_clfs(X_train, y_train, X_test, y_test, dataset_train_path):
    models = lazy_classification(X_train, X_test, y_train, y_test)
    folder_to_save = "/".join([f for f in dataset_train_path.split("/")[:-1]])
    save_predictions_to_csv(models, folder=folder_to_save)

    print("#"*50)
    print(models.head(5))

def set_seed(seed=SEED):
    seed_everything(seed)

def experiments(name_dataset: str):
    dataset_train_path = f"{name_dataset}/{TRAIN_DATASET}"
    dataset_val_path = f"{name_dataset}/{VALIDATION_DATASET}"
    dataset_test_path = f"{name_dataset}/{TEST_DATASET}"


    dataset_train = read_pickle(dataset_train_path)
    dataset_val = read_pickle(dataset_val_path)
    dataset_test = read_pickle(dataset_test_path)
    dataset = pd.concat([dataset_train, dataset_val], ignore_index=True)
    df_balanced = balance_dataset(dataset)
    
    if TINY:
        df_balanced = df_balanced.sample(frac=0.1, random_state=SEED)

    X_train, y_train = np.array(df_balanced["sample"].to_list()), np.array(df_balanced["label"].to_list())
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)

    train_clfs(X_train, y_train, X_test, y_test, dataset_train_path)

def main():
    set_seed()
    for name_dataset in NAME_DATASET:
        print(f"Running experiments for {name_dataset}")
        experiments(name_dataset)

    

if __name__ == "__main__":
    main()