from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from datasets import load_from_disk, Dataset, DatasetDict
from kneed import KneeLocator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import pickle
# not warnings
import warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
warnings.filterwarnings("ignore")


from utils import SEED, generate_training_prompt

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


dataset_path = "data/02-processed/spanish"
embedding_model_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model_spacy = None # None or 'es_core_news_sm'
distance_metric = 'cosine'
name_new_dataset = "data/03-combined/spanish_paragraphs_clustering"
percentage_of_data = None
top_k_sents = None

def load_dataset_and_model(dataset_path: str, embedding_model_path: str) -> tuple:
    dataset = load_from_disk(dataset_path)
    model   = SentenceTransformer(embedding_model_path)
    return dataset, model

def process_text_into_sentences(text, model_spacy):
    nlp = spacy.load(model_spacy, disable=["tagger", "parser", "ner"])
    nlp.add_pipe('sentencizer')
    doc = nlp(text)
    return doc

def process_text_into_paragraphs(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    return docs

def generate_embeddings_from_paragraphs(docs, model, get_sentences=False):
    paragraphs = [doc.page_content for doc in docs]
    embeddings = model.encode(paragraphs)
    #embeddings = StandardScaler().fit_transform(embeddings)
    return embeddings if not get_sentences else embeddings, paragraphs

def generate_embeddings(doc, model, get_sentences=False):
    sentences  = list(sent.text for sent in doc.sents)
    embeddings = model.encode(sentences)
    embeddings = StandardScaler().fit_transform(embeddings)
    return embeddings if not get_sentences else embeddings, sentences

def build_semantic_paragraph(sentences, clusters, useful_clusters, k_opt) -> dict:
    cluster_phrases = {i: [] for i in range(k_opt)}

    for sentence, cluster_id in zip(sentences, clusters):
        if cluster_id in useful_clusters:
            cluster_phrases[cluster_id].append(sentence)

    # reordena las frases en base a la posición original en cada key
    for cluster_id in cluster_phrases.keys():
        cluster_phrases[cluster_id] = " ".join(sorted(
            cluster_phrases[cluster_id], 
            key=lambda x: sentences.index(x)
        ))
    
    return cluster_phrases


def find_optimal_clusters(embeddings, seed=SEED, max_clusters=30, min_clusters=2, n_jobs=-1):
    """
    Finds the optimal number of clusters using the elbow method, with parallelized computations.

    Parameters:
        embeddings (ndarray): Embedding matrix for clustering.
        seed (int): Random seed for reproducibility.
        max_clusters (int): Maximum number of clusters to consider.
        min_clusters (int): Minimum number of clusters to consider.
        n_jobs (int): Number of parallel processes (-1 to use all available cores).

    Returns:
        int: Optimal number of clusters.
    """
    n_samples = embeddings.shape[0]
    cluster_range = range(min_clusters, min(max_clusters, n_samples))
    
    if len(cluster_range) == 0:
        raise ValueError("The cluster range is invalid. Check the input data.")

    # Auxiliary function for computing inertia
    def compute_inertia(n_clusters):
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=seed, 
            n_init="auto", 
            init="k-means++"
        ).fit(embeddings)
        return kmeans.inertia_

    # Parallelized computation of inertia values for each number of clusters
    inertia_values = Parallel(n_jobs=n_jobs)(
        delayed(compute_inertia)(n_clusters) for n_clusters in cluster_range
    )

    # Identify the elbow in the inertia curve
    knee_locator = KneeLocator(
        cluster_range, inertia_values, curve="convex", direction="decreasing"
    )
    k_opt = knee_locator.knee

    if k_opt is None:
        k_opt = min_clusters

    return k_opt

def cluster_sentences(embeddings, k_opt) -> tuple:
    kmeans = KMeans(n_clusters=k_opt, random_state=SEED)
    clusters = kmeans.fit_predict(embeddings)
    return kmeans, clusters

def compact_text_representation(doc, significant_phrases, tokenizer, original_text_len):
    all_significant_phrases = [
        phrase for _, phrases in significant_phrases.items() for phrase in phrases
    ]
    ordered_phrases = sorted(all_significant_phrases, key=lambda x: list(doc.sents).index(x))
    compact_representation = " ".join([str(phrase) for phrase in ordered_phrases])

    tokens = tokenizer.tokenize(compact_representation)
    compact_representation_len = len(tokens)

    print("Representación compacta del texto:")
    #print(compact_representation)
    print(f"Original: {original_text_len} caracteres")
    print(f"Compacto: {compact_representation_len} caracteres")
    print(f"Reducción: {1 - compact_representation_len / original_text_len:.2%}")


def create_dataset(dataset, model, model_spacy):
    embeddings_dataset = {
        "sample": [],
        "label": [],
    }
    new_dataset = {
        "instruction": [],
        "input": [],
        "output": [],
        "text": [],
        "language": [],
    }
    for data in tqdm(dataset, desc="Creating dataset"):
        text = data['input']
        test = data['output']

        try:
            if model_spacy:
                doc = process_text_into_sentences(text, model_spacy)
                embeddings, setences = generate_embeddings(doc, model, get_sentences=True)
            else:
                doc = process_text_into_paragraphs(text)
                embeddings, setences = generate_embeddings_from_paragraphs(doc, model)

            k_opt = find_optimal_clusters(embeddings)
            kmeans, cluster_train = cluster_sentences(embeddings, k_opt)
            train_centroids = kmeans.cluster_centers_


            if model_spacy:
                doc_test = process_text_into_sentences(test, model_spacy)
                embeddings_test, setences_test = generate_embeddings(doc_test, model, get_sentences=True)
            else:
                doc_test = process_text_into_paragraphs(test)
                embeddings_test, setences_test = generate_embeddings_from_paragraphs(doc_test, model)

            k_opt_test = find_optimal_clusters(embeddings_test)
            kmeans_test, cluster_test = cluster_sentences(embeddings_test, k_opt_test)
            test_centroids = kmeans_test.cluster_centers_
        except Exception as e:
            print(f"Error: {e}")
            continue


        distances = cdist(test_centroids, train_centroids, metric=distance_metric)
        closest_clusters = distances.argmin(axis=1)

        dict_clusters = { i:[] for i in range(k_opt) }
        for i, cluster in enumerate(closest_clusters):
            if cluster in dict_clusters:
                dict_clusters[cluster].append(i)
            else:
                dict_clusters[cluster] = [i]

        y = [1 if len(dict_clusters[i]) > 0 else 0 for i in range(k_opt)]

        embeddings_dataset["sample"] += train_centroids.tolist()
        embeddings_dataset["label"]  += y

        # get the list of index of train centroids that the label is 1
        index_train = [i for i, label in enumerate(y) if label == 1]
        
        train_paragraphs = build_semantic_paragraph(setences, cluster_train, index_train, k_opt)
        test_paragraphs = build_semantic_paragraph(setences_test, cluster_test, range(k_opt_test), k_opt_test)

        # link the paragraphs
        for index in index_train:
            list_index_summary = dict_clusters[index]
            summary = " ".join([test_paragraphs[i] for i in list_index_summary])

            new_dataset["instruction"].append(data["instruction"])
            new_dataset["input"].append(train_paragraphs[index])
            new_dataset["output"].append(summary)
            text = generate_training_prompt(data["instruction"], train_paragraphs[index], summary)
            new_dataset["text"].append(text)
            new_dataset["language"].append(data["language"])
    
    new_dataset = Dataset.from_dict(new_dataset)

    return embeddings_dataset, new_dataset

def save_dataset(new_dataset, name_new_dataset):
    with open(name_new_dataset, "wb") as f:
        pickle.dump(new_dataset, f)


def get_sample(dataset, num_samples):
    if num_samples:
        dataset = dataset.shuffle(seed=SEED).select(range(int(num_samples * dataset.num_rows)))
    return dataset

def main():

    dataset, model = load_dataset_and_model(dataset_path, embedding_model_path)

    train_dataset = get_sample(dataset["train"], percentage_of_data)
    train_dataset_cluster, new_train_dataset = create_dataset(train_dataset, model, model_spacy)

    validation_dataset = get_sample(dataset["validation"], percentage_of_data)
    validation_dataset_cluster, new_validation_dataset = create_dataset(validation_dataset, model, model_spacy)

    test_dataset = get_sample(dataset["test"], percentage_of_data)
    test_dataset_cluster, new_test_dataset = create_dataset(test_dataset, model, model_spacy)

    new_dataset = DatasetDict({
        "train": new_train_dataset,
        "validation": new_validation_dataset,
        "test": new_test_dataset
    }) 

    new_dataset.save_to_disk(name_new_dataset)
    save_dataset(train_dataset_cluster, f"{name_new_dataset}/clustring_embedding_train.pkl")
    save_dataset(validation_dataset_cluster, f"{name_new_dataset}/clustring_embedding_validation.pkl")
    save_dataset(test_dataset_cluster, f"{name_new_dataset}/clustring_embedding_test.pkl")
    
    


if __name__ == '__main__':
    main()