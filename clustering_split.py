from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from datasets import load_from_disk
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
warnings.filterwarnings("ignore")

from utils import SEED


dataset_path = "data/02-processed/spanish"
embedding_model_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model_spacy = 'es_core_news_sm'
distance_metric = 'cosine'
name_new_dataset = "data/02-processed/spanish/train_cluster.pkl"
number_samples = None
top_k_sents = None

def load_dataset_and_model(dataset_path: str, embedding_model_path: str) -> tuple:
    dataset = load_from_disk(dataset_path)
    model   = SentenceTransformer(embedding_model_path)
    return dataset, model

def process_text(text, model_spacy):
    nlp = spacy.load(model_spacy, disable=["tagger", "parser", "ner"])
    nlp.add_pipe('sentencizer')
    doc = nlp(text)
    return doc

def generate_embeddings(doc, model):
    sentences  = list(sent.text for sent in doc.sents)
    embeddings = model.encode(sentences)
    return embeddings

def find_optimal_clusters(embeddings, seed=SEED, max_clusters=100, min_clusters=5):
    """
    Encuentra el número óptimo de clusters usando el método del codo.

    Parameters:
        embeddings (ndarray): Matriz de embeddings para clustering.
        seed (int): Semilla para reproducibilidad.
        max_clusters (int): Máximo número de clusters a considerar.
        min_clusters (int): Mínimo número de clusters a considerar.

    Returns:
        int: Número óptimo de clusters.
    """
    n_samples = embeddings.shape[0]
    cluster_range = range(min_clusters, min(max_clusters, n_samples))
    
    if len(cluster_range) == 0:
        raise ValueError("El rango de clusters es inválido. Revisa la entrada de datos.")

    # Prealocar la lista para evitar el overhead del append.
    inertia_values = np.zeros(len(cluster_range))

    # Calcular KMeans para cada número de clusters en paralelo.
    for i, n_clusters in enumerate(cluster_range):
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=seed, 
            n_init="auto",
            init="k-means++"
        ).fit(embeddings)
        inertia_values[i] = kmeans.inertia_

    # Identificar el codo en la curva de inercia.
    knee_locator = KneeLocator(
        cluster_range, inertia_values, curve="convex", direction="decreasing"
    )
    k_opt = knee_locator.knee

    if k_opt is None:
        raise ValueError("No se pudo encontrar un 'knee' en la curva de inercia.")

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
    new_dataset = {
        "sample": [],
        "label": [],
    }
    for data in tqdm(dataset, desc="Creating dataset"):
        text = data['input']
        test = data['output']

        doc = process_text(text, model_spacy)
        embeddings = generate_embeddings(doc, model)
        k_opt = find_optimal_clusters(embeddings)
        kmeans, _ = cluster_sentences(embeddings, k_opt)
        train_centroids = kmeans.cluster_centers_

        doc_test = process_text(test, model_spacy)
        embeddings_test = generate_embeddings(doc_test, model)
        k_opt_test = find_optimal_clusters(embeddings_test)
        kmeans_test, _ = cluster_sentences(embeddings_test, k_opt_test)
        test_centroids = kmeans_test.cluster_centers_

        distances = cdist(test_centroids, train_centroids, metric=distance_metric)
        closest_clusters = distances.argmin(axis=1)

        dict_clusters = { i:[] for i in range(k_opt) }
        for i, cluster in enumerate(closest_clusters):
            if cluster in dict_clusters:
                dict_clusters[cluster].append(i)
            else:
                dict_clusters[cluster] = [i]

        y = [1 if len(dict_clusters[i]) > 0 else 0 for i in range(k_opt)]

        new_dataset["sample"] += train_centroids.tolist()
        new_dataset["label"]  += y

    return new_dataset

def save_dataset(new_dataset, name_new_dataset):
    with open(name_new_dataset, "wb") as f:
        pickle.dump(new_dataset, f)


def get_sample(dataset, num_samples):
    if num_samples:
        dataset = dataset.shuffle(seed=SEED).select(range(int(num_samples * dataset.num_rows)))
    return dataset

def main():

    dataset, model = load_dataset_and_model(dataset_path, embedding_model_path)

    train_dataset = get_sample(dataset["train"], number_samples)
    train_dataset_cluster = create_dataset(train_dataset, model, model_spacy)
    save_dataset(train_dataset_cluster, f"{name_new_dataset}_train.pkl")

    validation_dataset = get_sample(dataset["validation"], number_samples)
    validation_dataset_cluster = create_dataset(validation_dataset, model, model_spacy)
    save_dataset(validation_dataset_cluster, f"{name_new_dataset}_validation.pkl")

    test_dataset = get_sample(dataset["test"], number_samples)
    test_dataset_cluster = create_dataset(test_dataset, model, model_spacy)
    save_dataset(test_dataset_cluster, f"{name_new_dataset}_test.pkl")
    


if __name__ == '__main__':
    main()