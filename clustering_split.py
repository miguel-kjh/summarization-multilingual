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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
warnings.filterwarnings("ignore")
from typing import Tuple

from text2embeddings import Text2Embeddings, Text2EmbeddingsOpenAI, Text2EmbeddingsSetenceTransforms
from utils import SEED, generate_training_prompt

import argparse
import os
from distutils.util import strtobool
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process constants for dataset and model configuration.")

    parser.add_argument("--dataset_path", type=str, default="data/02-processed/spanish",
                        help="Path to the dataset to be processed.")
    parser.add_argument("--method", type=str, default="paragraphs",
                        help="Method to use for processing ('sentences', 'paragraphs'. 'chunks').")
    parser.add_argument("--embedding_model", type=str, choices=["openai", "sentence-transformers"], default="sentence-transformers",
                        help="Embedding model to use ('openai' or 'sentence-transformers').")
    parser.add_argument("--model_spacy", type=str, default="es_core_news_sm",
                        help="SpaCy model to use (e.g., 'es_core_news_sm') or None.")
    parser.add_argument("--distance_metric", type=str, default="cosine",
                        help="Distance metric to use (e.g., 'cosine').")
    parser.add_argument("--percentage_of_data", type=float, default=None,
                        help="Percentage of data to process (e.g., 0.01 for 1%).")
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=False,
                        help="Flag to enable logging with Weights & Biases.")

    args = parser.parse_args()
    language = args.dataset_path.split("/")[-1]
    name_new_dataset = f"data/04-clustering/{language}-{args.method}-{args.embedding_model}"
    args.name_new_dataset = name_new_dataset
    return args

def load_dataset_and_model(dataset_path: str, embedding_model: str) -> Tuple[Dataset, Text2Embeddings]:
    dataset = load_from_disk(dataset_path)
    model = None

    if embedding_model == "openai":
        model = Text2EmbeddingsOpenAI(model_name="text-embedding-3-large")
    elif embedding_model == "sentence-transformers":
        model = Text2EmbeddingsSetenceTransforms(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    else:
        raise ValueError("Invalid embedding model. Choose 'openai' or 'sentence-transformers'.")

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
    embeddings = model.transform(paragraphs)
    return embeddings if not get_sentences else embeddings, paragraphs

def generate_embeddings(doc, model, get_sentences=False):
    sentences  = list(sent.text for sent in doc.sents)
    embeddings = model.transform(sentences)
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

def evaluate_clustering(data, labels) -> dict:
    metrics = {
        "silhouette_score": silhouette_score(data, labels),
        "calinski_harabasz_score": calinski_harabasz_score(data, labels),
        "davies_bouldin_score": davies_bouldin_score(data, labels),
    }
    return metrics

def cluster_sentences(embeddings, k_opt) -> tuple:
    kmeans = KMeans(n_clusters=k_opt, random_state=SEED)
    clusters = kmeans.fit_predict(embeddings)
    metrics = evaluate_clustering(embeddings, clusters)
    return kmeans, clusters, metrics

def save_dataset(new_dataset, name_new_dataset):
    with open(name_new_dataset, "wb") as f:
        pickle.dump(new_dataset, f)


def get_sample(dataset, num_samples):
    if num_samples:
        dataset = dataset.shuffle(seed=SEED).select(range(int(num_samples * dataset.num_rows)))
    return dataset

class DocumentSplitter:

    def __init__(self, model: Text2Embeddings):
        self.model = model

    def process_data_entry(self, data, embeddings_dataset, new_dataset, final_metrics):
        pass

    def initialize_embeddings_dataset(self):
        return {"sample": [], "label": []}

    def initialize_new_dataset(self):
        return {
            "instruction": [],
            "input": [],
            "output": [],
            "text": [],
            "language": [],
            "original_index_document": [],
            "original_document": [],
            "original_summary": [],
        }

    def initialize_metrics(self):
        return {
            "documents": {
                "silhouette_score": [],
                "calinski_harabasz_score": [],
                "davies_bouldin_score": [],
            },
            "summaries": {
                "silhouette_score": [],
                "calinski_harabasz_score": [],
                "davies_bouldin_score": [],
            },
        }

    def process_document(self, text: str, model_spacy: str):
        sentences = process_text_into_sentences(text, model_spacy)
        embeddings, sentences = generate_embeddings(sentences, self.model, get_sentences=True)
        return {"embeddings": embeddings, "sentences": sentences}

    def generate_clusters_and_metrics(self, document_data):
        embeddings = document_data['embeddings']
        sentences = document_data['sentences']

        k_opt = find_optimal_clusters(embeddings)
        kmeans, cluster_labels, clustering_metrics = cluster_sentences(embeddings, k_opt)

        return {
            "centroids": kmeans.cluster_centers_,
            "clusters": cluster_labels,
            "metrics": clustering_metrics,
            "sentences": sentences,
            "k_opt": k_opt
        }

    def update_metrics(self, metrics, document_metrics, summary_metrics):
        for key in metrics["documents"]:
            metrics["documents"][key].append(document_metrics[key])
            metrics["summaries"][key].append(summary_metrics[key])

    def update_datasets(self, embeddings_dataset, new_dataset, train_data, test_data, original_data):
        distances = cdist(test_data['centroids'], train_data['centroids'], metric='cosine')
        closest_clusters = distances.argmin(axis=1)

        dict_clusters = {i: [] for i in range(train_data['k_opt'])}
        for i, cluster in enumerate(closest_clusters):
            dict_clusters[cluster].append(i)

        y = [1 if len(dict_clusters[i]) > 0 else 0 for i in range(train_data['k_opt'])]

        embeddings_dataset["sample"] += train_data['centroids'].tolist()
        embeddings_dataset["label"] += y

        index_train = [i for i, label in enumerate(y) if label == 1]
        train_paragraphs = build_semantic_paragraph(
            train_data['sentences'], train_data['clusters'], index_train, train_data['k_opt']
        )
        test_paragraphs = build_semantic_paragraph(
            test_data['sentences'], test_data['clusters'], range(test_data['k_opt']), test_data['k_opt']
        )

        instruction = original_data['instruction']
        language = original_data['language']
        original_index_document = original_data['index']
        original_document = original_data['input']
        original_summary = original_data['output']

        for index in index_train:
            list_index_summary = dict_clusters[index]
            summary = " ".join([test_paragraphs[i] for i in list_index_summary]).strip()

            new_dataset["instruction"].append(instruction)
            new_dataset["input"].append(train_paragraphs[index])
            new_dataset["output"].append(summary)

            text = generate_training_prompt(instruction, train_paragraphs[index], summary)
            new_dataset["text"].append(text)
            new_dataset["language"].append(language)

            new_dataset["original_index_document"].append(original_index_document)
            new_dataset["original_document"].append(original_document)
            new_dataset["original_summary"].append(original_summary)

    def compute_average_metrics(self, metrics):
        for key in metrics["documents"]:
            metrics["documents"][key] = np.mean(metrics["documents"][key])
            metrics["summaries"][key] = np.mean(metrics["summaries"][key])
        return metrics

    def create_dataset(self, dataset: Dataset) -> tuple:
        embeddings_dataset = self.initialize_embeddings_dataset()
        new_dataset = self.initialize_new_dataset()
        final_metrics = self.initialize_metrics()

        for index, data in tqdm(enumerate(dataset), total=len(dataset), desc="Creating dataset"):
            try:
                data['index'] = index
                self.process_data_entry(
                    data,
                    embeddings_dataset,
                    new_dataset,
                    final_metrics,
                )
            except Exception as e:
                print(f"Error processing data entry: {e}")
                continue

        final_metrics = self.compute_average_metrics(final_metrics)
        new_dataset = Dataset.from_dict(new_dataset)
        return embeddings_dataset, new_dataset, final_metrics

class DocumentSplitterSentences(DocumentSplitter):
    def __init__(self, model: Text2Embeddings, model_spacy: str):
        super().__init__(model)
        self.model_spacy = model_spacy

    def process_data_entry(self, data, embeddings_dataset, new_dataset, final_metrics):
        document_data = self.process_document(data['input'], self.model_spacy)
        summary_data = self.process_document(data['output'], self.model_spacy)

        train_data = self.generate_clusters_and_metrics(document_data)
        test_data = self.generate_clusters_and_metrics(summary_data)

        self.update_metrics(final_metrics, train_data['metrics'], test_data['metrics'])

        self.update_datasets(
            embeddings_dataset,
            new_dataset,
            train_data,
            test_data,
            data,
        )

    
class DocumentSplitterParagraphs(DocumentSplitter):
    def __init__(self, model: Text2Embeddings):
        super().__init__(model)

    def process_paragraph(self, text: str):
        paragraphs = process_text_into_paragraphs(text)
        embeddings, paragraphs = generate_embeddings_from_paragraphs(paragraphs, self.model, get_sentences=True)
        return {"embeddings": embeddings, "sentences": paragraphs}

    def process_data_entry(self, data, embeddings_dataset, new_dataset, final_metrics):
        document_data = self.process_paragraph(data['input'])
        summary_data = self.process_paragraph(data['output'])

        train_data = self.generate_clusters_and_metrics(document_data)
        test_data = self.generate_clusters_and_metrics(summary_data)

        self.update_metrics(final_metrics, train_data['metrics'], test_data['metrics'])

        self.update_datasets(
            embeddings_dataset,
            new_dataset,
            train_data,
            test_data,
            data,
        )

class DocumentSplitterChunks(DocumentSplitterParagraphs):

    def __init__(self, model: Text2Embeddings):
        super().__init__(model)
        self.chunker = SemanticChunker(model.model)

    def generate_clusters_and_metrics(self, document_data):
        embeddings = document_data['embeddings']
        sentences = document_data['sentences']

        k_opt = find_optimal_clusters(embeddings)
        kmeans, cluster_labels, clustering_metrics = cluster_sentences(embeddings, k_opt)

        return {
            "centroids": kmeans.cluster_centers_,
            "clusters": cluster_labels,
            "metrics": clustering_metrics,
            "sentences": sentences,
            "k_opt": k_opt
        }

    def process_text_into_paragraphs(self, text):
        return self.chunker.create_documents([text])


def main():
    args = parse_arguments()
    dataset, model = load_dataset_and_model(args.dataset_path, args.embedding_model)
    document_splitter = None
    if args.method == "sentences":
        document_splitter = DocumentSplitterSentences(model, args.model_spacy)
    elif args.method == "paragraphs":
        document_splitter = DocumentSplitterParagraphs(model)
    elif args.method == "chunks":
        document_splitter = DocumentSplitterChunks(model)
    else:
        raise ValueError("Invalid method. Choose 'sentences', 'paragraphs', or 'chunks'.")

    train_dataset = get_sample(dataset["train"], args.percentage_of_data)
    train_dataset_cluster, new_train_dataset, metrics = document_splitter.create_dataset(train_dataset)

    validation_dataset = get_sample(dataset["validation"], args.percentage_of_data)
    validation_dataset_cluster, new_validation_dataset, metrics = document_splitter.create_dataset(validation_dataset)

    test_dataset = get_sample(dataset["test"], args.percentage_of_data)
    test_dataset_cluster, new_test_dataset, metrics = document_splitter.create_dataset(test_dataset)

    new_dataset = DatasetDict({
        "train": new_train_dataset,
        "validation": new_validation_dataset,
        "test": new_test_dataset
    }) 

    new_dataset.save_to_disk(args.name_new_dataset)
    save_dataset(train_dataset_cluster, f"{args.name_new_dataset}/clustring_embedding_train.pkl")
    save_dataset(validation_dataset_cluster, f"{args.name_new_dataset}/clustring_embedding_validation.pkl")
    save_dataset(test_dataset_cluster, f"{args.name_new_dataset}/clustring_embedding_test.pkl")
    # sace metrics
    with open(f"{args.name_new_dataset}/metrics_{args.method}_{args.embedding_model}.pkl", "wb") as f:
        pickle.dump(metrics, f)

    if args.wandb:
        import wandb
        wandb.init(
            project="clustering_eur_lex_sum",
            entity="miguel_kjh",
            name=f"clustering_{args.method}_{args.embedding_model}"
        )
        wandb.log(metrics)
    

if __name__ == '__main__':
    main()