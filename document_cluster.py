import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any, List, Tuple, Dict
from scipy.spatial.distance import cosine
import spacy
import networkx as nx
from kneed import KneeLocator

from utils import SEED
from clustering_split import generate_embeddings, find_optimal_clusters, cluster_sentences


class DocumentClusterer:

    def __init__(
            self, 
            embedding_model: SentenceTransformer, 
            spacy_model: str,
        ):
        """
        Initialize the class with the embedding model, text splitter, and number of clusters.
        
        Args:
            embedding_model (SentenceTransformer): Pretrained embedding model.
            spacy_model (str): Spacy model to use for tokenization sents.
        """
        self.embedding_model = embedding_model
        self.nlp = spacy.load(
            spacy_model,
            disable=["tagger", "parser", "ner"]
        )
        self.nlp.add_pipe('sentencizer')

        tokenizer = self.embedding_model.tokenizer

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.embedding_model.resize_token_embeddings(len(tokenizer))


    def cluster_and_assign(self, document: str) -> List[str]:
        raise NotImplementedError("Method 'cluster_and_assign' must be implemented in a subclass.")
    

class DocumentClustererKMeans(DocumentClusterer):

    def __init__(
            self,
            embedding_model: SentenceTransformer,
            spacy_model: str,
            clf: Any,
        ):
        """
        Initialize the class with the embedding model, text splitter, and number of clusters.
        """
        super().__init__(embedding_model, spacy_model)
        self.clf = clf

    
    def cluster_and_assign(self, document: str, min_clusters: int = 5, max_clusters: int = 100) -> List[str]:
        
        """Processes the document and summary to cluster document parts and assign summary parts to clusters.
        
        Args:
            document (str): The full text of the document to be clustered.
        
        Returns:
            List[str]: A list of strings, each representing a cluster of the document.
        """
        
        doc = self.nlp(document)
        embeddings, _ = generate_embeddings(doc, self.embedding_model)
        k_opt = find_optimal_clusters(embeddings, min_clusters=min_clusters, max_clusters=max_clusters)

        kmeans, clusters = cluster_sentences(embeddings, k_opt)
        # get centroids
        centroids = kmeans.cluster_centers_

        # clf the centroids
        preds = self.clf.predict(centroids)
        useful_clusters = [i for i in range(len(preds)) if preds[i] == 1]

        cluster_phrases = {i: [] for i in range(k_opt)}

        for sentence, cluster_id in zip(list(doc.sents), clusters):
            if cluster_id in useful_clusters:
                cluster_phrases[cluster_id].append(sentence)

        # reordena las frases en base a la posición original en cada key
        for cluster_id in cluster_phrases.keys():
            cluster_phrases[cluster_id] = sorted(
                cluster_phrases[cluster_id], 
                key=lambda x: list(doc.sents).index(x)
            )

        # tener una lista con las frases pertenecientes a los clusters útiles
        results = [
            " ".join([str(phrase) for phrase in phrases])
            for phrases in cluster_phrases.values() if len(phrases) > 0
        ]

        return results


class DocumentClustererTopKSentences(DocumentClusterer):

    def __init__(
            self,
            embedding_model: SentenceTransformer,
            spacy_model: str,
            top_k_sents: int = 3,
        ):
        super().__init__(embedding_model, spacy_model)
        self.top_k_sents = top_k_sents
    
    def cluster_and_assign(self, document: str, min_clusters: int = 5, max_clusters: int = 100) -> List[str]:
        doc = self.nlp(document)
        embeddings, _ = generate_embeddings(doc, self.embedding_model)
        k_opt = find_optimal_clusters(embeddings, min_clusters=min_clusters, max_clusters=max_clusters)

        _, clusters = cluster_sentences(embeddings, k_opt)

        cluster_phrases = {i: [] for i in range(k_opt)}
        cluster_embeddings = {i: [] for i in range(k_opt)}

        for sentence, cluster_id, embedding in zip(list(doc.sents), clusters, embeddings):
            cluster_phrases[cluster_id].append(sentence)
            cluster_embeddings[cluster_id].append(embedding)

        significant_phrases = {}

        for cluster_id, cluster_embs in cluster_embeddings.items():
            if len(cluster_embs) < 2:
                significant_phrases[cluster_id] = [cluster_phrases[cluster_id][0]]
                continue

            sim_matrix = 1 - np.array([[cosine(e1, e2) for e2 in cluster_embs] for e1 in cluster_embs])

            G = nx.Graph()
            for i, phrase in enumerate(cluster_phrases[cluster_id]):
                G.add_node(i, phrase=str(phrase))

            for i in range(len(cluster_embs)):
                for j in range(i + 1, len(cluster_embs)):
                    G.add_edge(i, j, weight=sim_matrix[i, j])

            centrality = nx.degree_centrality(G)

            top_k_nodes = sorted(centrality, key=centrality.get, reverse=True)[:self.top_k_sents]

            significant_phrases[cluster_id] = [cluster_phrases[cluster_id][node] for node in top_k_nodes]

        all_significant_phrases = [
            phrase
            for _, phrases in significant_phrases.items()
            for phrase in phrases
        ]

        ordered_phrases = sorted(
            all_significant_phrases, key=lambda x: list(doc.sents).index(x)
        )

        return [" ".join([str(phrase) for phrase in ordered_phrases])]



#main
if __name__ == '__main__':

    # delete warnings
    import warnings
    import joblib

    warnings.filterwarnings("ignore")

    
    # Initialize the embedding model and text splitter
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model_spacy     = 'es_core_news_sm'
    top_k_sents     = 3

    # Create the DocumentClusterer instance
    #clusterer = DocumentClustererTopKSentences(embedding_model, model_spacy, top_k_sents=top_k_sents)
    # load the model
    file = "models/RandomForest_best_model.pkl"
    loaded_model = joblib.load(file)
    print(loaded_model)
    clusterer = DocumentClustererKMeans(embedding_model, model_spacy, loaded_model)
    # Example document and summary
    from datasets import load_from_disk

    dataset = load_from_disk("data/02-processed/spanish")
    document = dataset["test"]['input'][10]

    # Perform clustering and assignment
    result = clusterer.cluster_and_assign(document)

    #calcular la reducción
    print(f"Original: {len(document)} caracteres")
    for i, cluster in enumerate(result):
        print(f"Cluster {i}: {len(cluster)} caracteres")
        print(f"Reducción: {1 - len(cluster) / len(document):.2%}")

