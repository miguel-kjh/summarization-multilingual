import numpy as np
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine
import spacy
import networkx as nx
from kneed import KneeLocator

from utils import SEED

class DocumentClusterer:
    def __init__(
            self, 
            embedding_model: SentenceTransformer, 
            spacy_model: str,
            top_k_sents: int = 3
        ):
        """
        Initialize the class with the embedding model, text splitter, and number of clusters.
        
        Args:
            embedding_model (SentenceTransformer): Pretrained embedding model.
            spacy_model (str): Spacy model to use for tokenization sents.
            range_clusters (List[int]): List of number of clusters to evaluate.
        """
        self.embedding_model = embedding_model
        self.nlp = spacy.load(spacy_model)
        self.top_k_sents = top_k_sents

        tokenizer = self.embedding_model.tokenizer

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.embedding_model.resize_token_embeddings(len(tokenizer))

    """def cluster_and_assign(self, document: str, summary: str) -> str:
        
        Processes the document and summary to cluster document parts and assign summary parts to clusters.
        
        Args:
            document (str): The full text of the document to be clustered.
            summary (str): The summary text to be assigned to clusters.
        
        Returns:
            List[Tuple[List[str], List[str]]]: A list of tuples where:
                - The first element is a list of document parts in the cluster.
                - The second element is a list of summary parts assigned to the cluster.
        # Split the document into parts
        document_parts: List[str] = [doc.page_content for doc in self.text_splitter.create_documents([document])]
        
        # Generate embeddings for document parts
        document_embeddings = self.embedding_model.encode(document_parts)

        # Perform clustering using Birch
        clusterer = Birch(n_clusters=self.num_clusters)
        document_clusters: List[int] = clusterer.fit_predict(document_embeddings)

        # Split the summary into parts
        summary_parts: List[str] = [sum.page_content for sum in self.text_splitter.create_documents([summary])]

        # Generate embeddings for summary parts
        summary_embeddings = self.embedding_model.encode(summary_parts)

        # Assign each summary part to the closest document cluster
        cluster_to_summaries: Dict[int, List[str]] = {}
        for i, summary_embedding in enumerate(summary_embeddings):
            similarities = cosine_similarity(summary_embedding.reshape(1, -1), document_embeddings)
            closest_cluster = document_clusters[similarities.argmax()]
            if closest_cluster not in cluster_to_summaries:
                cluster_to_summaries[closest_cluster] = []
            cluster_to_summaries[closest_cluster].append(summary_parts[i])

        # Build the result: clusters with both document and summary parts
        result: List[Tuple[List[str], List[str]]] = []
        for cluster_id in set(document_clusters):
            if cluster_id in cluster_to_summaries:
                cluster_documents = [document_parts[i] for i, cluster in enumerate(document_clusters) if cluster == cluster_id]
                cluster_summaries = cluster_to_summaries[cluster_id]
                result.append((cluster_documents, cluster_summaries))
        
        return result"""
    
    def cluster_and_assign(self, document: str, min_clusters: int = 5, max_clusters: int = 100) -> str:
        doc = self.nlp(document)
        sentences = list(sent.text for sent in doc.sents)
        embeddings = self.embedding_model.encode(sentences)
        inertia_values = []
        silhouette_scores = []
        #print(min_clusters, max_clusters)

        cluster_range = range(min_clusters, max_clusters)

        for n_clusters in cluster_range:
            if n_clusters == 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(embeddings)
                inertia_values.append(kmeans.inertia_)
                silhouette_scores.append(None)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(embeddings)
                inertia_values.append(kmeans.inertia_)
                silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)

        knee_locator = KneeLocator(
            cluster_range, 
            inertia_values, 
            curve="convex", 
            direction="decreasing"
        )
        k_opt = knee_locator.knee
        #print(f"Optimal number of clusters: {k_opt}")

        kmeans = KMeans(n_clusters=k_opt, random_state=SEED)
        clusters = kmeans.fit_predict(embeddings)

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

        return " ".join([str(phrase) for phrase in ordered_phrases])



#main
if __name__ == '__main__':

    # delete warnings
    import warnings
    warnings.filterwarnings("ignore")

    
    # Initialize the embedding model and text splitter
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model_spacy     = 'es_core_news_sm'
    top_k_sents     = 1

    # Create the DocumentClusterer instance
    clusterer = DocumentClusterer(embedding_model, model_spacy, top_k_sents=top_k_sents)
    # Example document and summary
    from datasets import load_from_disk

    dataset = load_from_disk("data/02-processed/spanish")
    document = dataset["test"]['input'][2]

    # Perform clustering and assignment
    result = clusterer.cluster_and_assign(document)

    #calcular la reducción
    print(f"Original: {len(document)} caracteres")
    print(f"Compacto: {len(result)} caracteres")
    print(f"Reducción: {1 - len(result) / len(document):.2%}")

