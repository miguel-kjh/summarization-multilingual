from sklearn.cluster import Birch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict

class DocumentClusterer:
    def __init__(self, embedding_model: SentenceTransformer, text_splitter: RecursiveCharacterTextSplitter, num_clusters: int):
        """
        Initialize the class with the embedding model, text splitter, and number of clusters.
        
        Args:
            embedding_model (SentenceTransformer): Pretrained embedding model.
            text_splitter (RecursiveCharacterTextSplitter): Text splitter to divide documents and summaries.
            num_clusters (int): Number of clusters for the clustering algorithm.
        """
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
        self.num_clusters = num_clusters

        tokenizer = self.embedding_model.tokenizer

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.embedding_model.resize_token_embeddings(len(tokenizer))

    def cluster_and_assign(self, document: str, summary: str) -> List[Tuple[List[str], List[str]]]:
        """
        Processes the document and summary to cluster document parts and assign summary parts to clusters.
        
        Args:
            document (str): The full text of the document to be clustered.
            summary (str): The summary text to be assigned to clusters.
        
        Returns:
            List[Tuple[List[str], List[str]]]: A list of tuples where:
                - The first element is a list of document parts in the cluster.
                - The second element is a list of summary parts assigned to the cluster.
        """
        # Split the document into parts
        document_parts: List[str] = [doc.page_content for doc in self.text_splitter.create_documents([document])]
        print(len(document_parts))
        
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
        
        return result

#main
if __name__ == '__main__':


    
    # Initialize the embedding model and text splitter
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False)
    num_clusters = 10

    # Create the DocumentClusterer instance
    clusterer = DocumentClusterer(embedding_model, text_splitter, num_clusters)
    # Example document and summary
    from datasets import load_from_disk

    dataset = load_from_disk("data/02-processed/spanish")
    print(dataset)
    exit()
    document = dataset["test"]['input'][0]
    print(len(document))
    summary = dataset["test"]['output'][0]
    print(len(summary))

    # Perform clustering and assignment
    result = clusterer.cluster_and_assign(document, summary)

    # Display the results
    for cluster_id, (doc_parts, sum_parts) in enumerate(result):
        print("#" * 50)
        print(f"Cluster {cluster_id}:\n")
        print("Document Parts:")
        print(f"- {sum([len(part) for part in doc_parts])}\n")
        print("Summary Parts:")
        print(f"- {sum([len(part) for part in sum_parts])}\n")

