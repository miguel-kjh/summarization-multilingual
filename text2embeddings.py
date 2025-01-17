from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

class Text2Embeddings:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def transform(self, texts: List[str]) -> np.ndarray:
        pass

class Text2EmbeddingsSetenceTransforms(Text2Embeddings):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = SentenceTransformer(self.model_name)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
    
class Text2EmbeddingsOpenAI(Text2Embeddings):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = OpenAIEmbeddings(model=self.model_name)

    def transform(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.embed_documents(texts))
    
if __name__ == "__main__":
    texts = ["hola que hace?", "Que tal? que haces con haciendo?"]
    # SentenceTransformers
    text2embeddings = Text2EmbeddingsSetenceTransforms("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = text2embeddings.transform(texts)
    print(embeddings.shape)
    # OpenAI
    text2embeddings = Text2EmbeddingsOpenAI("text-embedding-3-large")
    embeddings = text2embeddings.transform(texts)
    print(embeddings.shape)