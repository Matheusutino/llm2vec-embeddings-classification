from sentence_transformers import SentenceTransformer
from typing import List

class BertEmbeddings:
    def __init__(self, model_name: str):
        """
        Initialize the model for sentence embeddings using SentenceTransformer.

        Parameters:
        - model_name: The name of the pretrained model.
        """
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, sentences: List[str]):
        """
        Generate sentence embeddings for a list of sentences.

        Parameters:
        - sentences: A list of sentences to be converted into embeddings.

        Returns:
        - ndarray: Sentence embeddings.
        """
        # Generate and return embeddings
        return self.model.encode(sentences)