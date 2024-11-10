from sentence_transformers import SentenceTransformer
from typing import List
import os
from src.core.embeddings.base_embeddings import BaseEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BertEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the model for sentence embeddings using SentenceTransformer.

        Parameters:
        - model_name: The name of the pretrained model.
        """
        self.model = SentenceTransformer(model_name, device = device)

    def get_embeddings(self, texts: List[str], show_progress_bar = True):
        """
        Generate sentence embeddings for a list of texts.

        Parameters:
        - texts: A list of texts to be converted into embeddings.

        Returns:
        - ndarray: Sentence embeddings.
        """
        # Generate and return embeddings
        return self.model.encode(texts, show_progress_bar = show_progress_bar)