import numpy as np
from typing import List, Dict, Any
import llama_cpp
from huggingface_hub import hf_hub_download

class LlamaCppEmbeddings:
    """
    A class to handle the generation and aggregation of token-level embeddings
    using LlamaCpp and manual pooling.

    Attributes:
        llm (llama_cpp.Llama): The Llama model instance.
        model_path (str): Path to the Llama model file.
    """

    def __init__(self, repo_id: str, filename: str):
        """
        Initializes the EmbeddingAggregator by downloading the specified model.

        Args:
            repo_id (str): The Hugging Face repository ID of the model.
            filename (str): The filename of the model to download.
        """
        # Download the model file
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.llm = llama_cpp.Llama(model_path=self.model_path, n_gpu_layers = -1, verbose = True, embedding=True)

    def create_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a list of input texts.

        Args:
            texts (List[str]): A list of strings for which to generate embeddings.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing embeddings for each text.
        """
        return self.llm.create_embedding(texts)['data']

    def average_pooling(self, token_embeddings: List[List[float]]) -> np.ndarray:
        """
        Computes the average of token-level embeddings to produce a sequence-level embedding.

        Args:
            token_embeddings (List[List[float]]): A list of embeddings for each token.

        Returns:
            np.ndarray: The averaged embedding for the entire sequence.
        """
        token_embeddings_array = np.array(token_embeddings)
        return np.mean(token_embeddings_array, axis=0)

    def aggregate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates and aggregates embeddings for a list of input texts.

        Args:
            texts (List[str]): A list of strings for which to generate and aggregate embeddings.

        Returns:
            List[np.ndarray]: A list of aggregated sequence-level embeddings.
        """
        embeddings = self.create_embeddings(texts)
        sequence_embeddings = [
            self.average_pooling(embedding_data['embedding']) 
            for embedding_data in embeddings
        ]
        return sequence_embeddings
