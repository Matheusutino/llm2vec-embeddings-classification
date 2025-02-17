import numpy as np
from typing import List, Dict, Any
import llama_cpp
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from src.core.utils import get_value_by_key_json
from src.core.embeddings.base_embeddings import BaseEmbeddings

class LlamaCppEmbeddings(BaseEmbeddings):
    """
    A class to handle the generation and aggregation of token-level embeddings
    using LlamaCpp and manual pooling.

    Attributes:
        llm (llama_cpp.Llama): The Llama model instance.
        model_path (str): Path to the Llama model file.
    """

    def __init__(self, repo_id: str, filename: str, n_ctx: int = 512, verbose: bool = False):
        """
        Initializes the EmbeddingAggregator by downloading the specified model.

        Args:
            repo_id (str): The Hugging Face repository ID of the model.
            filename (str): The filename of the model to download.
        """
        # Download the model file
        n_ctx = get_value_by_key_json(file_path="configs/context_lenght.json", key = repo_id)
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.llm = llama_cpp.Llama(model_path = self.model_path, 
                                   n_ctx = n_ctx, 
                                   n_gpu_layers = -1, 
                                   pooling_type = llama_cpp.LLAMA_POOLING_TYPE_MEAN,
                                   embedding = True,
                                   verbose = verbose)

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates and aggregates embeddings for a list of input texts, applying average pooling immediately.

        Args:
            texts (List[str]): A list of strings for which to generate and aggregate embeddings.

        Returns:
            List[np.ndarray]: A list of aggregated sequence-level embeddings.
        """

        embeddings = []

        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = self.llm.embed(text)
            embeddings.append(embedding)

        return embeddings


