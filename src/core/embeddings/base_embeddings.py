from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddings(ABC):
    """
    Abstract base class for embeddings models. 
    Any embedding model class should inherit from this class and implement the abstract methods.
    """
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]):
        """
        Generate sentence embeddings for a list of texts.
        
        Parameters:
        - texts: A list of texts to be converted into embeddings.

        Returns:
        - ndarray: Sentence embeddings.
        """
        pass