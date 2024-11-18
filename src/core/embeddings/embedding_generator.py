from src.core.embeddings.bert import BertEmbeddings
from src.core.embeddings.llama_cpp import LlamaCppEmbeddings
from src.core.embeddings.ollama import OllamaEmbeddings
from src.core.embeddings.llm2vec import LLM2VecEmbeddings
from src.core.prompt_generator import PromptGenerator

class EmbeddingGenerator:
    """
    A class to generate embeddings for different types of models, including BERT, LlamaCpp, and LLM2Vec.
    """
    
    def __init__(self, embedding_type, **kwargs):
        """
        Args:
        embedding_type (str): The type of embedding model to use. Supported values are 'bert', 'llama_cpp', and 'llm2vec'.
        **kwargs: Additional keyword arguments specific to the embedding model chosen.
            - For BERT:
                - model_name (str): The name of the BERT model to use.
            - For LlamaCpp:
                - repo_id (str): The Hugging Face repository ID for the LlamaCpp model.
                - filename (str): The name of the file containing the model.
            - For LLM2Vec:
                - model_base_name (str): The base name of the LLM2Vec model.
                - model_name_version (str): The version of the LLM2Vec model.
                - user_prompt (str): user_prompt or prompt to use for generating LLM2Vec embeddings.
        """
        self.embedding_type = embedding_type
        self.kwargs = kwargs
        
    def generate(self, dataset):
        """
        Generates embeddings based on the specified embedding type.

        Args:
            dataset (pd.DataFrame): The dataset containing text data for embedding generation.

        Returns:
            np.ndarray: A list of generated embeddings.
        
        Raises:
            ValueError: If the embedding type is not supported.
        """
        if self.embedding_type == "bert":
            return self._generate_bert_embeddings(dataset)
        elif self.embedding_type == "llama_cpp":
            return self._generate_llama_cpp_embeddings(dataset)
        elif self.embedding_type == "ollama":
            return self._generate_ollama_embeddings(dataset)
        elif self.embedding_type == "llm2vec":
            return self._generate_llm2vec_embeddings(dataset)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")

    def _generate_bert_embeddings(self, dataset):
        """
        Generates embeddings using the BERT model.

        Args:
            dataset (pd.DataFrame): The dataset containing text data for embedding generation.

        Returns:
            np.ndarray: A list of BERT embeddings.
        """
        model_name = self.kwargs.get("model_name")
        bert_embeddings = BertEmbeddings(model_name=model_name)
        return bert_embeddings.get_embeddings(dataset["text"].tolist())

    def _generate_llama_cpp_embeddings(self, dataset):
        """
        Generates embeddings using the LlamaCpp model.

        Args:
            dataset (pd.DataFrame): The dataset containing text data for embedding generation.

        Returns:
            np.ndarray: A list of LlamaCpp embeddings.
        """
        repo_id = self.kwargs.get("repo_id")
        filename = self.kwargs.get("filename")
        system_prompt = self.kwargs.get("system_prompt")
        user_prompt = self.kwargs.get("user_prompt")
        dataset["text"] = dataset["text"].apply(
            lambda text: PromptGenerator.generate_prompt_llama_cpp(
                repo_id=repo_id, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt.format(classes=dataset['class'].unique().tolist()), 
                text=text
            )
        )

        llama_cpp_embeddings = LlamaCppEmbeddings(repo_id=repo_id, filename=filename)
        return llama_cpp_embeddings.get_embeddings(dataset["text"].tolist())

    def _generate_ollama_embeddings(self, dataset):
        """
        Generates embeddings using the Ollama model.

        Args:
            dataset (pd.DataFrame): The dataset containing text data for embedding generation.

        Returns:
            np.ndarray: A list of Ollama embeddings.
        """
        model_name = self.kwargs.get("model_name")
        system_prompt = self.kwargs.get("system_prompt")
        user_prompt = self.kwargs.get("user_prompt")

        dataset["text"] = dataset["text"].apply(
            lambda text: PromptGenerator.generate_prompt_ollama(
                system_prompt=system_prompt, 
                user_prompt=user_prompt.format(classes=dataset['class'].unique().tolist()), 
                text=text
            )
        )

        llama_cpp_embeddings = OllamaEmbeddings(model_name=model_name)
        return llama_cpp_embeddings.get_embeddings(dataset["text"].tolist())

    def _generate_llm2vec_embeddings(self, dataset):
        """
        Generates embeddings using the LLM2Vec model.

        Args:
            dataset (pd.DataFrame): The dataset containing text data for embedding generation.

        Returns:
            np.ndarray: A list of LLM2Vec embeddings.
        """
        model_base_name = self.kwargs.get("model_base_name")
        model_name_version = self.kwargs.get("model_name_version")
        user_prompt = self.kwargs.get("user_prompt").format(classes=dataset['class'].unique().tolist())
        llm2vec_embeddings = LLM2VecEmbeddings(model_base_name=model_base_name, model_name_version=model_name_version)
        return llm2vec_embeddings.get_embeddings(user_prompt, dataset["text"].tolist())
