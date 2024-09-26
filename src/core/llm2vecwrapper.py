import os
import torch
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from typing import List
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class LLM2VecWrapper:
    def __init__(self, 
                 model_base_name: str, 
                 model_name_version: str, 
                 device: str = "cuda", 
                #  dtype = torch.bfloat16, 
                 dtype = torch.float16, 
                 pooling_mode: str = "mean", 
                 max_length: int = 512):
        """
        Initialize the model and tokenizer for LLM2Vec.
        """
        self.device = device
        self.model_base_name = model_base_name
        self.model_name_version = model_name_version
        self.dtype = dtype
        self.pooling_mode = pooling_mode
        self.max_length = max_length

        # Hugginface login
        login(os.environ.get("HUGGING_FACE_TOKEN"))

        # Load tokenizer, config, and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_base_name)
        self.config = AutoConfig.from_pretrained(model_base_name, trust_remote_code=True)
        self.l2v = self._initialize_llm2vec()
        

    def _load_model(self):
        """
        Load the base model and apply PeftModel for fine-tuning.
        """
        model = AutoModel.from_pretrained(
            self.model_base_name,
            config=self.config,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, self.model_base_name)

        model = model.merge_and_unload()  # This can take several minutes on cpu

        model = PeftModel.from_pretrained(
            model, self.model_name_version
        )
        return model

    def _initialize_llm2vec(self):
        """
        Initialize the LLM2Vec wrapper using the loaded model and tokenizer.
        """
        self.model = self._load_model()
        
        return LLM2Vec(self.model, self.tokenizer, pooling_mode=self.pooling_mode, max_length=self.max_length)

    def get_embeddings(self, 
                       instruction: str, 
                       texts: List[str]):
        """
        Generate embeddings for a list of texts based on the given instruction.

        Parameters:
        texts (list): A list of text strings to process.
        instruction (str): Instruction to guide the model during embedding.

        Returns:
        list: The embeddings for each text.
        """
        # Pair each text with the instruction
        queries = [[instruction, text] for text in texts]

        # Encode the text pairs into embeddings
        embeddings = self.l2v.encode(queries)
        return list(embeddings.numpy())
