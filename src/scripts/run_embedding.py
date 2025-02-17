import argparse
import time
import numpy as np
import pandas as pd
from src.core.embeddings.embedding_generator import EmbeddingGenerator
from src.core.utils import check_file_not_exists, create_directory, get_last_element_from_path, replace_character, save_json

def run_embedding(dataset_path: str, embedding_type: str, **kwargs):
    dataset_name = get_last_element_from_path(dataset_path)
    result_path = f"results/{dataset_name}/{embedding_type}/{replace_character(kwargs.get('model_name') or kwargs.get('repo_id') or kwargs.get('model_name_version'))}{f'/{kwargs.get("prompt_name")}' if embedding_type != "bert" else ""}"
    embeddings_path = f"{result_path}/embeddings.npy"
    time_path = f"{result_path}/embedding_time.json"
    
    check_file_not_exists(embeddings_path)
    
    dataset = pd.read_csv(dataset_path)
    embedding_generator = EmbeddingGenerator(embedding_type, **kwargs)
    
    # Medição do tempo de geração de embeddings
    start_time = time.time()
    embeddings = embedding_generator.generate(dataset)
    end_time = time.time()
    
    # Salvar embeddings e tempo de geração
    embedding_generation_time = end_time - start_time
    
    create_directory(result_path)
    np.save(embeddings_path, embeddings)
    
    # Salvar o tempo de geração em um arquivo JSON
    save_json({"embedding_generation_time": embedding_generation_time}, time_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save embeddings with timing.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--embedding_type", type=str, choices=["bert", "llama_cpp", "ollama", "llm2vec"], 
                        help="Type of embedding to use: 'bert', 'llama_cpp', or 'llm2vec'.")
    parser.add_argument("--model_name", type=str, help="Model name (for Bert and Ollama).", required=False)
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID (for LlamaCpp).", required=False)
    parser.add_argument("--filename", type=str, help="Filename of the model (for LlamaCpp).", required=False)
    parser.add_argument("--model_base_name", type=str, help="Model Base Name for the model (for Llm2Vec).", required=False)
    parser.add_argument("--model_name_version", type=str, help="Model Name Version of the model (for Llm2Vec).", required=False)
    parser.add_argument("--prompt_name", type=str, default="base_prompt", help="Prompt identifier", required=False)
    
    args = parser.parse_args()
    
    run_embedding(args.dataset_path, 
                    args.embedding_type, 
                    model_name=args.model_name, 
                    repo_id=args.repo_id, 
                    filename=args.filename, 
                    model_base_name=args.model_base_name, 
                    model_name_version=args.model_name_version, 
                    prompt_name=args.prompt_name)
