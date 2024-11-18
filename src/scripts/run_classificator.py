import argparse
import time
import pandas as pd
from sys import getsizeof
from src.core.embeddings.embedding_generator import EmbeddingGenerator
from src.core.model_tuning import ModelTuning
from src.core.utils import create_directory, get_last_element_from_path, replace_character, save_results


def run_classificator(dataset_path: str, embedding_type: str, model_classifier: str, cv: int, **kwargs):
    dataset_name = get_last_element_from_path(dataset_path)
    
    result_path = f"results/{dataset_name}/{embedding_type}/{replace_character(kwargs.get('model_name') or kwargs.get('repo_id') or kwargs.get('model_name_version'))}{f'/{kwargs.get('prompt_name')}' if embedding_type != 'bert' else ''}/{model_classifier}"
    create_directory(result_path)

    dataset = pd.read_csv(dataset_path)

    embedding_generator = EmbeddingGenerator(embedding_type, **kwargs)
    
    start_time = time.time()
    embeddings = embedding_generator.generate(dataset)
    end_time = time.time()

    X = embeddings
    y = dataset["class"].tolist()

    model_tuning = ModelTuning(model_name=model_classifier)
    results = model_tuning.tune_hyperparameters(X, y, cv=cv)

    embedding_generation_size = getsizeof(X)
    embedding_generation_time = end_time - start_time

    save_results(result_path, results, embedding_type, embedding_generation_time = embedding_generation_time, embedding_generation_size = embedding_generation_size, system_prompt = kwargs.get("system_prompt"), user_prompt = kwargs.get("user_prompt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--embedding_type", type=str, choices=["bert", "llama_cpp", "ollama", "llm2vec"], 
                        help="Type of embedding to use: 'bert', 'llama_cpp', or 'llm2vec'.")
    parser.add_argument("--model_name", type=str, help="Model name (for Bert and Ollama).", required=False)
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID (for LlamaCpp).", required=False)
    parser.add_argument("--filename", type=str, help="Filename of the model (for LlamaCpp).", required=False)
    parser.add_argument("--model_base_name", type=str, help="Model Base Name for the model (for Llm2Vec).", required=False)
    parser.add_argument("--model_name_version", type=str, help="Model Name Version of the model (for Llm2Vec).", required=False)
    parser.add_argument("--system_prompt", type=str, default="You are an intelligent assistant that follows the user’s instructions closely. Respond to the user's queries in a clear, concise, and helpful manner. Ensure that all answers are accurate, relevant, and tailored to the user's needs. Stay focused on providing valuable information and completing tasks as requested by the user.", help="System prompt for LLM2Vec embeddings.", required=False)
    parser.add_argument("--user_prompt", type=str, default="", help="User prompt for LLM2Vec embeddings.", required=False)
    parser.add_argument("--prompt_name", type=str, default="base_prompt", help="Prompt indentifier", required=False)
    parser.add_argument("--model_classifier", type=str,  default="knn", help="Model classifier name.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    
    args = parser.parse_args()

    # Chamada do run_classificator com parâmetros explícitos
    run_classificator(args.dataset_path, 
                      args.embedding_type, 
                      args.model_classifier,
                      args.cv, 
                      model_name=args.model_name, 
                      repo_id=args.repo_id, 
                      filename=args.filename, 
                      model_base_name=args.model_base_name, 
                      model_name_version=args.model_name_version, 
                      system_prompt=args.system_prompt, 
                      user_prompt=args.user_prompt,
                      prompt_name=args.prompt_name)