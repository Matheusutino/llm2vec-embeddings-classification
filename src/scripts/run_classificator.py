import argparse
import time
import pandas as pd
import numpy as np
from sys import getsizeof
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from src.core.embeddings.embedding_generator import EmbeddingGenerator
from src.core.utils import create_directory, get_last_element_from_path, save_json, replace_character


def run_classificator(dataset_path: str, embedding_type: str, cv: int = 5, **kwargs):
    dataset_name = get_last_element_from_path(dataset_path)
    dataset = pd.read_csv(dataset_path)

    embedding_generator = EmbeddingGenerator(embedding_type, **kwargs)
    
    start_time = time.time()
    
    embeddings = embedding_generator.generate(dataset)
    
    end_time = time.time()

    embedding_generation_time = end_time - start_time

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro")
    }

    model = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    X = embeddings
    y = dataset["class"].tolist()

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    cv_results["embedding_generation_time"] = embedding_generation_time
    cv_results["embedding_generation_size"] = getsizeof(X)

    result_path = f"results/{dataset_name}/{embedding_type}/{replace_character(kwargs.get('model_name') or kwargs.get('repo_id') or kwargs.get('model_name_version'))}"

    create_directory(result_path)
    save_json(cv_results, f'{result_path}/results.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--embedding_type", type=str, choices=['bert', 'llama_cpp', 'llm2vec'], 
                        help="Type of embedding to use: 'bert', 'llama_cpp', or 'llm2vec'.")
    parser.add_argument("--model_name", type=str, help="Model name (for Bert).", required=False)
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID (for LlamaCpp).", required=False)
    parser.add_argument("--filename", type=str, help="Filename of the model (for LlamaCpp).", required=False)
    parser.add_argument("--model_base_name", type=str, help="Model Base Name for the model (for Llm2Vec).", required=False)
    parser.add_argument("--model_name_version", type=str, help="Model Name Version of the model (for Llm2Vec).", required=False)
    parser.add_argument("--instruction", type=str, default = "Summarize and capture the main points of the text.", help="Instruction for LLM2Vec embeddings.", required=False)
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    
    args = parser.parse_args()

    # Chamada do run_classificator com parâmetros explícitos
    run_classificator(args.dataset_path, args.embedding_type, args.cv, model_name=args.model_name, repo_id=args.repo_id, filename=args.filename, model_base_name=args.model_base_name, model_name_version=args.model_name_version, instruction=args.instruction)