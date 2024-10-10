import argparse
import time
import pandas as pd
from sys import getsizeof
from sklearn.model_selection import cross_validate
from src.core.embeddings.embedding_generator import EmbeddingGenerator
from src.core.model_tuning import ModelTuning
from src.core.utils import create_directory, get_last_element_from_path, save_json, replace_character


def run_classificator(dataset_path: str, embedding_type: str, model_classifier: str, cv: int, **kwargs):
    dataset_name = get_last_element_from_path(dataset_path)
    dataset = pd.read_csv(dataset_path)

    embedding_generator = EmbeddingGenerator(embedding_type, **kwargs)
    
    start_time = time.time()
    embeddings = embedding_generator.generate(dataset)
    end_time = time.time()
    embedding_generation_time = end_time - start_time

    X = embeddings
    y = dataset["class"].tolist()

    model_tuning = ModelTuning(model_name=model_classifier)
    results = model_tuning.tune_hyperparameters(X, y, cv=cv)

    result_path = f"results/{dataset_name}/{embedding_type}/{replace_character(kwargs.get("model_name") or kwargs.get("repo_id") or kwargs.get("model_name_version"))}/{model_classifier}"

    create_directory(result_path)
    bayesian_search_cv = pd.DataFrame(results.cv_results_)
    bayesian_search_cv.to_csv(f"{result_path}/bayes_search_cv.csv", index=False)

    best_f1_row = bayesian_search_cv.loc[bayesian_search_cv["mean_test_f1_score"].idxmax()]

    best_f1_row["embedding_generation_time"] = embedding_generation_time
    best_f1_row["embedding_generation_size"] = getsizeof(X)
    best_f1_row = best_f1_row.to_dict()
    save_json(best_f1_row, f"{result_path}/results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--embedding_type", type=str, choices=["bert", "llama_cpp", "llm2vec"], 
                        help="Type of embedding to use: 'bert', 'llama_cpp', or 'llm2vec'.")
    parser.add_argument("--model_name", type=str, help="Model name (for Bert).", required=False)
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID (for LlamaCpp).", required=False)
    parser.add_argument("--filename", type=str, help="Filename of the model (for LlamaCpp).", required=False)
    parser.add_argument("--model_base_name", type=str, help="Model Base Name for the model (for Llm2Vec).", required=False)
    parser.add_argument("--model_name_version", type=str, help="Model Name Version of the model (for Llm2Vec).", required=False)
    parser.add_argument("--instruction", type=str, default = "Summarize and capture the main points of the text.", help="Instruction for LLM2Vec embeddings.", required=False)
    parser.add_argument("--model_classifier", type=str,  default="knn", help="Model classifier name.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    
    args = parser.parse_args()

    # Chamada do run_classificator com parâmetros explícitos
    run_classificator(args.dataset_path, 
                      args.embedding_type, 
                      args. model_classifier,
                      args.cv, 
                      model_name=args.model_name, 
                      repo_id=args.repo_id, 
                      filename=args.filename, 
                      model_base_name=args.model_base_name, 
                      model_name_version=args.model_name_version, 
                      instruction=args.instruction)