import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from src.core.embeddings.llama_cpp import LlamaCppEmbeddings
from src.core.utils import create_directory, get_last_element_from_path, save_json, replace_character

def main(dataset_path: str, 
         repo_id: str, 
         filename: str,
         cv: int):
    
    dataset_name = get_last_element_from_path(dataset_path)
    result_path = f"results/{dataset_name}/llama_cpp/{replace_character(repo_id)}"
    create_directory(result_path)

    dataset = pd.read_csv(dataset_path)

    # Inicializa o modelo de embeddings com repo_id e filename
    llama_cpp_embeddings = LlamaCppEmbeddings(repo_id=repo_id, filename=filename)
    
    embeddings = llama_cpp_embeddings.aggregate_embeddings(dataset["text"].tolist())

    dataset["embeddings"] = embeddings

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro")
    }

    model = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)

    X = np.array(dataset["embeddings"].tolist())
    y = dataset["class"].tolist()

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score = True, n_jobs = -1)

    save_json(cv_results, f'{result_path}/results.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID of the model.")
    parser.add_argument("--filename", type=str, help="Filename of the model to download.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    
    args = parser.parse_args()
    
    main(args.dataset_path, args.repo_id, args.filename, args.cv)

