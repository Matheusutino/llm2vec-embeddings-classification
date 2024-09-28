import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from core.embeddings.llm2vec import LLM2VecEmbeddings
from src.core.utils import create_directory, get_last_element_from_path

def main(dataset_path: str, 
         model_base_name: str, 
         model_name_version: str, 
         instruction: str,
         cv: int):
    
    dataset_name = get_last_element_from_path(dataset_path)
    result_path = f"results/{dataset_name}/{model_name_version}"
    create_directory(result_path)

    dataset = pd.read_csv(dataset_path)

    llm2vec_embeddings = LLM2VecEmbeddings(model_base_name = model_base_name,
                                    model_name_version = model_name_version)
    
    embeddings = llm2vec_embeddings.get_embeddings(instruction, dataset["text"].tolist())

    dataset["embeddings"] = embeddings

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro")
    }

    model = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs = -1)

    X = np.array(dataset["embeddings"].tolist())
    y = dataset["class"].tolist()

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs = -1)

    return cv_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--model_base_name", type=str, help="Base name of the model.")
    parser.add_argument("--model_name_version", type=str, help="Model name with version.")
    parser.add_argument("--instruction", type=str, help="Instruction for the model.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    
    args = parser.parse_args()
    
    cv_results = main(args.dataset_path, args.model_base_name, args.model_name_version, args.instruction, args.cv)
    
    print(cv_results)