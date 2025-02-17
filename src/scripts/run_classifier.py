import os
import argparse
import numpy as np
import pandas as pd
from src.core.model_tuning import ModelTuning
from src.core.utils import check_directory_exists, save_results, create_directory

def run_classifier(embeddings_path: str, model_classifier: str, cv: int):
    # Deriva dataset_path e result_path a partir do embeddings_path
    path_parts = embeddings_path.split("/")
    dataset_name = path_parts[1]  # Nome do dataset (primeiro elemento após a "/")
    dataset_path = f"datasets/{dataset_name}"  # Caminho do dataset 
    result_path = os.path.join(*path_parts[:-1], model_classifier)

    # Verifica se o diretório de resultados existe
    check_directory_exists(result_path)

    # Carrega o dataset e as embeddings
    dataset = pd.read_csv(dataset_path)
    embeddings = np.load(embeddings_path)
    
    X = embeddings
    y = dataset["class"].tolist()

    # Inicializa o ajuste do modelo e executa a validação cruzada
    model_tuning = ModelTuning(model_name=model_classifier)
    results = model_tuning.tune_hyperparameters(X, y, cv=cv)
    
    # Salva os resultados
    create_directory(result_path)
    save_results(result_path, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification using precomputed embeddings.")
    
    parser.add_argument("--embeddings_path", type=str, help="Path to the precomputed embeddings (npy file).")
    parser.add_argument("--model_classifier", type=str, default="knn", help="Model classifier name.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")

    args = parser.parse_args()

    # Chamada da função ajustada
    run_classifier(args.embeddings_path, 
                   args.model_classifier,
                   args.cv)
