import argparse
from tqdm import tqdm
from src.core.utils import get_all_npy_files_in_directory
from src.scripts.run_classifier import run_classifier

def run_all_classifier(model_classifier: str, cv: int):
    embeddings_path = get_all_npy_files_in_directory("results")

    for embedding_path in tqdm(embeddings_path, desc="Processing datasets"):
        try:
            tqdm.write(f"Processing: {embedding_path}")
            run_classifier(embedding_path, model_classifier, cv)
        except Exception as e:
            tqdm.write(f"Error processing {embedding_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification using precomputed embeddings.")

    parser.add_argument("--model_classifier", type=str, default="knn", help="Model classifier name.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")

    args = parser.parse_args()

    # Chamada da função ajustada
    run_all_classifier(args.model_classifier, args.cv)
