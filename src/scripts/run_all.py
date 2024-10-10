import argparse
from src.core.utils import read_json, get_all_files_in_directory
from src.scripts.run_classificator import run_classificator

def run_all_models(models_path: str, datasets_path: str, model_classifier:str, cv: int, instruction: str):
    models = read_json(models_path)
    datasets = get_all_files_in_directory(datasets_path)

    for dataset in datasets:
        for embedding_type, model_list in models.items():
            for model in model_list:
                try:
                    if embedding_type == 'bert':
                        model_name = model['model_name']
                        run_classificator(dataset, embedding_type, model_classifier, cv=cv, model_name=model_name)
                    elif embedding_type == 'llama_cpp':
                        repo_id = model['repo_id']
                        filename = model['filename']
                        run_classificator(dataset, embedding_type, model_classifier, cv=cv, repo_id=repo_id, filename=filename)
                    elif embedding_type == 'llm2vec':
                        model_base_name = model['model_base_name']
                        model_name_version = model['model_name_version']
                        run_classificator(dataset, embedding_type, model_classifier, cv=cv, model_base_name=model_base_name, model_name_version=model_name_version, instruction=instruction)
                except Exception as e:
                    print(f"Error : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--models_path", type=str, default= "configs/models_infos.json", help="Path to the JSON file containing model configurations.")
    parser.add_argument("--datasets_path", type=str, default= "datasets", help="Path to the dataset CSV file.")
    parser.add_argument("--model_classifier", type=str,  default="knn", help="Model classifier name.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--instruction", type=str, default = "Summarize and capture the main points of the text.", help="Instruction for LLM2Vec embeddings.", required=False)
    
    args = parser.parse_args()

    # Chamada do run_all_models com os parâmetros fornecidos
    run_all_models(args.models_path, args.datasets_path, args.model_classifier, cv=args.cv, instruction=args.instruction)
