import argparse
from src.core.utils import read_json, get_all_files_in_directory
from src.scripts.run_classificator import run_classificator

def run_all_models(models_path: str, datasets_path: str, cv: int):
    # Carregar os modelos do arquivo JSON
    models = read_json(models_path)
    datasets = get_all_files_in_directory(datasets_path)

    print(datasets)

    # Iterar sobre os modelos de cada tipo e chamar a função run_classificator
    for dataset in datasets:
        for embedding_type, model_list in models.items():
            for model in model_list:
                print(f"Running model for embedding type: {embedding_type}")
                if embedding_type == 'bert':
                    model_name = model['model_name']
                    run_classificator(dataset, embedding_type, cv=cv, model_name=model_name)
                elif embedding_type == 'llama_cpp':
                    repo_id = model['repo_id']
                    filename = model['filename']
                    run_classificator(dataset, embedding_type, cv=cv, repo_id=repo_id, filename=filename)
                elif embedding_type == 'llm2vec':
                    model_base_name = model['model_base_name']
                    model_name_version = model['model_name_version']
                    run_classificator(dataset, embedding_type, cv=cv, model_base_name=model_base_name, model_name_version=model_name_version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--models_path", type=str, default= "configs/models_infos.json", help="Path to the JSON file containing model configurations.")
    parser.add_argument("--datasets_path", type=str, default= "datasets", help="Path to the dataset CSV file.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    
    args = parser.parse_args()

    # Chamada do run_all_models com os parâmetros fornecidos
    run_all_models(args.models_path, args.datasets_path, cv=args.cv)
