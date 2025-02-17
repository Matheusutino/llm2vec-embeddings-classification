import argparse
from src.core.utils import read_json, get_all_files_in_directory
from src.scripts.run_embedding import run_embedding

def run_all_embeddings(models_path: str, prompt_path : str, datasets_path: str):
    models = read_json(models_path)
    prompts = read_json(prompt_path)
    datasets = get_all_files_in_directory(datasets_path)
    
    for embedding_type in sorted(models.keys(), reverse=True):
        model_list = models[embedding_type]
        for model in model_list:
            for dataset in datasets:
                for prompt_name in prompts.keys():
                    #print(f"Run model {embedding_type} - {model} - {prompt_name}")
                    try:
                        if embedding_type == 'bert':
                            model_name = model['model_name']
                            run_embedding(dataset, embedding_type, model_name=model_name)
                        elif embedding_type == 'ollama':
                            model_name = model['model_name']
                            run_embedding(dataset, embedding_type, model_name=model_name, prompt_name=prompt_name)
                        elif embedding_type == 'llm2vec':
                            model_base_name = model['model_base_name']
                            model_name_version = model['model_name_version']
                            run_embedding(dataset, embedding_type, model_base_name=model_base_name, model_name_version=model_name_version, prompt_name=prompt_name)
                    except Exception as e:
                        print(f"Error to process embedding type: {embedding_type}, model: {model}, dataset: {dataset}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--models_path", type=str, default= "configs/models_infos.json", help="Path to the JSON file containing model configurations.")
    parser.add_argument("--prompt_path", type=str, default= "configs/prompts.json", help="Path to the JSON file containing model configurations.")
    parser.add_argument("--datasets_path", type=str, default= "datasets", help="Path to the dataset CSV file.")
    
    args = parser.parse_args()

    # Chamada do run_all_models com os parâmetros fornecidos
    run_all_embeddings(args.models_path,
                    args.prompt_path, 
                    args.datasets_path)
