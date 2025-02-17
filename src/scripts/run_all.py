import argparse
from src.core.utils import read_json, get_all_files_in_directory
from src.scripts.run_classificator import run_classificator

def run_all_models(models_path: str, prompt_path : str, datasets_path: str, model_classifier:str, cv: int, system_prompt: str, user_prompt: str):
    models = read_json(models_path)
    prompts = read_json(prompt_path)
    datasets = get_all_files_in_directory(datasets_path)
    
    for embedding_type in sorted(models.keys(), reverse=True):
        model_list = models[embedding_type]
        for model in model_list:
            for dataset in datasets:
                for prompt_name, prompt in prompts.items():
                    # print(f"Run model {embedding_type} - {model}")
                    try:
                        if embedding_type == 'bert':
                            model_name = model['model_name']
                            run_classificator(dataset, embedding_type, model_classifier, cv=cv, model_name=model_name)
                        elif embedding_type == 'ollama':
                            model_name = model['model_name']
                            system_prompt = prompt['system_prompt']
                            user_prompt = prompt['user_prompt']
                            run_classificator(dataset, embedding_type, model_classifier, cv=cv, model_name=model_name, system_prompt=system_prompt, user_prompt=user_prompt, prompt_name=prompt_name)
                        elif embedding_type == 'llm2vec':
                            model_base_name = model['model_base_name']
                            model_name_version = model['model_name_version']
                            user_prompt = prompt['user_prompt']
                            run_classificator(dataset, embedding_type, model_classifier, cv=cv, model_base_name=model_base_name, model_name_version=model_name_version, user_prompt=user_prompt, prompt_name=prompt_name)
                    except Exception as e:
                        print(f"Error to process embedding type: {embedding_type}, model: {model}, dataset: {dataset}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")
    
    parser.add_argument("--models_path", type=str, default= "configs/models_infos.json", help="Path to the JSON file containing model configurations.")
    parser.add_argument("--prompt_path", type=str, default= "configs/prompts.json", help="Path to the JSON file containing model configurations.")
    parser.add_argument("--datasets_path", type=str, default= "datasets", help="Path to the dataset CSV file.")
    parser.add_argument("--model_classifier", type=str,  default="knn", help="Model classifier name.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--system_prompt", type=str, default="Provide a system-level overview of the process.", help="System prompt for LLM2Vec embeddings.", required=False)
    parser.add_argument("--user_prompt", type=str, default="Summarize and capture the main points of the text.", help="User prompt for LLM2Vec embeddings.", required=False)
    
    args = parser.parse_args()

    # Chamada do run_all_models com os parâmetros fornecidos
    run_all_models(args.models_path,
                   args.prompt_path, 
                   args.datasets_path, 
                   args.model_classifier, 
                   cv=args.cv, 
                   system_prompt=args.system_prompt, 
                   user_prompt=args.user_prompt)
