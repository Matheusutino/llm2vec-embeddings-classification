import os
import pandas as pd
import json

def load_results_to_dataframe(base_path):
    results = []
    
    # Percorre a estrutura de diretórios
    for dataset_name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_name)
        
        if os.path.isdir(dataset_path):
            for model_type in os.listdir(dataset_path):
                model_type_path = os.path.join(dataset_path, model_type)
                
                if os.path.isdir(model_type_path):
                    for model_name in os.listdir(model_type_path):
                        model_name_path = os.path.join(model_type_path, model_name)
                        json_file_path = os.path.join(model_name_path, 'results.json')
                        
                        # Verifica se o arquivo results.json existe
                        if os.path.isfile(json_file_path):
                            with open(json_file_path, 'r') as json_file:
                                result_data = json.load(json_file)
                                
                                # Calcula a média para cada lista
                                for key in result_data.keys():
                                    if isinstance(result_data[key], list):
                                        result_data[key] = sum(result_data[key]) / len(result_data[key])
                                
                                result_data['dataset_name'] = dataset_name
                                result_data['model_type'] = model_type
                                result_data['model_name'] = model_name
                                results.append(result_data)

    # Cria um DataFrame a partir dos resultados
    results_df = pd.DataFrame(results)

    columns_first = ['dataset_name', 'model_type', 'model_name']
    # Reorganiza as colunas para colocar 'dataset_name', 'model_type' e 'model_name' no início
    column_order = columns_first + [col for col in results_df.columns if col not in columns_first]
    results_df = results_df[column_order]
    
    return results_df

# Exemplo de uso
base_path = 'results'  # Altere para o caminho correto
results_df = load_results_to_dataframe(base_path)

print("DataFrame de Resultados:")
print(results_df)

results_df.to_csv('test.csv')
