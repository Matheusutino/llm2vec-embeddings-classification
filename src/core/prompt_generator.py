class PromptGenerator:
    @staticmethod
    def generate_prompt_llama_cpp(repo_id: str,
                                system_prompt: str, 
                                user_prompt: str, 
                                text: str) -> str:
        if repo_id == "bartowski/Phi-3.5-mini-instruct-GGUF":
            prompt = f"<|system|>{system_prompt}<|end|><|user|>{user_prompt}\n{text}<|end|><|assistant|>"
        elif repo_id == "SanctumAI/gemma-2-9b-it-GGUF":
            prompt = f"<bos><start_of_turn>user\n{system_prompt}\n{user_prompt}\n{text}<end_of_turn><start_of_turn>model"
        elif repo_id == "lmstudio-community/Llama-3.2-1B-Instruct-GGUF" or repo_id == "lmstudio-community/Llama-3.2-3B-Instruct-GGUF":
            prompt = f"<|start_header_id|>{system_prompt}<|end_header_id|><|eot_id|><|start_header_id|>{user_prompt}\n{text}<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif repo_id == "bartowski/Mistral-Nemo-Instruct-2407-GGUF":
            prompt = f"<s>[INST]{system_prompt}\n{user_prompt}\n{text}[/INST]</s>"
        elif repo_id == "bartowski/Qwen2.5-7B-Instruct-GGUF":
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}\n{text}<|im_end|>\n<|im_start|>assistant"
        elif repo_id == "bartowski/aya-expanse-8b-GGUF":
            prompt = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}\n{text}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

        return prompt

    @staticmethod
    def generate_prompt_ollama(system_prompt: str, 
                               user_prompt: str, 
                               text: str) -> str:
        prompt = f"{system_prompt}\n{user_prompt}\n{text}"

        return prompt