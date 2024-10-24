class PromptGenerator:
    @staticmethod
    def generate_prompt(repo_id: str,
                        system_prompt: str, 
                        user_prompt: str, 
                        text: str) -> str:
        if repo_id == "bartowski/Phi-3.5-mini-instruct-GGUF":
            prompt = f"<|system|>{system_prompt}<|end|><|user|>{user_prompt}\n{text}<|end|><|assistant|>"
        elif repo_id == "SanctumAI/gemma-2-9b-it-GGUF":
            prompt = f"<bos><start_of_turn>user\n{system_prompt}\n{user_prompt}\n{text}<end_of_turn><start_of_turn>model"
        elif repo_id == "lmstudio-community/Llama-3.2-1B-Instruct-GGUF" or repo_id == "lmstudio-community/Llama-3.2-3B-Instruct-GGUF":
            prompt = f"<|start_header_id|>{system_prompt}<|end_header_id|><|eot_id|><|start_header_id|>{user_prompt}\n{text}<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif repo_id == "bartowski/Mistral-7B-Instruct-v0.3-GGUF":
            prompt = f"<s>[INST]{system_prompt}\n{user_prompt}\n{text}[/INST]</s>"
        elif repo_id == "lmstudio-community/Yi-1.5-9B-Chat-GGUF":
            prompt = f"<|im_start|>system{system_prompt}<|im_end|><|im_start|>user{user_prompt}\n{text}<|im_end|><|im_start|>assistant<|im_end|>"

        return prompt