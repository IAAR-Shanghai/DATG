import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelResourceManager:
    """
    A resource manager for loading and storing the model and tokenizer.
    """
    def __init__(self, model_path):
        """
        Initializes the resource manager by loading the model and tokenizer.
        
        Args:
            model_path (str): The path or name of the model to load.
        """
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_path)

    def load_model_and_tokenizer(self, model_path):
        """
        Loads the model and tokenizer from a specified path.
        
        Args:
            model_path (str): The path or name of the model to load.
            
        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

class PreaddTextGenerator:
    """
    A text generator that adjusts logits based on a difference between 
    the prompt and a prefix+prompt combination to generate text.
    """
    def __init__(self, model_resource_manager):
        """
        Initializes the text generator with a model and tokenizer.
        
        Args:
            model_resource_manager (ModelResourceManager): A resource manager containing the model and tokenizer.
        """
        self.model = model_resource_manager.model
        self.tokenizer = model_resource_manager.tokenizer

    def compute_next_token_logits(self, input_ids, attention_mask):
        """
        Computes the logits for the next token in the sequence.
        
        Args:
            input_ids (torch.Tensor): Tensor of token ids.
            attention_mask (torch.Tensor): Tensor representing the attention mask.
            
        Returns:
            torch.Tensor: The logits for the next token.
        """
        outputs = self.model(input_ids, attention_mask=attention_mask).logits[:, -1, :]
        return outputs

    def adjust_logits(self, logits, diff, strength, temperature):
        """
        Adjusts the logits based on the difference and other parameters.
        
        Args:
            logits (torch.Tensor): The original logits.
            diff (torch.Tensor): The difference to adjust the logits with.
            strength (float): How strongly to apply the adjustment.
            temperature (float): The temperature for scaling the logits.
            
        Returns:
            torch.Tensor: The adjusted logits.
        """
        adjusted_logits = (logits + diff * strength) / temperature
        return adjusted_logits

    def select_next_token(self, logits):
        """
        Selects the next token based on the logits.
        
        Args:
            logits (torch.Tensor): The logits from which to sample the next token.
            
        Returns:
            torch.Tensor: The id of the next token.
        """
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        return next_token

    def generate_preadd_text(self, prompt, prefix, generation_configs, strength=1):
        """
        Generates text by pre-adding a prefix to the prompt and adjusting the generation process.
        
        Args:
            prompt (str): The original prompt for generation.
            prefix (str): The prefix to add to the prompt for adjusted generation.
            generation_configs (dict): Configuration dict for generation, including 'max_new_tokens' and 'temperature'.
            strength (float): The strength of adjustment.
            
        Returns:
            str: The generated text.
        """
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")
        encoded_prefix = self.tokenizer(prefix + " " + prompt, return_tensors="pt")
        prompt_length = len(encoded_prompt["input_ids"][0])
        generated_ids = encoded_prompt["input_ids"]
        generated_attention_mask = encoded_prompt["attention_mask"]

        for _ in range(generation_configs.get("max_new_tokens")):
            prompt_logits = self.compute_next_token_logits(input_ids=encoded_prompt["input_ids"], attention_mask=encoded_prompt["attention_mask"])
            prefix_logits = self.compute_next_token_logits(input_ids=encoded_prefix["input_ids"], attention_mask=encoded_prefix["attention_mask"])

            diff = prompt_logits - prefix_logits
            final_logits = self.adjust_logits(prompt_logits, diff, strength, generation_configs.get("temperature"))

            next_token = self.select_next_token(final_logits)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            
            # Update attention mask for the next generation step
            attention_mask_update = torch.ones_like(next_token)
            generated_attention_mask = torch.cat((generated_attention_mask, attention_mask_update), dim=1)
            
            # Update encoded prompt and prefix for next generation step
            encoded_prompt["input_ids"] = torch.cat((encoded_prompt["input_ids"], next_token), dim=1)
            encoded_prompt["attention_mask"] = torch.cat((encoded_prompt["attention_mask"], attention_mask_update), dim=1)
            
            encoded_prefix["input_ids"] = torch.cat((encoded_prefix["input_ids"], next_token), dim=1)
            encoded_prefix["attention_mask"] = torch.cat((encoded_prefix["attention_mask"], attention_mask_update), dim=1)

        return self.tokenizer.decode(generated_ids[0][prompt_length:])