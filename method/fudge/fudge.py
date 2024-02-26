from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LogitsProcessorList
from method.fudge.classifier import GuideClassifier
from method.fudge.logits_processor import ClassifierLogitsProcessor
import torch

class ModelResourceManager:
    """
    Manages resources for models and tokenizers, including loading a language model and a classifier.
    """
    def __init__(self, model_path, classifier_path, base_model_path):
        """
        Initializes the resource manager by loading the required models and tokenizers.
        
        Args:
            model_path (str): Path to the causal language model.
            classifier_path (str): Path to the classifier model.
            base_model_path (str): Path to the base model for the classifier.
        """
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_path)
        self.classifier, self.classifier_tokenizer = self.load_classifier(classifier_path, base_model_path)

    def load_model_and_tokenizer(self, model_path):
        """
        Loads a causal language model and its tokenizer.
        
        Args:
            model_path (str): Path to the causal language model.
            
        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Ensure that tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def load_classifier(self, classifier_path, base_model_path):
        """
        Loads a classifier and its tokenizer.
        
        Args:
            classifier_path (str): Path to the classifier model.
            base_model_path (str): Path to the base model for the classifier.
            
        Returns:
            tuple: A tuple containing the loaded classifier and its tokenizer.
        """
        classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).to('cuda').eval()

        # Load classifier state dict
        state_dict = torch.load(f"{classifier_path}/pytorch_model.bin")
        # Adjust for DataParallel wrapper
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

        classifier = GuideClassifier(base_model)
        classifier = torch.nn.DataParallel(classifier).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        classifier.load_state_dict(state_dict)

        return classifier, classifier_tokenizer

class FudgeTextGenerator:
    """
    Generates text using a language model adjusted by a classifier for specific tasks (e.g., mitigating toxicity).
    """
    def __init__(self, resource_manager):
        """
        Initializes the text generator with models and tokenizers from a given resource manager.
        
        Args:
            resource_manager (ModelResourceManager): A resource manager containing the models and tokenizers.
        """
        self.model = resource_manager.model
        self.tokenizer = resource_manager.tokenizer
        self.classifier = resource_manager.classifier
        self.classifier_tokenizer = resource_manager.classifier_tokenizer

    def generate_fudge_texts(self, prompt, generation_configs, fudge_alpha=1.0, task="toxicMitigation"):
        """
        Generates text based on a given prompt, applying adjustments from a classifier to influence generation.
        
        Args:
            prompt (str): The initial text prompt for generation.
            generation_configs (dict): Configuration for the generation process (e.g., max_length).
            fudge_alpha (float, optional): The weight of the classifier's influence on generation. Defaults to 1.0.
            task (str, optional): The task identifier for classifier adjustments. Defaults to "toxicMitigation".
        
        Returns:
            str: The generated text.
        """
        # Encode input text
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to('cuda')

        # Initialize logits processor with classifier adjustments
        classifier_logits_processor = ClassifierLogitsProcessor(
            classifier=self.classifier,
            gen_tokenizer=self.tokenizer,
            class_tokenizer=self.classifier_tokenizer,
            alpha=fudge_alpha,
            task=task
        )
        
        logits_processor = LogitsProcessorList([classifier_logits_processor])

        # Generate text with classifier-influenced adjustments
        outputs = self.model.generate(
            **input_ids,
            logits_processor=logits_processor,
            **generation_configs,
            pad_token_id=self.tokenizer.pad_token_id
        )

        input_len = input_ids["input_ids"].size(1)
        # Decode and return the generated text, skipping the initial prompt part
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
