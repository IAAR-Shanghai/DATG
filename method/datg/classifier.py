import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class GuideClassifier(PreTrainedModel):
    """
    A classifier that wraps a base transformer model to produce sentence-level embeddings
    and applies a linear layer for classification.
    
    Inherits from PreTrainedModel to leverage Hugging Face's serialization and configuration
    utilities.
    """
    config_class = PretrainedConfig
    
    def __init__(self, base_model):
        """
        Initializes the classifier with a base transformer model.
        
        Args:
            base_model (PreTrainedModel): The transformer model used for extracting
                                          sentence embeddings.
        """
        config = base_model.config
        super(GuideClassifier, self).__init__(config)
        
        self.sentence_transformer = base_model
        self.classification_head = nn.Linear(config.hidden_size, 1)
        # Linear layer for classification. It maps the pooled sentence embedding
        # to a single logit as output.
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Defines the forward pass of the classifier.
        
        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            
        Returns:
            torch.Tensor: Logits produced by the classification head.
        """
        def mean_pooling(model_output, attention_mask):
            """
            Applies mean pooling on the token embeddings using the attention mask.
            
            Args:
                model_output (Tuple[torch.Tensor]): Output from the base transformer model.
                attention_mask (torch.Tensor): The attention mask for the input tokens.
                
            Returns:
                torch.Tensor: The pooled sentence embedding.
            """
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # Applies weighted mean where weights are provided by the attention mask.

        # Retrieve base model output.
        model_output = self.sentence_transformer(input_ids=input_ids, 
                                                 attention_mask=attention_mask, 
                                                 token_type_ids=token_type_ids)

        # Pool the base model output to get sentence embeddings.
        sentence_embeddings = mean_pooling(model_output, attention_mask)
        
        # Pass sentence embeddings through the classification head.
        logits = self.classification_head(sentence_embeddings)

        return logits
