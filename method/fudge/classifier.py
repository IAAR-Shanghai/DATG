import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class GuideClassifier(PreTrainedModel):
    """
    A classifier model that uses a pre-trained transformer model as a sentence encoder
    and adds a linear layer on top to perform binary classification.
    """
    config_class = PretrainedConfig
    
    def __init__(self, base_model):
        """
        Initializes the classifier with a base transformer model.
        
        Args:
            base_model (PreTrainedModel): A pre-trained transformer model used for sentence encoding.
        """
        config = base_model.config
        super(GuideClassifier, self).__init__(config)
        
        self.sentence_transformer = base_model
        # The classification head that maps the encoded sentence embeddings to a single logit
        self.classification_head = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the classifier.
        
        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
        
        Returns:
            torch.Tensor: The logits representing the model's predictions.
        """

        def mean_pooling(model_output, attention_mask):
            """
            Performs mean pooling on the token embeddings using the attention mask.
            
            Args:
                model_output (tuple): The output from the base transformer model.
                attention_mask (torch.Tensor): The attention mask for the input tokens.
            
            Returns:
                torch.Tensor: The mean-pooled sentence embeddings.
            """
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Get the output from the base transformer model
        model_output = self.sentence_transformer(input_ids=input_ids, 
                                                 attention_mask=attention_mask, 
                                                 token_type_ids=token_type_ids)

        # Obtain sentence embeddings through mean pooling
        sentence_embeddings = mean_pooling(model_output, attention_mask)
        
        # Compute logits using the classification head
        logits = self.classification_head(sentence_embeddings)

        return logits
