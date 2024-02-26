"""The code for the classifier that is used to calculate the classification scores for the evaluaion."""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class Classifier(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, base_model):
        config = base_model.config
        super(Classifier, self).__init__(config)

        self.sentence_transformer = base_model
        self.classification_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        def mean_pooling(model_output, attention_mask):
            # First element of model_output contains all token embeddings
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        model_output = self.sentence_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sentence_embeddings = mean_pooling(model_output, attention_mask)
        logits = self.classification_head(sentence_embeddings)
        return logits
