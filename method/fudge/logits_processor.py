from transformers import LogitsProcessor
import torch

class ClassifierLogitsProcessor(LogitsProcessor):
    """
    A logits processor that adjusts the logits of generated tokens based on a classifier's output.
    This is used to influence the text generation process, for example, by mitigating toxic content.
    """
    def __init__(self, classifier, gen_tokenizer, class_tokenizer, alpha, task="toxicMitigation", top_k=100, max_length=512):
        """
        Initializes the logits processor with a classifier and related configurations.
        
        Args:
            classifier (torch.nn.Module): The classifier model used to adjust logits.
            gen_tokenizer (PreTrainedTokenizer): Tokenizer for the generation model.
            class_tokenizer (PreTrainedTokenizer): Tokenizer for the classifier model.
            alpha (float): Adjustment factor for classifier logits.
            task (str, optional): Specifies the task for the logits adjustment. Defaults to "toxicMitigation".
            top_k (int, optional): Number of top logits to consider for adjustment. Defaults to 100.
            max_length (int, optional): Maximum sequence length for classifier input. Defaults to 512.
        """
        self.classifier = classifier
        self.gen_tokenizer = gen_tokenizer
        self.class_tokenizer = class_tokenizer
        self.alpha = alpha
        self.max_length = max_length
        self.top_k = top_k
        self.task = task

    def __call__(self, input_ids, scores):
        """
        Adjusts generation logits based on the classifier's predictions.
        
        Args:
            input_ids (torch.Tensor): Current input IDs for generation.
            scores (torch.Tensor): Logits for the next token to be generated.
        
        Returns:
            torch.Tensor: Adjusted logits.
        """
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            # Decode the current input IDs to text for classifier processing.
            current_input_ids = input_ids[batch_idx, :].unsqueeze(0)
            current_text = self.gen_tokenizer.decode(current_input_ids[0], skip_special_tokens=True)

            # Select the top_k logits for adjustment.
            top_logits, top_indices = torch.topk(scores[batch_idx], self.top_k, dim=-1)

            # Generate new text sequences for the top_k logits.
            new_texts = [current_text + self.gen_tokenizer.decode(indices.unsqueeze(0), skip_special_tokens=True) for indices in top_indices]

            # Prepare inputs for the classifier.
            inputs = self.class_tokenizer(new_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

            with torch.no_grad():  # Disable gradient computation.
                logits = self.classifier(**inputs)  # Get classification logits.

            # Adjust logits based on the task.
            if self.task == "2Positive":
                probs = torch.log(torch.sigmoid(logits))  # Log probabilities for positive sentiment.
            elif self.task == "2Negative":
                probs = torch.log(1 - torch.sigmoid(logits))  # Log probabilities for negative sentiment.
            elif self.task == "toxicMitigation":
                probs = torch.log(1 - torch.sigmoid(logits))  # Log probabilities for non-toxic content.

            # Calculate the final adjusted logits.
            final_logits = top_logits + self.alpha * probs.squeeze()

            # Update the scores with adjusted logits.
            for i, index in enumerate(top_indices):
                scores[batch_idx, index] = final_logits[i]

        return scores
