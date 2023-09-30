from typing import List

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLMScorer:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str = "cpu") -> None:
        """Initialize masked language model scorer.

        Args:
            model (AutoModelForCausalLM): Masked language model.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            device (str, optional): Device to use. Defaults to "cpu".
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)

    def score_sentences(self, sentences: List[str], normalize: bool = False, per_token: bool = False) -> List[float]:
        """Score sentences.

        Args:
            sentences (List[str]): List of sentences.
            normalize (bool, optional): Normalize the score by the number of tokens. Defaults to False.
            per_token (bool, optional): Return scores for each token. Defaults to False.
        Returns:
            List[float]: List of scores.
        """
        return [self.score_sentence(text, normalize=normalize, per_token=per_token) for text in sentences]

    def score_sentence(self, text: str, normalize: bool = False, per_token: bool = False) -> float:
        """Score a sentence.

        Args:
            text (str): Sentence
            normalize (bool, optional): Normalize the score by the number of tokens. Defaults to False.
            per_token (bool, optional): Return scores for each token. Defaults to False.
        Returns:
            float: Score.
        """
        tokenized_text = self.tokenizer.tokenize(text)
        input_ids = []
        masked_indexes = []
        attention_mask = []
        token_type_ids = []
        for i in range(len(tokenized_text)):
            masked_text = tokenized_text.copy()
            masked_text[i] = "[MASK]"
            masked_indexes.append(i)
            masked_input_ids = self.tokenizer.encode(masked_text)
            input_ids.append(masked_input_ids)
            attention_mask.append([1] * len(masked_input_ids))
            token_type_ids.append([0] * len(masked_input_ids))

        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, device=self.device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        token_scores = []
        log_prob_score = 0.0
        for i, mask_index in enumerate(masked_indexes):
            predictions = logits[i].squeeze(0)
            probs = softmax(predictions, dim=1)
            masked_token_id = self.tokenizer.convert_tokens_to_ids([tokenized_text[mask_index]])[0]
            log_prob = np.log(probs[mask_index + 1, masked_token_id].cpu().numpy()).item()
            token_scores.append(log_prob)
            log_prob_score += log_prob

        if normalize:
            log_prob_score /= len(tokenized_text)
        if per_token:
            return token_scores
        else:
            return log_prob_score
