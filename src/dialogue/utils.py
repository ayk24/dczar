from os import path
from typing import Dict

import datasets
import evaluate
import pandas as pd
from datasets import Dataset
from transformers import BertJapaneseTokenizer, EncoderDecoderModel

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "[CLS]",
    "eos_token": "[SEP]",
    "pad_token": "[PAD]",
    "additional_special_tokens": ("[SPK1]", "[SPK2]"),
}


def add_special_tokens_(tokenizer: BertJapaneseTokenizer, model: EncoderDecoderModel, update_model=True) -> None:
    """Add special tokens to the tokenizer and the model if they have not already been added."""
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0 and update_model:
        model.encoder.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
        model.decoder.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def load_dataset(data_dir: path.abspath, split: str) -> Dataset:
    """Load dataset."""
    dialogs = []
    with open("{}{}.src".format(data_dir, split)) as src_file, open("{}{}.dst".format(data_dir, split)) as dst_file:
        for history, response in zip(src_file, dst_file):
            dialogue = [history.replace("\n", ""), response.replace("\n", "")]
            dialogs.append(dialogue)

    df = pd.DataFrame(data=dialogs, columns=["history", "response"])
    return Dataset.from_pandas(df)


def compute_average(score_list: list) -> float:
    """Compute average score."""
    return sum(score_list) / len(score_list)


def compute_bleu_score(tokenizer: BertJapaneseTokenizer, pred_str: list, label_str: list) -> Dict:
    """Compute BLEU score."""
    bleu = datasets.load_metric("bleu")
    predictions = [tokenizer.tokenize(sentence) for sentence in pred_str]
    references = [[tokenizer.tokenize(sentence)] for sentence in label_str]
    bleu_output = bleu.compute(predictions=predictions, references=references)
    return bleu_output


def compute_rouge_score(pred_str: list, label_str: list) -> Dict:
    """Compute ROUGE score"""
    rouge = datasets.load_metric("rouge")
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    return {
        "rouge_1": rouge_output["rouge1"].mid.fmeasure,
        "rouge_2": rouge_output["rouge2"].mid.fmeasure,
        "rouge_L": rouge_output["rougeL"].mid.fmeasure,
    }


def compute_bert_score(pred_str: list, label_str: list) -> Dict:
    """Compute BERT score."""
    bertscore = evaluate.load("bertscore")
    bertscore_output = bertscore.compute(predictions=pred_str, references=label_str, lang="ja")
    precision = compute_average(bertscore_output["precision"])
    recall = compute_average(bertscore_output["recall"])
    f1 = compute_average(bertscore_output["f1"])
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_meteor_score(pred_str: list, label_str: list) -> Dict:
    """Compute METEOR score."""
    meteor = evaluate.load("meteor")
    meteor_output = meteor.compute(predictions=pred_str, references=label_str)
    return meteor_output


def compute_dist_score(pred_str: list) -> Dict:
    """Compute DIST score."""
    inter_dist_1, inter_dist_2 = [], []
    unigrams_all, bigrams_all = 0, 0

    for pred in pred_str:
        unigram = [tuple(pred[i : i + 1]) for i in range(len(pred))]
        bigram = [tuple(pred[i : i + 2]) for i in range(len(pred) - 1)]

        inter_dist_1.extend(unigram)
        inter_dist_2.extend(bigram)

        unigrams_all += len(unigram)
        bigrams_all += len(bigram)

    dist_1 = len(set(inter_dist_1)) / unigrams_all if unigrams_all != 0 else 0.0
    dist_2 = len(set(inter_dist_2)) / bigrams_all if bigrams_all != 0 else 0.0

    return {"dist_1": dist_1, "dist_2": dist_2}
