import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from os import path
from typing import Dict

import logzero
import torch
import yaml
from logzero import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertJapaneseTokenizer, EncoderDecoderModel
from utils import (
    add_special_tokens_,
    compute_bert_score,
    compute_bleu_score,
    compute_dist_score,
    compute_meteor_score,
    compute_rouge_score,
    load_dataset,
)

LOG_DIR_BASENAME = "./outputs/logs/dialogue"
LOGZERO_LOG_FILE = "evaluating.logzero.txt"


@dataclass
class EvaluatingArgs:
    """The parameters will be loaded from a yaml file."""

    data_dir: path.abspath

    model_name_or_path: str = None
    tokenizer_name: str = None

    device: torch.device = None
    n_gpu: int = None

    batch_size: int = None
    encoder_max_length: int = None
    decoder_max_length: int = None
    top_p: int = None
    length_penalty: int = None
    no_repeat_ngram_size: int = None

    now_time: str = None

    def set_additional_parameters(self) -> None:
        """Set additional parameters"""
        assert path.exists(self.data_dir), f"not found: {self.data_dir}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.now_time = datetime.now().strftime("%Y%m%d%H%M")
        os.makedirs(path.join(LOG_DIR_BASENAME, self.now_time), exist_ok=True)

        logzero.loglevel(logging.INFO)
        logzero.logfile(path.join(LOG_DIR_BASENAME, self.now_time, LOGZERO_LOG_FILE))
        logger.warning(
            f"device: {self.device}, " f"n_gpu: {self.n_gpu}",
        )


class EvaluatingComponents:
    def __init__(self):
        self.args = None

        self.test_data = None
        self.test_dataloader = None

        self.tokenizer = None
        self.model = None

    def set_evaluating_components(self, args: EvaluatingArgs) -> None:
        """Set evaluating components."""
        args.set_additional_parameters()
        self.args = args

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.args.tokenizer_name)
        self.model = EncoderDecoderModel.from_pretrained(self.args.model_name_or_path)

        add_special_tokens_(tokenizer=self.tokenizer, model=self.model)
        self.model.to(self.args.device)

        self.test_data = load_dataset(data_dir=self.args.data_dir, split="test")

        def process_data_to_model_inputs(batch):
            for idx, example in enumerate(batch["history"]):
                batch["history"][idx] = example

            for idx, example in enumerate(batch["response"]):
                batch["response"][idx] = example

            inputs = self.tokenizer(
                batch["history"],
                padding="max_length",
                truncation=True,
                max_length=self.args.encoder_max_length,
            )
            outputs = self.tokenizer(
                batch["response"],
                padding="max_length",
                truncation=True,
                max_length=self.args.decoder_max_length,
            )

            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            batch["labels"] = outputs.input_ids.copy()
            batch["labels"] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
            ]

            return batch

        self.test_data = self.test_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=self.args.batch_size,
            remove_columns=["history", "response"],
        )

        self.test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.args.batch_size)

    def compute_metrics(self, pred_ids: torch.Tensor, label_ids: torch.Tensor) -> Dict:
        """Compute metrics

        Args:
            pred_ids (torch.Tensor): Predicted ids
            label_ids (torch.Tensor): Label ids

        Returns:
            Dict: Result
        """
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        try:
            bleu_output = compute_bleu_score(tokenizer=self.tokenizer, pred_str=pred_str, label_str=label_str)
            bleu_output["precision_1_gram"] = bleu_output["precisions"][0]
            bleu_output["precision_2_gram"] = bleu_output["precisions"][1]
            bleu_output["precision_3_gram"] = bleu_output["precisions"][2]
            bleu_output["precision_4_gram"] = bleu_output["precisions"][3]
        except Exception:
            bleu_output = {
                "bleu": 0.0,
                "precision_1_gram": 0.0,
                "precision_2_gram": 0.0,
                "precision_3_gram": 0.0,
                "precision_4_gram": 0.0,
            }

        rouge_output = compute_rouge_score(pred_str=pred_str, label_str=label_str)
        dist_output = compute_dist_score(pred_str=pred_str)
        meteor_output = compute_meteor_score(pred_str=pred_str, label_str=label_str)
        bertscore_output = compute_bert_score(pred_str=pred_str, label_str=label_str)

        return {
            "bleu_1": bleu_output["precision_1_gram"],
            "bleu_2": bleu_output["precision_2_gram"],
            "bleu_3": bleu_output["precision_3_gram"],
            "bleu_4": bleu_output["precision_4_gram"],
            "rouge_1": rouge_output["rouge_1"],
            "rouge_2": rouge_output["rouge_2"],
            "rouge_l": rouge_output["rouge_L"],
            "dist_1": dist_output["dist_1"],
            "dist_2": dist_output["dist_2"],
            "meteor": meteor_output["meteor"],
            "bertscore": bertscore_output["f1"],
        }

    def evaluating(self) -> None:
        self.model.eval()

        bleu_1 = 0.0
        bleu_2 = 0.0
        bleu_3 = 0.0
        bleu_4 = 0.0
        rouge_1 = 0.0
        rouge_2 = 0.0
        rouge_l = 0.0
        dist_1 = 0.0
        dist_2 = 0.0
        meteor = 0.0
        bertscore = 0.0

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                pred_ids = self.model.generate(
                    batch["input_ids"].to(self.args.device),
                    do_sample=True,
                    top_p=self.args.top_p,
                    length_penalty=self.args.length_penalty,
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                )

                metrics = self.compute_metrics(pred_ids=pred_ids, label_ids=batch["labels"])

                bleu_1 += metrics["bleu_1"]
                bleu_2 += metrics["bleu_2"]
                bleu_3 += metrics["bleu_3"]
                bleu_4 += metrics["bleu_4"]

                rouge_1 += metrics["rouge_1"]
                rouge_2 += metrics["rouge_2"]
                rouge_l += metrics["rouge_l"]

                dist_1 += metrics["dist_1"]
                dist_2 += metrics["dist_2"]

                meteor += metrics["meteor"]

                bertscore += metrics["bertscore"]

            logger.info("**** Score ****")
            logger.info("BLEU-1: {}".format(bleu_1 / len(self.test_dataloader)))
            logger.info("BLEU-2: {}".format(bleu_2 / len(self.test_dataloader)))
            logger.info("BLEU-3: {}".format(bleu_3 / len(self.test_dataloader)))
            logger.info("BLEU-4: {}".format(bleu_4 / len(self.test_dataloader)))

            logger.info("ROUGE-1: {}".format(rouge_1 / len(self.test_dataloader)))
            logger.info("ROUGE-2: {}".format(rouge_2 / len(self.test_dataloader)))
            logger.info("ROUGE-L: {}".format(rouge_l / len(self.test_dataloader)))

            logger.info("DIST-1: {}".format(dist_1 / len(self.test_dataloader)))
            logger.info("DIST-2: {}".format(dist_2 / len(self.test_dataloader)))

            logger.info("METEOR: {}".format(meteor / len(self.test_dataloader)))

            logger.info("BERTScore: {}".format(bertscore / len(self.test_dataloader)))

            logger.info("**** CSV format ****")
            logger.info(
                "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                    bleu_1 / len(self.test_dataloader),
                    bleu_2 / len(self.test_dataloader),
                    bleu_3 / len(self.test_dataloader),
                    bleu_4 / len(self.test_dataloader),
                    rouge_1 / len(self.test_dataloader),
                    rouge_2 / len(self.test_dataloader),
                    rouge_l / len(self.test_dataloader),
                    dist_1 / len(self.test_dataloader),
                    dist_2 / len(self.test_dataloader),
                    meteor / len(self.test_dataloader),
                    bertscore / len(self.test_dataloader),
                )
            )


def create_parser() -> argparse.ArgumentParser:
    """Create parser"""
    parser = argparse.ArgumentParser(description="Evaluating")
    parser.add_argument("--yaml_file", type=path.abspath, required=True, help="Path to yaml file")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    params_dict = yaml.safe_load(open(args.yaml_file))
    params = EvaluatingArgs(**params_dict)

    tc = EvaluatingComponents()
    tc.set_evaluating_components(params)

    tc.evaluating()


if __name__ == "__main__":
    main()
