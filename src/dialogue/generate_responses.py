import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from os import path

import logzero
import pandas as pd
import torch
import yaml
from logzero import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertJapaneseTokenizer, EncoderDecoderModel
from utils import add_special_tokens_, load_dataset

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
        """Set additional parameters."""
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
        """Initialize evaluating components."""
        self.args = None

        self.test_data = None
        self.test_dataloader = None

        self.utt_before = None
        self.utt_after = None

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

        with open("data/emp-persona/raw/test.src") as file:
            self.utt_before = file.read().splitlines()

        with open(path.join(self.args.data_dir, "test.src")) as file:
            self.utt_after = file.read().splitlines()

    def evaluating(self) -> None:
        self.model.eval()

        responses = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                outputs = self.model.generate(
                    batch["input_ids"].to(self.args.device),
                    do_sample=True,
                    top_p=self.args.top_p,
                    length_penalty=self.args.length_penalty,
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                    num_return_sequences=1,
                )
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                response = list(map(lambda x: x.replace(" ", ""), response))
                responses += response

        df = pd.DataFrame()
        df["utt_before"] = self.utt_before
        df["utt_after"] = self.utt_after

        diff = []
        for utt_b, utt_a in zip(self.utt_before, self.utt_after):
            if utt_a != utt_b:
                diff.append(True)
            else:
                diff.append(False)
        df["diff"] = diff
        df["response"] = responses
        df.to_csv(path.join(LOG_DIR_BASENAME, self.args.now_time, "result.csv"))


def create_parser() -> argparse.ArgumentParser:
    """Create parser."""
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
