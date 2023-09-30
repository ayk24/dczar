import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from os import path

import logzero
import torch
import yaml
from logzero import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertJapaneseTokenizer, EncoderDecoderModel
from utils import add_special_tokens_, load_dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


LOG_DIR_BASENAME = "./outputs/logs/dialogue/"
LOGZERO_LOG_FILE = "training.logzero.txt"
TENSOR_BOARD_LOG_FILE = "tensorboard_log"


@dataclass
class TrainingArgs:
    """The parameters will be loaded from a yaml file."""

    output_dir: path.abspath
    data_dir: path.abspath

    model_name_or_path: str = None
    tokenizer_name: str = None

    device: torch.device = None
    n_gpu: int = None

    per_gpu_train_batch_size: int = None
    per_gpu_eval_batch_size: int = None
    predict_with_generate: bool = False
    encoder_max_length: int = None
    decoder_max_length: int = None
    no_repeat_ngram_size: int = None
    length_penalty: int = None
    early_stopping: bool = False
    num_epochs: int = None

    now_time: str = None

    def set_additional_parameters(self) -> None:
        """Set additional parameters."""
        assert path.exists(self.data_dir), f"not found: {self.data_dir}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.now_time = datetime.now().strftime("%Y%m%d%H%M")
        os.makedirs(path.join(LOG_DIR_BASENAME, self.now_time), exist_ok=True)

        logzero.loglevel(logging.INFO)
        logzero.logfile(path.join(LOG_DIR_BASENAME, self.now_time, LOGZERO_LOG_FILE))
        logger.warning(
            f"device: {self.device}, " f"n_gpu: {self.n_gpu}",
        )


class TrainingComponents:
    def __init__(self):
        self.args = None
        self.writer = None

        self.train_data = None
        self.valid_data = None
        self.train_dataloader = None
        self.valid_dataloader = None

        self.tokenizer = None
        self.model = None
        self.optimizer = None

    def set_training_components(self, args: TrainingArgs) -> None:
        """Set training components."""
        args.set_additional_parameters()
        self.args = args
        self.writer = SummaryWriter(log_dir=path.join(LOG_DIR_BASENAME, args.now_time, TENSOR_BOARD_LOG_FILE))

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.tokenizer_name)
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name_or_path, args.model_name_or_path)

        add_special_tokens_(tokenizer=self.tokenizer, model=self.model)
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.config.decoder.is_decoder = True
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.no_repeat_ngram_size = args.no_repeat_ngram_size
        self.model.config.length_penalty = args.length_penalty
        self.model.config.early_stopping = args.early_stopping
        self.model.to(self.args.device)

        self.train_data = load_dataset(data_dir=self.args.data_dir, split="train")
        self.valid_data = load_dataset(data_dir=self.args.data_dir, split="valid")

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

        self.train_data = self.train_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=args.per_gpu_train_batch_size,
            remove_columns=["history", "response"],
        )

        self.valid_data = self.valid_data.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=args.per_gpu_eval_batch_size,
            remove_columns=["history", "response"],
        )

        self.train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.valid_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        self.train_dataloader = DataLoader(self.train_data, shuffle=True, batch_size=args.per_gpu_train_batch_size)
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=args.per_gpu_eval_batch_size)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def training(self) -> None:
        step = 0
        best_valid_epoch_loss = sys.float_info.max

        for epoch in range(self.args.num_epochs):
            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0

            for batch in tqdm(self.train_dataloader):
                self.model.train()

                input_ids = batch["input_ids"].to(self.args.device)
                attention_mask = batch["attention_mask"].to(self.args.device)
                labels = batch["labels"].to(self.args.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                train_loss = outputs.loss
                train_epoch_loss += train_loss.item()

                train_loss.backward()
                self.optimizer.step()

                step += 1

                if step % len(self.train_dataloader) == 0:
                    logger.info("**** Running evaluation ****")
                    self.model.eval()

                    with torch.no_grad():
                        for batch in tqdm(self.valid_dataloader):
                            input_ids = batch["input_ids"].to(self.args.device)
                            attention_mask = batch["attention_mask"].to(self.args.device)
                            labels = batch["labels"].to(self.args.device)

                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                            valid_loss = outputs.loss
                            valid_epoch_loss += valid_loss.item()

                    average_train_epoch_loss = train_epoch_loss / (len(self.train_dataloader) // 2)
                    average_valid_epoch_loss = valid_epoch_loss / len(self.valid_dataloader)

                    logger.info("**** Eval results ****")
                    logger.info(
                        "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}".format(
                            epoch + 1,
                            self.args.num_epochs,
                            step,
                            self.args.num_epochs * len(self.train_dataloader),
                            average_train_epoch_loss,
                            average_valid_epoch_loss,
                        )
                    )

                    self.writer.add_scalar("Loss/train", average_train_epoch_loss, epoch)
                    self.writer.add_scalar("Loss/valid", average_valid_epoch_loss, epoch)

                    if best_valid_epoch_loss > average_valid_epoch_loss:
                        logger.info("**** Update best model ****")
                        self.model.save_pretrained(f"{self.args.output_dir}")
                        best_valid_epoch_loss = average_valid_epoch_loss

        self.writer.close()
        logger.info("**** Finished Training ****")


def create_parser() -> argparse.ArgumentParser:
    """Create parser."""
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--yaml_file", type=path.abspath, required=True, help="Path to yaml file")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    params_dict = yaml.safe_load(open(args.yaml_file))
    params = TrainingArgs(**params_dict)

    tc = TrainingComponents()
    tc.set_training_components(params)

    tc.training()


if __name__ == "__main__":
    main()
