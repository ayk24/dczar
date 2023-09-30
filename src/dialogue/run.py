import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from os import path

import logzero
import torch
import yaml
from logzero import logger
from transformers import BertJapaneseTokenizer, EncoderDecoderModel
from utils import add_special_tokens_

LOG_DIR_BASENAME = "./outputs/logs/dialogue/"
LOGZERO_LOG_FILE = "generating.logzero.txt"

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "[CLS]",
    "eos_token": "[SEP]",
    "pad_token": "[PAD]",
    "additional_special_tokens": ("[SPK1]", "[SPK2]"),
}


@dataclass
class GeneratingArgs:
    """The parameters will be loaded from a yaml file."""

    model_name_or_path: str = None
    tokenizer_name: str = None

    device: torch.device = None
    n_gpu: int = None

    sep_token: str = "[SEP]"
    spk1_token: str = "[SPK1]"
    spk2_token: str = "[SPK2]"

    max_contexts: int = None
    encoder_max_length: int = None
    top_p: int = None
    length_penalty: int = None
    no_repeat_ngram_size: int = None

    now_time: int = str

    def set_additional_parameters(self) -> None:
        """Set additional parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.now_time = datetime.now().strftime("%Y%m%d%H%M")
        os.makedirs(path.join(LOG_DIR_BASENAME, self.now_time), exist_ok=True)

        logzero.loglevel(logging.INFO)
        logzero.logfile(path.join(LOG_DIR_BASENAME, self.now_time, LOGZERO_LOG_FILE))
        logger.warning(
            f"device: {self.device}, " f"n_gpu: {self.n_gpu}",
        )


class GeneratinigComponents:
    def __init__(self):
        """Initialize generating components."""
        self.args = None

        self.tokenizer = None
        self.model = None

        self.contexts = []

    def set_generating_components(self, args: GeneratingArgs) -> None:
        """Set generating components."""
        args.set_additional_parameters()
        self.args = args

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.args.tokenizer_name)
        self.model = EncoderDecoderModel.from_pretrained(self.args.model_name_or_path)

        add_special_tokens_(tokenizer=self.tokenizer, model=self.model)
        self.model.to(self.args.device)

    def add_contexts(self, speaker: str, utterance: str) -> None:
        """Add contexts (speaker and utterance))."""
        self.contexts.append({"speaker": speaker, "utterance": utterance})

    def make_input(self, speaker: str, utterance: str) -> str:
        """Make input for the model."""
        lines = []
        for context in self.contexts[-self.args.max_contexts :]:
            lines.append(context["speaker"] + context["utterance"] + self.args.sep_token)

        inputs = ""
        for line in lines[::-1]:
            if len(inputs) + len(line) > self.args.encoder_max_length - len(utterance):
                break
            inputs = line + inputs

        inputs += speaker + utterance + self.args.sep_token
        inputs = inputs[: -len(self.args.sep_token)]

        return inputs

    def generate(self, utterance: str) -> str:
        """Generate a response."""
        context = self.make_input(self.args.spk2_token, utterance)
        logger.info("[*] context: " + str(context))
        self.add_contexts(self.args.spk2_token, utterance)

        inputs = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.args.encoder_max_length)
        inputs = {k: torch.tensor(v) for k, v in inputs.items()}

        outputs = self.model.generate(
            inputs["input_ids"].unsqueeze(0).to(self.args.device),
            do_sample=True,
            top_p=self.args.top_p,
            length_penalty=self.args.length_penalty,
            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
        )
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = responses[0].replace(" ", "")
        self.add_contexts(self.args.spk1_token, response)
        return response


def create_parser() -> argparse.ArgumentParser:
    """Create parser."""
    parser = argparse.ArgumentParser(description="Generating")
    parser.add_argument("--yaml_file", type=path.abspath, required=True, help="Path to yaml file")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    params_dict = yaml.safe_load(open(args.yaml_file))
    params = GeneratingArgs(**params_dict)

    gc = GeneratinigComponents()
    gc.set_generating_components(params)
    while True:
        utterance = input(">> ")
        logger.info("[*] user: " + utterance)
        response = gc.generate(utterance)
        logger.info("[*] system: " + str(response))


if __name__ == "__main__":
    main()
