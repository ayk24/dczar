import argparse
import pickle
from os import path
from typing import Dict, List

import torch
from jmlm_scoring.jmlm.scorers import MLMScorer
from transformers import BertForMaskedLM, BertJapaneseTokenizer

CL_TOHOKU_BERT = "cl-tohoku/bert-base-japanese-whole-word-masking"


def complement_of_case(line: str, parsed: Dict, pred: Dict, case_info: str) -> str:
    """Complement of case.

    Args:
        line (str): Line.
        parsed (Dict): Parsed line.
        pred (Dict): Predicate.
        case_info (str): Case information.
    Returns:
        str: Complement of case.
    """
    complement = ""

    if case_info == "ga":
        case_str = "が"
    elif case_info == "o":
        case_str = "を"
    elif case_info == "ni":
        case_str = "に"

    if case_info in pred["result"]:
        if pred["result"][case_info] == "私（一人称）":
            pred["result"][case_info] = "私"
        elif pred["result"][case_info] == "あなた（二人称）":
            pred["result"][case_info] = "あなた"

        if pred["result"][case_info] != "一般（その他）":
            if pred["result"][case_info] not in line:
                is_noun = True
                for idx, word_dict in parsed.items():
                    for key, value in word_dict.items():
                        if key == pred["result"][case_info]:
                            if value != "名詞":
                                is_noun = False

                if is_noun:
                    complement += pred["result"][case_info] + case_str
                else:
                    complement += pred["result"][case_info] + "こと" + case_str

    return complement


def complement(model_name_or_path: str, lines: List, parsed_lines: List, preds: List) -> List[str]:
    """Complement.

    Args:
        model_name_or_path (str): Model name or path.
        lines (List): Lines.
        parsed_lines (List): Parsed lines.
        preds (List): Predicates.
    Returns:
        List[str]: Complemented lines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(model_name_or_path)
    tokenizer = BertJapaneseTokenizer.from_pretrained(CL_TOHOKU_BERT)
    scorer = MLMScorer(model, tokenizer, device=device)

    last_send_id = -1
    for pred in preds:
        if pred["sent_id"] != last_send_id:
            start_idx = 0

        line = lines[pred["sent_id"]]
        parsed = parsed_lines[pred["sent_id"]]
        end_idx = line.find(pred["result"]["pred"], start_idx)  # pred index

        complement = ""
        complement += complement_of_case(line=line[start_idx:end_idx], parsed=parsed, pred=pred, case_info="ga")
        complement += complement_of_case(line=line[start_idx:end_idx], parsed=parsed, pred=pred, case_info="o")
        complement += complement_of_case(line=line[start_idx:end_idx], parsed=parsed, pred=pred, case_info="ni")

        candidates = []
        candidates.append(line)

        start_of_predicate = min(pred["word_ids"]) - 1
        for idx, word_dict in parsed.items():
            if complement == "":
                break

            if int(idx) > int(start_of_predicate):
                candidates.append(line[:end_idx] + complement + line[end_idx:])
                break

            for key, value in word_dict.items():
                if key in line[start_idx:end_idx]:
                    idx = line[start_idx:end_idx].find(key) + start_idx
                    candidates.append(line[:idx] + complement + line[idx:])

        end_of_predicate = max(pred["word_ids"])
        for idx in range(end_of_predicate + 1):
            if idx in parsed:
                parsed.pop(idx)

        result = scorer.score_sentences(candidates)
        if len(result) != 0:
            best_idx = result.index(max(result))
            lines[pred["sent_id"]] = candidates[best_idx]

        start_idx = line.find(pred["result"]["pred"], end_idx - 1) + len(pred["result"]["pred"])
        last_send_id = pred["sent_id"]

    return lines


def create_parser() -> argparse.ArgumentParser:
    """Create parser."""
    parser = argparse.ArgumentParser(description="Extracting Predicates")
    parser.add_argument("--in", dest="in_file", type=path.abspath, required=True, help="Path to input file.")
    parser.add_argument("--parsed_text", dest="parsed_text", type=path.abspath, required=True, help="Path to parsed text.")
    parser.add_argument("--preds", dest="preds", type=path.abspath, required=True, help="Path to preds (pickle).")
    parser.add_argument("--model", dest="model", type=path.abspath, required=True, help="Path to model.")
    parser.add_argument("--out", dest="out_file", type=path.abspath, required=True, help="Path to output file.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    lines = []
    with open(args.in_file) as in_file:
        for line in in_file:
            lines.append(line)

    parsed_lines = []
    with open(args.parsed_text) as in_file_parsed:
        word_id = 0
        cases = {}
        for parsed_line in in_file_parsed:
            tokens = parsed_line.strip().split("\t")
            if tokens[0] == "EOS":
                parsed_lines.append(cases)
                word_id = 0
                cases = {}
            elif parsed_line[0] != "*":
                case_name = tokens[1].split(",")
                cases[word_id] = {tokens[0]: case_name[0]}
                word_id += 1

    with open(args.preds, "rb") as in_file_preds:
        preds = pickle.load(in_file_preds)

    complement_lines = complement(model_name_or_path=args.model, lines=lines, parsed_lines=parsed_lines, preds=preds)

    with open(args.out_file, "w", encoding="utf-8") as out_file:
        out_file.write("".join(complement_lines))


if __name__ == "__main__":
    main()
