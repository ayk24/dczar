import argparse
import pickle
from itertools import chain, groupby
from os import path
from typing import List

import numpy as np
import torch
from decode import decode_for_pas
from instances import AS, AS_PZERO, GA, NI, PREDICATE, WO, NTCPasDocument, Pas, PasGoldLabel, PasGoldLabels
from iterators import PasBucketIterator
from logzero import logger
from models import AsModel, AsPzeroModel
from preprocess import PreprocessForFinetuningAS, PreprocessForFinetuningASPzero
from tokenizer import load_tokenizer
from torch.nn.utils.rnn import pad_sequence


def create_parser():
    parser = argparse.ArgumentParser(description="Pseudo Zero Pronoun Resolution Improves Zero Anaphora Resolution")

    group = parser.add_argument_group("essential elements")
    group.add_argument(
        "--parsed_text",
        type=path.abspath,
        required=True,
        help="Path to input file where the sentences were parsed by CaboCha and the position of predicate was added.",
    )
    parser.add_argument("--out", dest="out_file", type=path.abspath, required=True, help="Path to output file.")
    group.add_argument("--model", type=path.abspath, required=True, help="Path to trained model.")
    group.add_argument(
        "--model_type",
        type=str,
        default="as-pzero",
        choices=["as-pzero", "as"],
        help="Model types ('as' or 'as-pzero'). default='as-pzero'",
    )

    group.add_argument_group("optional")
    group.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")

    return parser


def convert_cabocha_format_to_training_instances(
    file_path: str,
    model_type: str = AS_PZERO,
):
    """
    This function extracts the surface forms and predicate positions from the input text.
    Here input text is parsed by CaboCha and the number indicating a target predicates is appended to the end of a line.

    Args:
        file_path (str)       : Path to input file (parsed)
        model_type (str)      : The type of trained model ("as" or "as-pzero")

    Returns:
        training_instances (List[Union[AsTrainingInstance, AsPzeroTrainingInstance]]): List of instances
        ntc_instance (NTCPasDocument): This instance contains the information of the predicate-argument before preprocessing
    """

    assert model_type in ["as", "as-pzero"], f"Unsupported type: {model_type}"

    # extract surfaces and numbers from input file
    doc_surfaces: List[List[str]] = []
    doc_predicate_positions: List[List[int]] = []

    sentence_surfaces: List[str] = []
    sentence_predicate_positions: List[int] = []

    with open(file_path) as fi:
        for is_eos, lines in groupby(fi, key=lambda x: x.startswith("EOS")):
            if is_eos:
                # append
                doc_surfaces.append(sentence_surfaces)
                doc_predicate_positions.append(sentence_predicate_positions)
                # init
                sentence_surfaces = []
                sentence_predicate_positions = []
            else:
                for line in lines:
                    if line.startswith("*"):
                        continue
                    # extract
                    surface, *_, tail_string = line.rstrip("\n").split("\t")
                    is_predicate = tail_string.isdigit()
                    # append
                    sentence_surfaces.append(surface)
                    sentence_predicate_positions.append(int(tail_string) if is_predicate else 0)

    # prepare for creating instances
    dummy_gold_labels = PasGoldLabels(
        ga=PasGoldLabel(gold_cases=[], case_name="ga", case_type=""),
        o=PasGoldLabel(gold_cases=[], case_name="o", case_type=""),
        ni=PasGoldLabel(gold_cases=[], case_name="ni", case_type=""),
    )

    padded_positions = pad_sequence(
        sequences=[torch.LongTensor(ids) for ids in doc_predicate_positions], batch_first=True, padding_value=0
    ).numpy()

    uniq_predicate_numbers = set(chain.from_iterable(doc_predicate_positions))
    uniq_predicate_numbers -= {0}  # '0' is not the index of predicate
    assert len(uniq_predicate_numbers) != 0, "Not found the position of predicates."

    # create the indices of predicate positions
    pas_list: List[Pas] = []

    for predicate_number in uniq_predicate_numbers:
        prd_sent_ids, prd_word_ids = np.where(padded_positions == predicate_number)

        assert len(set(prd_sent_ids)) == 1, "The predicate exists across sentences"
        sent_idx = prd_sent_ids.tolist()[0]
        word_ids = prd_word_ids.tolist()
        pas = Pas(
            prd_sent_idx=sent_idx,
            prd_word_ids=word_ids,
            gold_labels=dummy_gold_labels,
            alt_type="",
        )
        pas_list.append(pas)

    sent_ids = list(range(len(doc_surfaces)))
    ntc_instance = NTCPasDocument(
        file_path=file_path,
        sents=doc_surfaces,
        sent_ids=sent_ids,
        pas_list=pas_list,
    )

    # create the list of training instances
    if model_type == AS:
        preprocessor = PreprocessForFinetuningAS()
        training_instances = list(preprocessor.create_as_instances(ntc_instance))
    elif model_type == AS_PZERO:
        preprocessor = PreprocessForFinetuningASPzero()
        training_instances = list(preprocessor.create_as_pzero_instances(ntc_instance))
    else:
        raise ValueError(f"unsupported value: {model_type}")

    return training_instances, ntc_instance


def main():
    parser = create_parser()
    args = parser.parse_args()
    logger.info(args)

    # model type
    if args.model_type == "as-pzero":
        model = AsPzeroModel()
    elif args.model_type == "as":
        model = AsModel()
    else:
        raise ValueError(f"unsupported value: '{args.model_type}'")
    logger.info(f"model type: {args.model_type}")
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    if torch.cuda.is_available():
        model = model.cuda()

    # dataset
    instances, ntc_instance = convert_cabocha_format_to_training_instances(
        file_path=args.parsed_text,
        model_type=args.model_type,
    )
    logger.info("Number of instances: {}".format(len(instances)))
    tokenizer = load_tokenizer()
    data_loader = PasBucketIterator(
        file_path=args.parsed_text,
        batch_size=args.batch_size,
        dataset=instances,
        padding_value=tokenizer.pad_token_id,
        model_type=args.model_type,
    )

    # decode
    results = decode_for_pas(model=model, data_loader=data_loader)

    # display
    print("\n".join("".join(sent) for sent in ntc_instance["sents"]))
    print()

    predicts = []
    for result, pas in zip(results, ntc_instance["pas_list"]):
        count = 0
        predict = {}
        for key in [PREDICATE, GA, WO, NI]:
            if key in result:
                sent_idx, word_idx = result[key]["sent"], result[key]["id"]

                if count == 0:
                    predict["sent_id"] = sent_idx
                    predict["word_ids"] = pas["prd_word_ids"]
                    predict["result"] = {}

                    assert sent_idx == pas["prd_sent_idx"], "The indexes of ntc instance and result are misaligned"
                    assert word_idx in pas["prd_word_ids"], "The indexes of ntc instance and result are misaligned"

                if sent_idx == -1:
                    answer = "私（一人称）"
                elif sent_idx == -2:
                    answer = "あなた（二人称）"
                elif sent_idx == -3:
                    answer = "一般（その他）"
                else:
                    answer = ntc_instance["sents"][sent_idx][word_idx]

                if key == "pred":
                    sent = ntc_instance["sents"][sent_idx]
                    tokens = ""
                    for w_id in predict["word_ids"]:
                        tokens += str(sent[w_id])
                    predict["result"][key] = tokens
                else:
                    predict["result"][key] = answer
                count += 1

                print("{}: {}".format(key, predict["result"][key]))

        predicts.append(predict)
        print()

    if args.out_file is not None:
        with open(args.out_file, "wb") as out_file:
            pickle.dump(predicts, out_file)

        with open(args.out_file, "rb") as file:
            data = pickle.load(file)
            print(data)

    logger.info("done")


if __name__ == "__main__":
    main()
