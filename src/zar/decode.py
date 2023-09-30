import argparse
import json
import re
from collections import defaultdict
from os import path
from typing import Dict, List, Union

import torch
import yaml
from evaluating import generate_pas_evaluation_instance
from instances import FILE, PREDICATE, PasDecodeInstance
from iterators import PasBucketIterator
from logzero import logger
from models import AsModel, AsPzeroModel
from training import AS, AS_PZERO, PAS_MODEL_BASENAME, FinetuningArgs

LOG_DIR_BASENAME = "./outputs/logs/zar/"


def atoi(text: str) -> Union[int, str]:
    """Convert a string to an integer if possible

    Args:
        text (str): string
    Returns:
        int or str
    """
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List:
    """Sort strings with numbers

    Args:
        text (str): string
    Returns:
        List: sorted list
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def decode_for_pas(model: Union[AsModel, AsPzeroModel], data_loader: PasBucketIterator) -> List[Dict[str, PasDecodeInstance]]:
    """Decoding for evaluation using a test set

    Args:
        model (Union[AsModel, AsPzeroModel]): The model for decoding
        data_loader (PasBucketIterator): data loader
    Returns:
        results (List[Dict[str, PasDecodeInstance]]):
            ```
            results = [
                {
                    "file": path to each file for evaluation (str),
                    "pred": PasDecodeInstance(sent=sent_index, id=word_index),
                    "ga": PasDecodeInstance(sent=sent_index, id=word_index),
                    ...
                },
                ...
            ]
            ```
            where "pred" indicates target predicate
    """

    # In the final result, values with the same key are combined into a single dictionary.
    # key: '{file_name}-{predicate_sentence_index}-{predicate_word_index}'
    # value: prediction for each label
    decodes = defaultdict(dict)

    for instance in generate_pas_evaluation_instance(model, data_loader):
        zip_iter = zip(instance["predicts"], instance["exo_predicts"], instance["case_names"], instance["eval_infos"])
        for p_idx, exo_p_idx, case_name, eval_info in zip_iter:
            file_path = eval_info["file_path"]
            predicate_sent_idx = eval_info["prd_sent_idx"]
            predicate_word_idx = eval_info["prd_word_ids"][-1]

            key = f"{file_path}-{predicate_sent_idx}-{predicate_word_idx}"

            # 'file'
            if FILE in decodes[key]:
                assert decodes[key][FILE] == file_path
            else:
                decodes[key][FILE] = file_path

            # 'target predicate'
            if PREDICATE in decodes[key]:
                assert decodes[key][PREDICATE] == PasDecodeInstance(sent=predicate_sent_idx, id=predicate_word_idx)
            else:
                decodes[key][PREDICATE] = PasDecodeInstance(sent=predicate_sent_idx, id=predicate_word_idx)

            assert case_name not in decodes[key]

            # 'intra', 'inter'
            if p_idx != 0 and p_idx in eval_info["sw2w_position"]:
                sent_idx, word_idx = eval_info["sw2w_position"][p_idx]  # convert subword idx to sentence/word idx
                decodes[key][case_name] = PasDecodeInstance(sent=sent_idx, id=word_idx)

            # 'exophoric'
            elif p_idx == 0 and exo_p_idx != 0:
                decodes[key][case_name] = PasDecodeInstance(sent=-exo_p_idx, id=-1)  # EXO1 = -1, EXO2 = -2, EXOG = -3

    # sort values by keys and remove keys
    decode_results = [result for _, result in sorted(decodes.items(), key=lambda x: natural_keys(x[0]))]

    return decode_results


def create_arg_parser():
    """Create arg parser"""
    parser = argparse.ArgumentParser(description="Decoding")
    parser.add_argument("--yaml_file", type=path.abspath, required=True, help="Path to yaml file")

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # file check
    if not path.exists(args.yaml_file):
        raise FileNotFoundError("not found: {}".format(args.yaml_file))

    # load yaml file
    params_dict = yaml.safe_load(open(args.yaml_file))
    assert "model_type" in params_dict, "error: 'model_type' doesn't exist."
    params = FinetuningArgs(**params_dict)
    params.set_additional_parameters()

    # file check
    if not path.exists(params.output_dir):
        raise FileNotFoundError("not found: {}".format(params.output_dir))

    model_file = path.join(params.output_dir, PAS_MODEL_BASENAME)

    if not path.exists(model_file):
        raise FileNotFoundError("not found: {}".format(model_file))

    result_file = path.join(LOG_DIR_BASENAME, params.now_time, "result.txt")

    # load dataset
    eval_data_loader = PasBucketIterator(
        file_path=params.test_data_file,
        batch_size=params.per_gpu_eval_batch_size,
        n_max_tokens=params.per_gpu_eval_max_tokens,
        padding_value=params.pad_token_id,
        model_type=params.model_type,
    )

    # load model
    if params.model_type == AS:
        model = AsModel()
    elif params.model_type == AS_PZERO:
        model = AsPzeroModel()
    else:
        raise ValueError(f"unsupported value: {params.model_type}")
    model.load_state_dict(torch.load(model_file))
    if torch.cuda.is_available():
        model = model.cuda()

    # decode
    logger.info("Eval file: {}".format(args.data))
    logger.debug("Start decoding")
    model.eval()
    results = decode_for_pas(model=model, data_loader=eval_data_loader)
    logger.info("save to: {}".format(result_file))
    with open(result_file, "w") as fo:
        for result in results:
            print(json.dumps(result), file=fo)
    logger.info("done")


if __name__ == "__main__":
    main()
