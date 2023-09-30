import argparse
from os import path

import torch
import yaml
from evaluating import compute_pas_f_score, generate_pas_evaluation_instance
from iterators import PasBucketIterator
from training import PAS_MODEL_BASENAME, AsPzeroModel, FinetuningArgs

LOG_DIR_BASENAME = "./outputs/logs/zar/"
LOGZERO_LOG_FILE = "evaluating.logzero.txt"


def create_arg_parser():
    """Create arg parser"""
    parser = argparse.ArgumentParser(description="Decoding")
    parser.add_argument("--yaml_file", type=path.abspath, required=True, help="Path to yaml file")

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    params_dict = yaml.safe_load(open(args.yaml_file))
    params = FinetuningArgs(**params_dict)
    params.set_additional_parameters()

    model_file = path.join(params.output_dir, PAS_MODEL_BASENAME)
    if not path.exists(model_file):
        raise FileNotFoundError("not found: {}".format(model_file))

    test_data_loader = PasBucketIterator(
        file_path=params.test_data_file,
        batch_size=params.per_gpu_eval_batch_size,
        n_max_tokens=params.per_gpu_eval_max_tokens,
        padding_value=params.pad_token_id,
        model_type=params.model_type,
    )

    model = AsPzeroModel()
    model.load_state_dict(torch.load(model_file))

    device = torch.device("cuda" if torch.cuda.is_available() and not params.no_cuda else "cpu")
    model.to(device)

    instances = generate_pas_evaluation_instance(model, test_data_loader)
    _, _ = compute_pas_f_score(instances)


if __name__ == "__main__":
    main()
