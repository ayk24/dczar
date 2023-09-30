# PAS Model
- Training
    ```python
    $ poetry run python src/zar/training.py \
      --yaml_file [path to yaml file]
    ```
    - Example of `yaml file`: [src/zar/args/finetuning_cloze_as_pzero_jawiki_params.yml](../src/zar/args/finetuning_cloze_as_pzero_jawiki_params.yml)

- Evaluating
    ```python
    $ poetry run python src/zar/compute_score.py \
    --yaml_file [path to yaml file]
    ```
    - Example of `yaml file`: [src/zar/args/evaluating_cloze_as_pzero_jawiki_params.yml](../src/zar/args/evaluating_cloze_as_pzero_jawiki_params.yml)
