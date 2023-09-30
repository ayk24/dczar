# RG Model
- Training
    ```python
    $ poetry run python src/dialogue/training.py \
    --yaml_file [path to yaml file]
    ```
    - Example of `yaml file`: [src/dialogue/args/finetuning_cloze_complement_jawiki_params.yml](../src/dialogue/args/finetuning_cloze_complement_jawiki_params.yml)

- Evaluating
    ```python
    $ poetry run python src/dialogue/evaluating.py \
    --yaml_file [path to yaml file]
    ```
    - Example of `yaml file`: [src/dialogue/args/evaluating_cloze_complement_jawiki_params.yml](../src/dialogue/args/evaluating_cloze_complement_jawiki_params.yml)
