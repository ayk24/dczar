# Pretraining
Fill in the contents of [yaml files](../src/zar/args) before executing.
- Cloze task
    ```python
    $ poetry run python src/zar/training.py \
      --yaml_file [path to yaml file]
    ```
    - Example of `yaml file`: [src/zar/args/pretraining_cloze_jawiki_params.yml](../src/zar/args/pretraining_cloze_jawiki_params.yml)

- Pzero task
    ```python
    $ poetry run python src/zar/training.py \
      --yaml_file [path to yaml file]
    ```
    - Example of `yaml file`: [src/zar/args/pretraining_pzero_jawiki_params.yml](../src/zar/args/pretraining_pzero_jawiki_params.yml)
