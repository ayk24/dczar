# Preprocessing

## Wikipedia data
- Download [Wikipedia dump data](https://dumps.wikimedia.org/jawiki/latest/) and unzip it
    ```python
    $ poetry run python -m wikiextractor.WikiExtractor \
      [path to input (dump data)] \
      -o - --processes 8 \
      > [path to output (raw text)] \
    ```

- Run preprocessing (with `preprocess_jawiki.py`) scripts
    - remove
        - doc tag (e.g. `<doc id= ... >`, `</doc>`)
        - title (e.g. アンパサンド, 日本語)
        - file tag (e.g. [[File: .. ]], [[ファイル: .. ]])
    - replace
        - two or more newlines (e.g. \n\n -> \n)
        - [[, ]] (e.g. [[宗教]] -> 宗教)
    ```python
    $ poetry run python src/zar/preprocess_jawiki.py \
      --in [path to input (raw text)] \
      --out [path to output (text after preprocessed_jawiki)] \
    ```

- Run preprocessing (with `preprocess.py`) scripts
    ```python
    $ poetry run python src/zar/preprocess.py \
      --in [path to input (text after preprocessed_jawiki)] \
      --type "raw" \
      > [path to output (full-width text)]
    ```
    - Example of `full-width text`: [samples/jawiki.txt](../samples/jawiki.txt)

- Run preprocessing (with CaboCha) scripts
    ```sh
    $ poetry run sh src/zar/preprocess_with_cabocha.sh \
      [path to input (full-width text)] \
      > [path to output (text parsed with CaboCha)]
    ```
    - Example of `text parsed with CaboCha`: [samples/jawiki-parsed.txt](../samples/jawiki-parsed.txt)

## Twitter data
- Run preprocessing (with `preprocess_tweet.py`) scripts
    - remove
        - mention (e.g. @abc, @def)
        - URL (e.g. http://..., https://...)
        - hash tag (e.g. #yyy, #zzz)
        - RT (e.g. RT )
        - emoji
    - replace
        - two or more EOS (e.g. \n\n -> \n)
        - special characters (e.g. \&amp; -> &)
    ```python
    $ poetry run python src/zar/preprocess_tweet.py \
      --in [path to input (directory containing tweets)] \
      --out [path to output (directory containing preprocessed tweets)]
    ```

- Concat tweets
    ```sh
    $ cat [path to input (directory containing preprocessed tweets)]/*.txt \
      > [path to output (raw text)]
    ```

- Run preprocessing (with `preprocess.py`) scripts
    ```python
    $ poetry run python src/zar/preprocess.py \
      --in [path to input (raw text)] \
      --type "raw" \
      > [path to output (full-width text)]
    ```

- Run preprocessing (with CaboCha) scripts
    ```sh
    $ poetry run sh src/zar/preprocess_with_cabocha.sh \
      [path to input (full-width text)] \
      > [path to output (text parsed with CaboCha)]
    ```

## Create instances
### Cloze Task
- Create instances for training the model in Cloze Task
  ```python
  $ poetry run python src/zar/preprocess.py \
    --in [path to input (text parsed with CaboCha)] \
    --type "cloze" \
    > [path to output (file extension is `jsonl`)]
  ```
  - Example of `text parsed with CaboCha`: [samples/jawiki-parsed.txt](../samples/jawiki-parsed.txt)
  - Example of `output`: [samples/jawiki-cloze-instances.jsonl](../samples/jawiki-cloze-instances.jsonl)

### Pzero Task
- Create instances for training the model in Pzero Task
  ```python
  $ poetry run python src/preprocess.py \
    --in [path to input (text parsed with CaboCha)] \
    --type "pzero" \
    > [path to output (Pzero instances)]
  ```
  - Example of `text parsed with CaboCha`: [samples/jawiki-parsed.txt](../samples/jawiki-parsed.txt)
  - Example of `output`: [samples/jawiki-pzero-instances.jsonl](../samples/jawiki-pzero-instances.jsonl)

### AS-Pzero Task
- Create instances training the model in AZ-Pzero Task
  ```python
  $ poetry run python src/zar/preprocess.py
    --in [path to directory of NAIST Text Corpus]
    --type "as-pzero"
    > [path to outoput (AS-Pzero instances)]
  ```
