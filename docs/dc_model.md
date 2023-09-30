# DC Model
- Run preprocessing (with CaboCha) scripts and annotate with predicate information
    ```python
    $ poetry run python src/dialog/extract_predicates.py \
      --in [path to input (dialogue data)] \
      --out [path to output (dialogue data parsed with CaboCha)] \
      --cabocharc [path to cabocharc] \
      --mecabrc [path to mecabrc]
    ```
    - Example of `dialogue data`: [samples/dialogue.txt](../samples/dialogue.txt)
    - Example of `dialogue data parsed with CaboCha`: [samples/dialogue-parsed.txt](../samples/dialogue-parsed.txt)

- Predict predicate arguments in the dialogue data using PAS model
    ```python
    $ poetry run python src/zar/run.py \
      --parsed_text [path to dialogue data parsed with CaboCha] \
      --out [path to output (AS-Pzero instances)] \
      --model [path to PAS model] \
      --model_type "as-pzero"
    ```
    - Example of `dialogue data parsed with CaboCha`: [samples/dialogue-parsed.txt](../samples/dialogue-parsed.txt)
    - Example of `AS-Pzero instances`: [samples/dialogue-aspzero-instances.pkl](../samples/dialogue-aspzero-instances.pkl)
        ```
        [{'sent_id': 0, 'word_ids': [4, 5, 6], 'result': {'pred': '観てきました', 'ga': '私（一人称）', 'o': '映画'}}, {'sent_id': 1, 'word_ids': [0], 'result': {'pred': 'いい', 'ga': '映画'}}, {'sent_id': 2, 'word_ids': [2, 3], 'result': {'pred': '観ました', 'ga': '私（一人称）', 'o': 'アバター'}}, {'sent_id': 2, 'word_ids': [6], 'result': {'pred': '面白かった', 'ga': 'アバター'}}]
        ```
        - Details
            ```
            pred: 観てきました
            ga: 私（一人称）
            o: 映画

            pred: いい
            ga: 映画

            pred: 観ました
            ga: 私（一人称）
            o: アバター

            pred: 面白かった
            ga: アバター
            ```

- Complement the dialogue data with DC model
    ```python
    $ poetry run python src/dialogue/complement.py \
      --in [path to dialogue data] \
      --parsed_text [path to dialogue data parsed with CaboCha] \
      --preds [path to AS-Pzero instances] \
      --model [path to pretrained model] \
      --out [path to output (dialogue data complemented by omitted arguments)]
    ```
    - Example of `dialogue data`: [samples/dialogue.txt](../samples/dialogue.txt)
    - Example of `dialogue data parsed with CaboCha`: [samples/dialogue-parsed.txt](../samples/dialogue-parsed.txt)
    - Example of `AS-Pzero instances`: [samples/dialogue-aspzero-instances.pkl](../samples/dialogue-aspzero-instances.pkl)
    - Example of `dialogue data complemented by omitted arguments`: [samples/dialogue-complemented.txt](../samples/dialogue-complemented.txt)
        ```
        [私が]話題の映画を観てきました。
        [映画が]いいですね。何の映画ですか？
        アバターを観ました。[アバターが]とても面白かったです。
        ```
