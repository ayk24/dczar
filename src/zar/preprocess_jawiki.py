import argparse
import re
from os import path
from typing import List


def preprocess(file_path: path.abspath) -> List[str]:
    """Preprocess jawiki file.

    Args:
        file_path (str): Path to jawiki file.
    Returns:
        jawiki_all_text (List[str]): Preprocessed text.
    Details:
        remove:
            - doc tag (e.g. <doc id= ... >, </doc>)
            - title (e.g. アンパサンド, 日本語)
            - file tag (e.g. [[File: .. ]], [[ファイル: .. ]])
        replace:
            - two or more EOS (e.g. \n\n -> \n)
            - [[, ]] (e.g. [[宗教]] -> 宗教)
    """

    pattern_doc_start_tag = r"<doc id=.+ url=.+ title=.+>"
    pattern_doc_end_tag = r"</doc>"
    pattern_file_tag = r"(\[\[File.+?\]\])|(\[\[ファイル.+?\]\])"
    pattern_brackets = r"(\[\[)|(\]\])"

    has_matched_doc_start = False
    has_matched_doc_title = False

    jawiki_all_text = []
    with open(file_path) as file:
        for line in file:
            line = line.rstrip()

            if (has_matched_doc_start is True) and (has_matched_doc_title is False):
                has_matched_doc_title = True
                print("delete: title")
                continue
            elif (has_matched_doc_start is True) and (has_matched_doc_title is True):
                if len(line) == 0:
                    print("delete: EOS")
                    continue
                else:
                    has_matched_doc_start = False
                    has_matched_doc_title = False

            # file tag (e.g. [[File: .. ]], [[ファイル: .. ]])
            if re.search(pattern_file_tag, line) is not None:
                print("delete: {}".format(re.search(pattern_file_tag, line).group()))
                continue
            # [[, ]] (e.g. [[宗教]] -> 宗教)
            elif re.search(pattern_brackets, line) is not None:
                jawiki_all_text.append(re.sub(pattern_brackets, "", line))
                print("replace: {} -> {}".format(line, re.sub(pattern_brackets, "", line)))
                continue
            # doc end tag (e.g. </doc>)
            elif re.search(pattern_doc_end_tag, line) is not None:
                print("delete: {}".format(re.search(pattern_doc_end_tag, line).group()))
                continue
            # doc start tag (e.g.  <doc id= ... >)
            elif re.search(pattern_doc_start_tag, line) is not None:
                has_matched_doc_start = True
                print("delete: {}".format(re.search(pattern_doc_start_tag, line).group()))
                continue

            jawiki_all_text.append(line)

    return jawiki_all_text


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocessing ja-wiki")
    parser.add_argument("--in", dest="in_file", type=path.abspath, required=True, help="Path to input file")
    parser.add_argument("--out", dest="out_file", type=path.abspath, required=True, help="Path to output file")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    preprocessed_text = preprocess(file_path=args.in_file)
    with open(args.out_file, mode="w") as file:
        file.write("\n".join(preprocessed_text))


if __name__ == "__main__":
    main()
