import argparse
import re
from os import path
from typing import Dict, List

import CaboCha


def extract_predicates(sents: List[str], cabocharc: path.abspath, mecabrc: path.abspath) -> Dict[int, str]:
    """Extract predicates.

    Args:
        sents (List[str]): List of sentences.
        cabocharc (path.abspath): Path to cabocharc.
        mecabrc (path.abspath): Path to mecabrc.
    Returns:
        Dict[int, str]: Dictionary of predicates.
    """
    cabocha = CaboCha.Parser("-f1 -O4 -n0 -r " + str(cabocharc) + " -b " + str(mecabrc))

    lines = {}
    morphs = {}
    idx = 0
    for sent in sents:
        tree = cabocha.parse(sent)
        parsed_texts = tree.toString(CaboCha.FORMAT_LATTICE).split("\n")

        for parsed_text in parsed_texts:
            morph = re.split("[\s,]", parsed_text)
            if parsed_text:
                morphs[idx] = morph
                lines[idx] = parsed_text
                idx += 1

    pred_id = 1
    skip_num = 0
    for i in range(len(lines)):
        if skip_num > 0:
            skip_num -= 1
        else:
            if not morphs[i][0] == "*" and not morphs[i][0] == "EOS":
                # サ変名詞 + サ変動詞 (+ 動詞 or 形容詞 or 形容動詞 or 接尾辞)
                if morphs[i][2] == "サ変名詞":
                    # サ変名詞 -> EOS
                    if morphs[i + 1][0] == "EOS":
                        lines[i] += "\tO"
                        continue

                    # サ変名詞 -> サ変動詞
                    if morphs[i + 1][2] == "サ変動詞" or morphs[i + 1][3] == "サ変動詞":
                        lines[i] += "\tO\t" + str(pred_id)
                        lines[i + 1] += "\tO\t" + str(pred_id)
                        skip_num += 1
                    else:
                        lines[i] += "\tO"
                        continue

                    for j in range(len(lines)):
                        now_idx = i + 2 + j
                        if morphs[now_idx][0] == "EOS":
                            skip_num += 1
                            pred_id += 1
                            break

                        if (
                            morphs[now_idx][1] == "動詞"
                            or morphs[now_idx][1] == "形容詞"
                            or morphs[now_idx][1] == "形容動詞"
                            or morphs[now_idx][2] == "動詞性接尾辞"
                            or morphs[now_idx][2] == "形容詞性名詞接尾辞"
                            or morphs[now_idx][2] == "形容詞性述語接尾辞"
                        ):
                            lines[now_idx] += "\tO\t" + str(pred_id)
                            skip_num += 1
                        else:
                            pred_id += 1
                            break

                # 動詞 or 形容詞 or 形容動詞 (+ 動詞 or 形容詞 or 形容動詞 or 接尾辞)
                elif morphs[i][1] == "動詞" or morphs[i][1] == "形容詞" or morphs[i][1] == "形容動詞":
                    lines[i] += "\tO\t" + str(pred_id)

                    for j in range(len(lines)):
                        now_idx = i + 1 + j
                        if morphs[now_idx][0] == "EOS":
                            skip_num += 1
                            pred_id += 1
                            break

                        if (
                            morphs[now_idx][1] == "動詞"
                            or morphs[now_idx][1] == "形容詞"
                            or morphs[now_idx][1] == "形容動詞"
                            or morphs[now_idx][2] == "動詞性接尾辞"
                            or morphs[now_idx][2] == "形容詞性名詞接尾辞"
                            or morphs[now_idx][2] == "形容詞性述語接尾辞"
                        ):
                            lines[now_idx] += "\tO\t" + str(pred_id)
                            skip_num += 1
                        else:
                            pred_id += 1
                            break

                else:
                    lines[i] += "\tO"

    return lines


def create_parser() -> argparse.ArgumentParser:
    """Create parser."""
    parser = argparse.ArgumentParser(description="Extracting Predicates")
    parser.add_argument("--in", dest="in_file", type=path.abspath, required=True, help="Path to input file.")
    parser.add_argument("--out", dest="out_file", type=path.abspath, required=True, help="Path to output file.")
    parser.add_argument("--cabocharc", type=path.abspath, required=True, help="Path to cabocharc.")
    parser.add_argument("--mecabrc", type=path.abspath, required=True, help="Path to mecabrc.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    sentences = []
    with open(args.in_file) as in_file:
        for sentence in in_file:
            sentences.append(sentence)

    lines = extract_predicates(sentences, args.cabocharc, args.mecabrc)
    with open(args.out_file, "w", encoding="utf-8") as out_file:
        for key, value in lines.items():
            out_file.write(value + "\n")


if __name__ == "__main__":
    main()
