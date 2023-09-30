import argparse
import glob
import json
import os
import re
from os import path

import emoji


def preprocess(in_dir: path.abspath, out_dir: path.abspath):
    """Preprocess Twitter data.

    Args:
        in_dir (str): Path to input directory.
        out_dir (str): Path to output directory.
    Details:
        remove:
            - mention (e.g. @abc, @def)
            - URL (e.g. http://..., https://...)
            - hash tag (e.g. #yyy, #zzz)
            - emoji
            - RT (e.g. RT )
        replace:
            - two or more EOS (e.g. \n\n -> \n)
            - special characters (e.g. &amp; -> &)
        others:
            - Sort tweets from old -> new
    """
    pattern_mention = r"@([A-Za-z0-9_]+)"
    pattern_url = r"https?://[w/:%#$&?()~.=+-…]+"
    pattern_tag = r"#(\w+)"
    pattern_rt = r"RT "
    pattern_more_eos = r"\n+"
    special_chars_bf = ["&amp;", "&lt;", "&gt;", "&quot;", "&nbsp;"]
    special_chars_af = ["&", "<", ">", '"', " "]

    for user_name in os.listdir(in_dir):
        if os.path.isdir(in_dir + "/" + user_name):
            tweet_all_text = []
            for file in sorted(glob.glob(in_dir + "/" + user_name + "/" + "*.json")):
                with open(os.path.join(file), encoding="utf_8_sig") as f:
                    try:
                        tweets = json.load(f)
                    except json.JSONDecodeError as e:
                        print(str(e) + ": " + str(f))
                        continue

                    if "data" in tweets:
                        for tweet in tweets["data"]:
                            tweet_text = tweet["text"]

                            # mention (e.g. @abc, @def)
                            tweet_text = re.sub(pattern_mention, "", tweet_text)

                            # URL (e.g. http://..., https://...)
                            tweet_text = re.sub(pattern_url, "", tweet_text)

                            # hash tag (e.g. #yyy, #zzz)
                            tweet_text = re.sub(pattern_tag, "", tweet_text)

                            # RT (e.g. RT )
                            tweet_text = re.sub(pattern_rt, "", tweet_text)

                            # emoji
                            tweet_text = emoji.replace_emoji(tweet_text, replace="")

                            # 2 or more EOS
                            tweet_text = re.sub(pattern_more_eos, "\n", tweet_text)
                            tweet_text = tweet_text.strip()

                            # &amp;, &lt;, &gt;, &quot;, &nbsp;
                            for bf, af in zip(special_chars_bf, special_chars_af):
                                tweet_text = tweet_text.replace(bf, af)

                            tweet_all_text.append(tweet_text)

                        # Sort chronologically
                        tweet_all_text.reverse()

                        out_file_name = user_name + ".txt"
                        with open(os.path.join(out_dir, out_file_name), mode="w") as f:
                            f.write("\n".join(tweet_all_text))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocessing tweet")
    parser.add_argument("--in", dest="in_dir", type=path.abspath, required=True, help="Path to input directory")
    parser.add_argument("--out", dest="out_dir", type=path.abspath, required=True, help="Path to output directory")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    preprocess(in_dir=args.in_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
