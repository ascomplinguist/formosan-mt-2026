import sys
import json
import pandas as pd
from bleu_eval import BleuEvalHelperFromFile
from check_git_status import ensure_files_clean, get_current_commit_hash
from convert_lang_code import get_langs, get_codes, LANG_NAMES, LANG_CODES

ensure_files_clean(["bleu_eval.py"])
COMMIT_HASH = get_current_commit_hash()

if len(sys.argv) < 2:
    print("Usage: python3 evaluate_file.py path/to/file1 path/to/file2 ...")
    raise ValueError("No file given")

eval_files = sys.argv[1:]
checkpoints = ["/".join(eval_file.split("/")[:-1]) for eval_file in eval_files]
langs = []
for eval_file in eval_files:
    filename = eval_file.split("/")[-1]
    detected_lang = None
    for lang_name in LANG_NAMES + LANG_CODES:
        if detected_lang is None and lang_name in filename:
            detected_lang = lang_name
            print(f"Found {lang_name} in {filename}")
        elif detected_lang is not None and lang_name in filename and len(lang_name) > len(detected_lang):
            print(f"Updating: {detected_name} -> {lang_name} in {filename}")
            detected_lang = lang_name
    if detected_lang is None:
        eval_df = pd.read_csv(eval_file, sep="\t")
        first_eval_source = eval_df['source'].iloc[0]
        last_eval_zh = eval_df['zh'].iloc[-1]
        eval_length = len(eval_df['source'])
        for lang_name in LANG_NAMES:
            try:
                df = pd.read_parquet(f"FormosanBench_new/{lang_name}_val_v3.parquet")
            except FileNotFoundError:
                print(f"No reference file found for '{lang_name}'")
                continue
            first_source = df['source'].iloc[0]
            last_zh = df['zh'].iloc[-1]
            length = len(df['source'])
            if first_source == first_eval_source and last_zh == last_eval_zh and length == eval_length:
                detected_lang = lang_name
                print(f"Matched eval file '{filename}' to {lang_name}")
                break
    if detected_lang is None:
        raise ValueError(f"Could not identify language of {filename}")
    langs.append(detected_lang)
        

bleu_helpers = [
    BleuEvalHelperFromFile(
        eval_bootstrap = True,
        eval_file = eval_file,
    )
    for eval_file in eval_files
]

for bleu_helper, checkpoint, lang in zip(bleu_helpers, checkpoints, langs):
    scores = bleu_helper.compute_metrics(model=None, return_detail=True)

    print(scores)

    scores["commit"] = COMMIT_HASH

    with open(f"{checkpoint}/eval_scores_{lang}.json", "w+") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
