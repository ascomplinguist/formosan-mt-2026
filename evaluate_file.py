import sys
from bleu_eval import BleuEvalHelperFromFile
from check_git_status import ensure_files_clean, get_current_commit_hash

ensure_files_clean(["eval_nllb.py", "bleu_eval.py"])
COMMIT_HASH = get_current_commit_hash()

if len(sys.argv) < 2:
    print("Usage: python3 evaluate_file.py path/to/file1 lang1 path/to/file2 lang2 ...")
    raise ValueError("No file given")

eval_files = sys.argv[1::2]
checkpoints = [eval_file.split("/")[:-1] for eval_file in eval_files]
eval_langs = sys.argv[2::2]

bleu_helpers = [
    BleuEvalHelperFromFile(
        eval_df = None,
        tokenizer = None,
        eval_bootstrap = True,
        eval_file = eval_file,
    )
    for eval_file in eval_files
]

for bleu_helper, checkpoint, lang in zip(bleu_helpers, checkpoints, eval_langs):
    scores = bleu_helper.compute_metrics(model=None, return_detail=True)

    print(scores)

    scores["commit"] = COMMIT_HASH

    with open(f"{checkpoint}/eval_scores_{lang}.json", "w+") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


        #TODO NOT TESTED!!
