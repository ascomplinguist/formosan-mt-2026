def main():
    import gc
    import os

    # The model is already fully cached locally; skip the network HEAD-request freshness check that
    # from_pretrained otherwise does on every call (each retries 5x with backoff on timeout, wasting time)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import sys
    import json
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    import peft
    import bitsandbytes as bnb

    from tqdm.auto import tqdm
    from peft import PeftModel
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
    )

    from bleu_eval_madlad import BleuEvalHelperMADLAD, load_base_model_madlad
    from check_git_status import ensure_files_clean, get_current_commit_hash

    # Quick git reproducibility check (if not testing)
    EVAL_CONFIG = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] != "testing") else "eval_config.json"
    if not "testing" in sys.argv:
        ensure_files_clean(["eval_madlad.py", "bleu_eval.py", "bleu_eval_madlad.py"])
    COMMIT_HASH = get_current_commit_hash()

    print(
        f"torch=={torch.__version__} "
        f"transformers=={transformers.__version__} "
        f"peft=={peft.__version__} "
        f"numpy=={np.__version__} "
        f"bitsandbytes=={bnb.__version__}"
    )

    print(torch.__version__, "is the correct version" if torch.__version__ == "1.13.1+cu116" else "is not the correct version")

    free, total = torch.cuda.mem_get_info()
    print(f"{free/2**30:.3f} of {total/2**30:.3f} GB free")

    if free / total < 0.9:
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"{free/2**30:.3f} of {total/2**30:.3f} GB free (after basic clean-up)")

    if free / total < 0.9:
        print("You should have a look what's occupying all your VRAM")

    with open(EVAL_CONFIG, "r") as f:
        eval_config = json.load(f)
    if "last_run.json" in os.listdir():
        with open("last_run.json", "r") as f:
            LAST_RUN_SAVE_DIR = json.load(f).get("save_dir")

    FINETUNE_DIR = eval_config.get('finetune_dir', None) or LAST_RUN_SAVE_DIR
    CHECKPOINT = f"{FINETUNE_DIR}/{eval_config['checkpoint']}"
    LANGUAGE_NAME_EN = eval_config["eval_lang_en"]  # languages to evaluate on
    LANGUAGE_TOK = eval_config["language_tok"]  # MADLAD-400 target-language tags for eval languages, e.g. "<2ami>"

    print(f"Evaluating finetune '{CHECKPOINT}' on {LANGUAGE_NAME_EN} using tokens {LANGUAGE_TOK}")

    with open(f"{FINETUNE_DIR}/config.json", "r") as f:
        config = json.load(f)

    MODEL_NAME = config["model_name"]
    DATA_DIR = config["data_dir"]
    INIT_LANGS = config.get("init_langs", None)  # ["<2ceb>", "<2ilo>", "<2pag>", "<2fil>", "<2war>"]
    INIT_VECS = config.get("init_vecs", "")
    if INIT_LANGS:
        if INIT_VECS:
            raise RuntimeError(f"config.json of {FINETUNE_DIR} specifies both 'init_langs' and 'init_vecs'; choose one")
    else:
        if not INIT_VECS:
            raise RuntimeError(f"config.json of {FINETUNE_DIR} specifies neither 'init_langs' or 'init_vecs'; one is required")

    INIT_NOISE = config.get("init_noise", 0)
    INIT_NOISE_SEED = config.get("init_noise_seed", 42)

    if INIT_NOISE > 0:
        DATA_SET_NAME = config["data_set_name"]
        OUTPUT_PREFIX, run_id = FINETUNE_DIR.split(DATA_SET_NAME)
        run_id = DATA_SET_NAME + run_id
        INIT_VECS = f"{OUTPUT_PREFIX}vectors/{run_id}.pt"
        print(f"Extracting initialisation from `{INIT_VECS}`")
        INIT_LANGS = None

    # MADLAD-400 language tags, e.g. "<2ami>", "<2zh_Hant>" - special tokens
    SRC_LANG_CODE = config["src_lang_code"]
    if isinstance(SRC_LANG_CODE, str):
        SRC_LANG_CODE = [SRC_LANG_CODE]
    ZH_LANG_CODE = config["zh_lang_code"]

    val_dfs = []
    USE_TEST_SET = eval_config.get("use_test_set", False)
    file_suffix = "test" if USE_TEST_SET else "val"
    for lang in LANGUAGE_NAME_EN:
        val_df = pd.read_parquet(f"{DATA_DIR}/{lang}_{file_suffix}_v3.parquet")
        val_df["to_zuyu_msg"] = val_df["zh"]
        val_df["to_zh_msg"] = val_df["source"]
        val_dfs.append(val_df)

    model, tokenizer = load_base_model_madlad(MODEL_NAME, SRC_LANG_CODE, INIT_LANGS, INIT_VECS, INIT_NOISE, INIT_NOISE_SEED,
                                              adapter_path=CHECKPOINT)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    max_source_length = 256
    max_target_length = 256

    for val_df, lang_tok, lang in zip(val_dfs, LANGUAGE_TOK, LANGUAGE_NAME_EN):
        print(f"Using '{lang_tok}' for '{lang}'")
        bleu_helper = BleuEvalHelperMADLAD(
            src_lang_code=lang_tok,
            zh_lang_code=ZH_LANG_CODE,
            eval_df=val_df,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            score_weights={"eval_bleu_zh": 0.4, "eval_chrf_zuyu": 0.6},
            batch_size=16,
            save_dir=CHECKPOINT,
            eval_bootstrap=True,
            save_file_prefix="test" if USE_TEST_SET else "eval"
        )

        print(f"allocated: {torch.cuda.memory_allocated() / 2**30:.3f}GB")
        print(f"reserved: {torch.cuda.memory_reserved() / 2**30:.3f}GB")
        free, total = torch.cuda.mem_get_info()
        print(f"{free/2**30:.3f} of {total/2**30:.3f}GB free")

        gc.collect()
        torch.cuda.empty_cache()

        print(f"allocated: {torch.cuda.memory_allocated() / 2**30:.3f}GB")
        print(f"reserved: {torch.cuda.memory_reserved() / 2**30:.3f}GB")
        free, total = torch.cuda.mem_get_info()
        print(f"{free/2**30:.3f} of {total/2**30:.3f}GB free")

        scores = bleu_helper.compute_metrics(model, return_detail=True)

        print(scores)

        scores["commit"] = COMMIT_HASH

        file_prefix = "test" if USE_TEST_SET else "eval"
        with open(f"{CHECKPOINT}/{file_prefix}_scores_{lang}.json", "w+") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
