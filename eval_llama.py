def main():
    import gc
    import os

    # The model is already fully cached locally; skip the network HEAD-request freshness check that
    # from_pretrained otherwise does on every call (each retries 5x with backoff on timeout, wasting time)
    #os.environ.setdefault("HF_HUB_OFFLINE", "1")
    #os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import sys
    import json
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    import peft

    from bleu_eval_llama import BleuEvalHelperLlama, load_base_model_llama, add_prompt_columns
    from check_git_status import ensure_files_clean, get_current_commit_hash

    # Quick git reproducibility check (if not testing)
    EVAL_CONFIG = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] != "testing") else "eval_config.json"
    if not "testing" in sys.argv:
        ensure_files_clean(["eval_llama.py", "bleu_eval.py", "bleu_eval_llama.py"])
    COMMIT_HASH = get_current_commit_hash()

    print(
        f"torch=={torch.__version__} "
        f"transformers=={transformers.__version__} "
        f"peft=={peft.__version__} "
        f"numpy=={np.__version__}"
    )

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

    with open(f"{FINETUNE_DIR}/config.json", "r") as f:
        config = json.load(f)

    LANGUAGE_NAME_EN = eval_config.get("eval_lang_en", None) or config["lang_en"]  # languages to evaluate on
    if isinstance(LANGUAGE_NAME_EN, str):
        LANGUAGE_NAME_EN = [LANGUAGE_NAME_EN]

    MODEL_NAME = config["model_name"]
    DATA_DIR = config["data_dir"]

    # Chinese-script language names used inside the prompts (the Llama analogue of NLLB's 'language_tok');
    # defaults to the training language's name from the finetune config
    LANGUAGE_NAME_ZH = eval_config.get("eval_lang_zh", None) or config["lang"]
    if isinstance(LANGUAGE_NAME_ZH, str):
        LANGUAGE_NAME_ZH = [LANGUAGE_NAME_ZH]
    if len(LANGUAGE_NAME_ZH) != len(LANGUAGE_NAME_EN):
        raise RuntimeError("'eval_lang_zh' must have one prompt language name per entry in 'eval_lang_en'")

    print(f"Evaluating finetune '{CHECKPOINT}' on {LANGUAGE_NAME_EN} using prompt names {LANGUAGE_NAME_ZH}")

    val_dfs = []
    USE_TEST_SET = eval_config.get("use_test_set", False)
    file_suffix = "test" if USE_TEST_SET else "val"
    for lang, lang_zh in zip(LANGUAGE_NAME_EN, LANGUAGE_NAME_ZH):
        val_df = pd.read_parquet(f"{DATA_DIR}/{lang}_{file_suffix}_v3.parquet")
        val_df = add_prompt_columns(val_df, lang_zh)
        val_dfs.append(val_df)

    model, tokenizer = load_base_model_llama(MODEL_NAME, adapter_path=CHECKPOINT)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    for val_df, lang_zh, lang in zip(val_dfs, LANGUAGE_NAME_ZH, LANGUAGE_NAME_EN):
        print(f"Using '{lang_zh}' for '{lang}'")
        bleu_helper = BleuEvalHelperLlama(
            eval_df=val_df,
            tokenizer=tokenizer,
            score_weights={"eval_bleu_zh": 0.4, "eval_chrf_zuyu": 0.6},
            batch_size=16,
            save_dir=CHECKPOINT,
            eval_bootstrap=True,
            eval_lang=lang,
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
