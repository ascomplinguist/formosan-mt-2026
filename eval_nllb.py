def main():
    import gc
    import os
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

    from bleu_eval import BleuEvalHelperNLLB
    # Quick git reproducibility check
    from check_git_status import ensure_files_clean, get_current_commit_hash
    ensure_files_clean(["eval_nllb.py", "bleu_eval.py"])
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

    EVAL_CONFIG = sys.argv[1] if len(sys.argv) > 1 else "eval_config.json"

    with open(EVAL_CONFIG, "r") as f:
        eval_config = json.load(f)
    if "last_run.json" in os.listdir():
        with open("last_run.json", "r") as f:
            LAST_RUN_SAVE_DIR = json.load(f).get("save_dir")

    FINETUNE_DIR = eval_config.get('finetune_dir', None) or LAST_RUN_SAVE_DIR
    CHECKPOINT = f"{FINETUNE_DIR}/{eval_config['checkpoint']}"
    LANGUAGE_NAME_EN = eval_config["eval_lang_en"]  # languages to evaluate on
    LANGUAGE_TOK = eval_config["language_tok"]  # tokens for eval languages

    print(f"Evaluating finetune '{CHECKPOINT}' on {LANGUAGE_NAME_EN} using tokens {LANGUAGE_TOK}")

    with open(f"{FINETUNE_DIR}/config.json", "r") as f:
        config = json.load(f)

    MODEL_NAME = config["model_name"]
    DATA_DIR = config["data_dir"]

    # NEW: language codes for NLLB
    # Example:
    #   src_lang_code = "tay_Latn"   # needs to be supported/added
    #   zh_lang_code  = "zho_Hant"
    SRC_LANG_CODE = config["src_lang_code"]
    if isinstance(SRC_LANG_CODE, str):
        SRC_LANG_CODE = [SRC_LANG_CODE]
    ZH_LANG_CODE = config["zh_lang_code"]
    INIT_LANGS = config["init_langs"]  # ["ceb_Latn", "ilo_Latn", "pag_Latn", "tgl_Latn", "war_Latn"]

    val_dfs = []
    for lang in LANGUAGE_NAME_EN:
        val_df = pd.read_parquet(f"{DATA_DIR}/{lang}_val_v3.parquet")
        val_df["to_zuyu_msg"] = val_df["zh"]
        val_df["to_zh_msg"] = val_df["source"]
        val_dfs.append(val_df)

    #bnb_config = BitsAndBytesConfig(load_in_8bit=True)  # If quantisation needed, comment back in

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    #model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    '''
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # loading base tokeniser and updating after

    # Check source tags exist
    new_lang_tok_id = tokenizer.convert_tokens_to_ids(SRC_LANG_CODE)
    if any([n is None or n == tokenizer.unk_token_id for n in new_lang_tok_id]):
        print(f"Initialising new lang token for {SRC_LANG_CODE}")
        for lang in INIT_LANGS:
            tok_id = tokenizer.convert_tokens_to_ids(lang)
            print(lang, tok_id)
            if tok_id is None or tok_id == tokenizer.unk_token_id:
                raise ValueError(f"Initializer language tag not found: {lang}")
        
        # Add new language tag
        num_toks = len(tokenizer)
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokenizer.additional_special_tokens + SRC_LANG_CODE}
        )
        
        print(f"num_added: {num_added}, delta: {len(tokenizer) - num_toks}")
        
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        new_id = tokenizer.convert_tokens_to_ids(SRC_LANG_CODE)
        init_ids = tokenizer.convert_tokens_to_ids(INIT_LANGS)
        
        print("new_id:", new_id)
        print("init_ids:", init_ids)
        
        # Initialize shared embeddings
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[new_id] = emb[init_ids].mean(dim=0)
        
            # M2M100/NLLB usually ties lm_head to shared embeddings, but handle both cases
            out_emb = model.get_output_embeddings()
            if out_emb is not None and out_emb.weight is not emb:
                print("Output embeddings not tied, initialising separately")
                out_emb.weight[new_id] = out_emb.weight[init_ids].mean(dim=0)
            else:
                print("Output embeddings tied, no need for separate training/init")
    '''
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        CHECKPOINT,
    )

    model.eval()

    # Re-tie lm_head and embeddings
    model.base_model.model.lm_head.weight = model.get_input_embeddings().weight

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    max_source_length = 256
    max_target_length = 256

    for val_df, lang_tok, lang in zip(val_dfs, LANGUAGE_TOK, LANGUAGE_NAME_EN):
        print(f"Using '{lang_tok}' for '{lang}'")
        bleu_helper = BleuEvalHelperNLLB(
            eval_df=val_df,
            tokenizer=tokenizer,
            src_lang_code=lang_tok,
            zh_lang_code=ZH_LANG_CODE,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            score_weights={"eval_bleu_zh": 0.4, "eval_chrf_zuyu": 0.6},
            batch_size=16,
            save_dir=CHECKPOINT,
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

        with open(f"{CHECKPOINT}/eval_scores_{lang}.json", "w+") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
