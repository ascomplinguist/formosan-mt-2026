def main():
    import gc
    import os
    import sys
    import json
    import random
    from datetime import datetime, timezone, timedelta
    import numpy as np
    import pandas as pd
    import torch
    import datasets
    import transformers
    import peft
    import trl
    import bitsandbytes as bnb

    from datasets import Dataset, concatenate_datasets
    from tqdm.auto import tqdm
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        ProgressCallback,
    )

    from bleu_eval import (
        MySeq2SeqTrainer,
        BleuEvalHelperNLLB,
        BleuEarlyStoppingCallback,
        load_base_model_nllb,
    )
    from lm_head import NLLBEmbeddingGradientCallback
    from check_git_status import ensure_files_clean, get_current_commit_hash

    # Quick git reproducibility check (if not testing)
    CONFIG_PATH = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] != "testing") else "config.json"
    if not "testing" in sys.argv:
        ensure_files_clean(["train_nllb.py", "bleu_eval.py", "lm_head.py"])
    COMMIT_HASH = get_current_commit_hash()

    print(
        f"torch=={torch.__version__} "
        f"datasets=={datasets.__version__} "
        f"transformers=={transformers.__version__} "
        f"peft=={peft.__version__} "
        f"trl=={trl.__version__} "
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

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    MODEL_NAME = config["model_name"]
    LANGUAGE_NAME_EN = config["lang_en"]
    if isinstance(LANGUAGE_NAME_EN, str):
        LANGUAGE_NAME_EN = [LANGUAGE_NAME_EN]
    DATA_DIR = config["data_dir"]

    OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "")

    data_set_name = config["data_set_name"]
    trainable_modules = config["trainable_modules"]
    lang_code = config["lang_code"]  # only for save_dir, can be concatenated codes for multi-language training
    learning_rate = config.get("learning_rate", 2e-4)
    oversample_zuyu = config.get("oversample_zuyu", 1)
    subsample_zh = config.get("subsample_zh", 1)
    max_epochs = config.get("max_epochs", 10)
    early_stopping = config.get("early_stopping", False)
    subset = config.get("subset", -1)
    EVAL_SAVE_STEPS = config.get("eval_save_steps", 500)

    # language codes for NLLB
    SRC_LANG_CODE = config["src_lang_code"]
    if isinstance(SRC_LANG_CODE, str):
        SRC_LANG_CODE = [SRC_LANG_CODE]
    INTERLEAVE_PROBS = config.get("interleave_probs", None) or [1] * len(train_dss)
    LANG_WEIGHTS = {lang.split("_")[0]: p/np.sum(INTERLEAVE_PROBS) for lang, p in zip(SRC_LANG_CODE, INTERLEAVE_PROBS)}
    ZH_LANG_CODE = config["zh_lang_code"]
    TRAIN_ZH_HANT = config.get("train_zh_hant", True)
    INIT_LANGS = config.get("init_langs", None)  # ["ceb_Latn", "ilo_Latn", "pag_Latn", "tgl_Latn", "war_Latn"]
    INIT_VECS = config.get("init_vecs", "")
    if INIT_LANGS:
        init_lang_short = "".join([l[0] for l in INIT_LANGS])
        if INIT_VECS:
            raise RuntimeError(f"{CONFIG_PATH} specifies both 'init_langs' and 'init_vecs'; choose one")
    else:
        init_lang_short = "vecload"
        if not INIT_VECS:
            raise RuntimeError(f"{CONFIG_PATH} specifies neither 'init_langs' or 'init_vecs'; one is required")

    INIT_NOISE = config.get("init_noise", 0)
    INIT_NOISE_SEED = config.get("init_noise_seed", 42)

    save_dir = f"{OUTPUT_PREFIX}{data_set_name}_{trainable_modules}_{init_lang_short}_{lang_code}_nllb"
    sub_dir = save_dir.split("/")[-1]
    version = ""
    if sub_dir in os.listdir(OUTPUT_PREFIX):
        version = 2
        while f"{sub_dir}_v{version}" in os.listdir(OUTPUT_PREFIX):
            version += 1
        save_dir += f"_v{version}"
    with open("last_run.json", "w+") as f:
        tz = timezone(timedelta(hours=8))  # UTC+08:00
        created_at = datetime.now(tz).isoformat()
        json.dump({
            "save_dir": save_dir,
            "created_at": created_at
        }, f, indent=2, ensure_ascii=False)
    #if input(f"save dir is '{save_dir}', please confirm by typing 'YES'") != "YES":
    #    raise ValueError()

    train_dfs = []
    val_dfs = []
    for lang in LANGUAGE_NAME_EN:
        train_df = pd.read_parquet(f"{DATA_DIR}/{lang}_train_v3.parquet")
        train_dfs.append(train_df)
        val_df = pd.read_parquet(f"{DATA_DIR}/{lang}_val_v3.parquet")
        val_df["to_zuyu_msg"] = val_df["zh"]
        val_df["to_zh_msg"] = val_df["source"]
        if subset > 0:
            val_df = val_df.iloc[:subset].copy()
        val_dfs.append(val_df)

    # Build one row per direction, but now plain seq2seq pairs instead of chat messages
    def build_bidirectional_examples(df, lang):
        rows = []
        for _, r in df.iterrows():
            src = str(r["source"]).strip()
            zh = str(r["zh"]).strip()
            if not src or not zh:
                continue
            if random.random() <= subsample_zh:  # not well supported, same examples sampled every epoch
                rows.append({
                    "src_text": src,
                    "tgt_text": zh,
                    "src_lang": lang,
                    "tgt_lang": ZH_LANG_CODE,
                    "direction": "to_zh",
                })
            for _ in range(oversample_zuyu):
                rows.append({
                    "src_text": zh,
                    "tgt_text": src,
                    "src_lang": ZH_LANG_CODE,
                    "tgt_lang": lang,
                    "direction": "to_zuyu",
                })
        return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

    def sample_n_examples(ds, n, seed):
        """
        Sample n examples from a Hugging Face Dataset.

        If n <= len(ds), this samples without replacement.
        If n > len(ds), this samples with replacement by repeated shuffling.
        """
        if n <= 0:
            return None

        if n <= len(ds):
            return ds.shuffle(seed=seed).select(range(n))

        chunks = []
        remaining = n
        k = 0

        while remaining > 0:
            shuffled = ds.shuffle(seed=seed + k)
            take = min(remaining, len(ds))
            chunks.append(shuffled.select(range(take)))
            remaining -= take
            k += 1

        return concatenate_datasets(chunks)


    def make_epoch_dataset(train_sets, interleave_probs, seed=42):
        """
        Build one controlled 'target-language epoch'.

        train_dss[0] is treated as the target-language dataset.
        interleave_probs are ratios, not necessarily normalized.

        Example:
            len(train_dss[0]) = 4800
            interleave_probs = [0.9, 0.1]

        Then:
            dataset 0 contributes 4800 examples
            dataset 1 contributes round(4800 * 0.1 / 0.9) = 533 examples
        """
        if len(train_sets) != len(interleave_probs):
            raise ValueError("train_dss and interleave_probs must have the same length.")

        if len(train_sets) == 0:
            raise ValueError("train_dss must contain at least one dataset.")

        target_prob = interleave_probs[0]

        if target_prob <= 0:
            raise ValueError("The first probability/ratio must be > 0, because it anchors the epoch size.")

        target_n = len(train_sets[0])

        epoch_parts = []

        for i, (ds, prob) in enumerate(zip(train_sets, interleave_probs)):
            if prob < 0:
                raise ValueError("Ratios/probabilities must be non-negative.")

            if i == 0:
                # Target language: exactly one full pass.
                part = ds.shuffle(seed=seed + i)
            else:
                # Auxiliary languages: sampled relative to target exposure.
                n_i = round(target_n * prob / target_prob)
                part = sample_n_examples(ds, n_i, seed=seed + i)

            if part is not None:
                epoch_parts.append(part)

        return concatenate_datasets(epoch_parts).shuffle(seed=seed)

    random.seed(42)
    train_dss = []
    for train_df, lang in zip(train_dfs, SRC_LANG_CODE):
        train_ds = build_bidirectional_examples(train_df, lang)
        train_dss.append(train_ds)

    epoch_datasets = [
        make_epoch_dataset(train_dss, INTERLEAVE_PROBS, seed=42 + epoch)
        for epoch in range(max_epochs)
    ]
    train_ds = concatenate_datasets(epoch_datasets)

    val_ds = build_bidirectional_examples(val_dfs[0].iloc[:10], SRC_LANG_CODE[0])  # need a 'fake' val_ds to satisfy the trainer


    try:
        del trainer
    except NameError:
        pass
    try:
        del model
    except NameError:
        pass
    try:
        del tokenizer
    except NameError:
        pass

    gc.collect()
    torch.cuda.empty_cache()

    print(f"allocated: {torch.cuda.memory_allocated() / 2**30:.3f}GB")
    print(f"reserved: {torch.cuda.memory_reserved() / 2**30:.3f}GB")
    free, total = torch.cuda.mem_get_info()
    print(f"{free/2**30:.3f} of {total/2**30:.3f}GB free (after basic clean-up)")

    model, tokenizer = load_base_model_nllb(MODEL_NAME, SRC_LANG_CODE, INIT_LANGS, INIT_VECS, INIT_NOISE, INIT_NOISE_SEED, OUTPUT_PREFIX,
                                            trainable_modules=trainable_modules)
    
    # Freeze most embeddings to prevent drift of underrepresented tokens
    src_ids = tokenizer.convert_tokens_to_ids(SRC_LANG_CODE)
    zho_id = [tokenizer.convert_tokens_to_ids(ZH_LANG_CODE)] if TRAIN_ZH_HANT else []
    special_ids = []  # [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id]

    trainable_indices = torch.tensor(src_ids + zho_id + special_ids).to(model.device)

    callbacks = [
        NLLBEmbeddingGradientCallback(
            model.get_input_embeddings().weight.shape[0],
            torch.bfloat16,
            model.device,
            trainable_indices,
        )
    ]

    if early_stopping:
        callbacks.append(
            BleuEarlyStoppingCallback(
                patience=4,
                unacceptable_score_threshold=-0.03,
                verbose=False,
            )
        )

    max_source_length = 256
    max_target_length = 256

    def preprocess_function(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for src_text, tgt_text, src_lang, tgt_lang in zip(
            examples["src_text"],
            examples["tgt_text"],
            examples["src_lang"],
            examples["tgt_lang"],
        ):
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang

            model_inputs = tokenizer(
                src_text,
                max_length=max_source_length,
                truncation=True,
            )

            labels = tokenizer(
                text_target=tgt_text,
                max_length=max_target_length,
                truncation=True,
            )

            model_inputs["labels"] = labels["input_ids"]

            input_ids_list.append(model_inputs["input_ids"])
            attention_mask_list.append(model_inputs["attention_mask"])
            labels_list.append(model_inputs["labels"])

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    print("Starting dataset processing...")
    train_ds_tok = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds_tok = val_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    bleu_helpers = []
    tracked_tokens = [tokenizer.unk_token_id]
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        tracked_tokens.append(model.get_input_embeddings().weight.shape[0] - 1)
    for val_df, lang in zip(val_dfs, SRC_LANG_CODE):
        bleu_helper = BleuEvalHelperNLLB(
            eval_df=val_df,
            tokenizer=tokenizer,
            src_lang_code=lang,
            zh_lang_code=ZH_LANG_CODE,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            score_weights={"eval_bleu_zh": 0.4, "eval_chrf_zuyu": 0.6},
            tracked_tokens=tracked_tokens
        )
        bleu_helpers.append(bleu_helper)

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=EVAL_SAVE_STEPS,
        learning_rate=learning_rate,
        num_train_epochs=1,  # real number is achieved by sampling examples 'max_epochs' times
        logging_steps=5,
        save_strategy="steps",
        save_steps=EVAL_SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_score",
        greater_is_better=True,
        bf16=True,
        fp16=False,
        predict_with_generate=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,  # only needed to make the trainer call evaluate(), which triggers our bleu_helpers
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        bleu_helpers=bleu_helpers,
        lang_weights=LANG_WEIGHTS,  # for reweighting of different languages, ex.: {'tay': 0.8, 'pwn': 0.1, 'ami': 0.1}
        tracked_tok_embs=SRC_LANG_CODE + [ZH_LANG_CODE, "eng_Latn"],  # tracking eng_Latn just to make sure it doesn't move
    )

    # Suppress log prints 
    class SilentProgressCallback(ProgressCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if 'eval_score' in logs:
                print({k: v for k, v in logs.items() if any([k.startswith(x) for x in ['eval_score', 'eval_bleu_zh', 'eval_chrf_zuyu']])})


    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(SilentProgressCallback())

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

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted, saving current model...")
        trainer.save_model(f"{save_dir}/interrupted")
        tokenizer.save_pretrained(f"{save_dir}/interrupted")
        trainer.save_state()
        config["commit"] = COMMIT_HASH
        with open(f"{save_dir}/config.json", "w+") as f:
            json.dump(config, f)
        raise KeyboardInterrupt

    trainer.save_model(f"{save_dir}/final")
    tokenizer.save_pretrained(f"{save_dir}/final")
    trainer.save_state()
    config["commit"] = COMMIT_HASH
    with open(f"{save_dir}/config.json", "w+") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
