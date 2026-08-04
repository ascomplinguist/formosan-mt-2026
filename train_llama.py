def main():
    import gc
    import os

    # The model is already fully cached locally; skip the network HEAD-request freshness check that
    # from_pretrained otherwise does on every call (each retries 5x with backoff on timeout, wasting time)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import sys
    import json
    from datetime import datetime, timezone, timedelta
    import numpy as np
    import pandas as pd
    import torch
    import datasets
    import transformers
    import peft
    import trl

    from datasets import concatenate_datasets
    from peft import LoraConfig
    from transformers import ProgressCallback
    from trl import SFTConfig

    from bleu_eval import BleuEarlyStoppingCallback
    from bleu_eval_llama import (
        MySFTTrainerMulti,
        BleuEvalHelperLlama,
        load_base_model_llama,
        build_bidirectional_examples,
        add_prompt_columns,
    )
    from check_git_status import ensure_files_clean, get_current_commit_hash

    # Quick git reproducibility check (if not testing)
    CONFIG_PATH = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] != "testing") else "config.json"
    if not "testing" in sys.argv:
        ensure_files_clean(["train_llama.py", "bleu_eval.py", "bleu_eval_llama.py"])
    COMMIT_HASH = get_current_commit_hash()

    print(
        f"torch=={torch.__version__} "
        f"datasets=={datasets.__version__} "
        f"transformers=={transformers.__version__} "
        f"peft=={peft.__version__} "
        f"trl=={trl.__version__} "
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

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    MODEL_NAME = config["model_name"]
    LANGUAGE_NAME = config["lang"]  # Chinese-script language names used inside the prompts
    if isinstance(LANGUAGE_NAME, str):
        LANGUAGE_NAME = [LANGUAGE_NAME]
    LANGUAGE_NAME_EN = config["lang_en"]
    if isinstance(LANGUAGE_NAME_EN, str):
        LANGUAGE_NAME_EN = [LANGUAGE_NAME_EN]
    if len(LANGUAGE_NAME) != len(LANGUAGE_NAME_EN):
        raise RuntimeError(f"{CONFIG_PATH}: 'lang' and 'lang_en' must have the same length")
    DATA_DIR = config["data_dir"]

    OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "")

    data_set_name = config["data_set_name"]
    trainable_modules = config["trainable_modules"]
    lang_code = config["lang_code"]  # only for save_dir, can be concatenated codes for multi-language training
    learning_rate = config.get("learning_rate", 2e-4)
    max_epochs = config.get("max_epochs", 10)
    early_stopping = config.get("early_stopping", False)
    subset = config.get("subset", -1)
    EVAL_SAVE_STEPS = config.get("eval_save_steps", 500)
    per_device_train_batch_size = config.get("per_device_train_batch_size", 2)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    eval_batch_size = config.get("eval_batch_size", 8)

    INTERLEAVE_PROBS = config.get("interleave_probs", None) or [1] * len(LANGUAGE_NAME_EN)
    LANG_WEIGHTS = {lang: p/np.sum(INTERLEAVE_PROBS) for lang, p in zip(LANGUAGE_NAME_EN, INTERLEAVE_PROBS)}

    save_dir = f"{OUTPUT_PREFIX}{data_set_name}_{trainable_modules}_{lang_code}_llama"
    sub_dir = save_dir.split("/")[-1]
    version = ""
    if sub_dir in os.listdir(OUTPUT_PREFIX or "."):
        version = 2
        while f"{sub_dir}_v{version}" in os.listdir(OUTPUT_PREFIX or "."):
            version += 1
        save_dir += f"_v{version}"
    with open("last_run.json", "w+") as f:
        tz = timezone(timedelta(hours=8))  # UTC+08:00
        created_at = datetime.now(tz).isoformat()
        json.dump({
            "save_dir": save_dir,
            "created_at": created_at
        }, f, indent=2, ensure_ascii=False)

    train_dfs = []
    val_dfs = []
    for lang_en, lang_zh in zip(LANGUAGE_NAME_EN, LANGUAGE_NAME):
        train_df = pd.read_parquet(f"{DATA_DIR}/{lang_en}_train_v3.parquet")
        train_dfs.append(train_df)
        val_df = pd.read_parquet(f"{DATA_DIR}/{lang_en}_val_v3.parquet")
        val_df = add_prompt_columns(val_df, lang_zh)
        if subset > 0:
            val_df = val_df.iloc[:subset].copy()
        val_dfs.append(val_df)

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

    train_dss = []
    for train_df, lang_zh in zip(train_dfs, LANGUAGE_NAME):
        train_ds = build_bidirectional_examples(
            train_df,
            source_col="source",
            target_col="zh",
            language_name=lang_zh,
        )
        train_dss.append(train_ds)

    epoch_datasets = [
        make_epoch_dataset(train_dss, INTERLEAVE_PROBS, seed=42 + epoch)
        for epoch in range(max_epochs)
    ]
    train_ds = concatenate_datasets(epoch_datasets)

    val_ds = build_bidirectional_examples(
        val_dfs[0].iloc[:10],
        source_col="source",
        target_col="zh",
        language_name=LANGUAGE_NAME[0],
    )  # need a 'fake' val_ds to satisfy the trainer

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

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model, tokenizer = load_base_model_llama(MODEL_NAME)

    print(f"allocated: {torch.cuda.memory_allocated() / 2**30:.3f}GB")
    print(f"reserved: {torch.cuda.memory_reserved() / 2**30:.3f}GB")
    free, total = torch.cuda.mem_get_info()
    print(f"{free/2**30:.3f} of {total/2**30:.3f}GB free")

    module_dict = {
        'q': 'q_proj',
        'v': 'v_proj',
        'k': 'k_proj',
        'o': 'o_proj',
        'u': 'up_proj',
        'g': 'gate_proj',
        'd': 'down_proj'
    }
    target_modules = [module_dict[x] for x in trainable_modules]
    print(f"Trainable Modules: {target_modules}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_args = SFTConfig(
        output_dir=save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        report_to="none",
        dataloader_num_workers=0,
        max_seq_length=256,
    )

    model.config.use_cache = False

    bleu_helpers = []
    for val_df, lang_en in zip(val_dfs, LANGUAGE_NAME_EN):
        bleu_helper = BleuEvalHelperLlama(
            eval_df=val_df,
            tokenizer=tokenizer,
            score_weights={"eval_bleu_zh": 0.4, "eval_chrf_zuyu": 0.6},
            batch_size=eval_batch_size,
            eval_lang=lang_en,
        )
        bleu_helpers.append(bleu_helper)

    callbacks = []
    if early_stopping:
        callbacks.append(
            BleuEarlyStoppingCallback(
                patience=4,
                unacceptable_score_threshold=-0.03,
                verbose=False,
            )
        )

    trainer = MySFTTrainerMulti(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # only needed to make the trainer call evaluate(), which triggers our bleu_helpers
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
        bleu_helpers=bleu_helpers,
        lang_weights=LANG_WEIGHTS,  # for reweighting of different languages, ex.: {'Atayal': 0.9, 'Amis': 0.1}
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
