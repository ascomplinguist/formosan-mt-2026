import torch
from tqdm.auto import tqdm
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from bleu_eval import BleuEvalHelper, make_orthogonal_noises


def load_base_model_madlad(
    model_name: str,
    src_lang_code: list[str],                       # MADLAD tags for newly added languages, e.g. ["<2ami>"]
    init_langs: list[str],                           # MADLAD tags used for mean init of new lang token embeddings, e.g. ["<2ceb>", "<2ilo>", "<2pag>", "<2fil>", "<2war>"]
    init_vecs: str,                                  # filename for vecs for init of new lang token embeddings
    init_noise: float | int,                         # target l2 distance for random deviation of mean init
    init_noise_seed,                                 # seed for random init of new lang token embeddings
    output_prefix : str = "",                        # directory prefix for output of randomly generated init vecs
    save_dir : str = "",                              # directory for output of randomly generated init vecs
    trainable_modules : str | None = None,            # non-None triggers LoRA training mode (actual target modules are fixed below)
    adapter_path : str | None = None,                 # path to load adapter from (for eval only)
) -> tuple[PeftModel, AutoTokenizer]:
    # Training needs the weights (and therefore the embedding lookup feeding T5's causal-mask builder) in fp32:
    # torch 1.13.1+cu116 has no CUDA kernel for torch.triu/tril on bfloat16, which that mask builder calls on a
    # full padded batch. bf16=True in Seq2SeqTrainingArguments still gives bf16 compute via autocast. Eval only
    # does incremental (KV-cached) generation, which doesn't hit that code path, so it can stay in bf16.
    torch_dtype = torch.float32 if trainable_modules is not None else torch.bfloat16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Check target tags exist
    new_lang_tok_id = tokenizer.convert_tokens_to_ids(src_lang_code)
    if any([n is None or n == tokenizer.unk_token_id for n in new_lang_tok_id]):
        print(f"Initialising new lang token for {src_lang_code}")
        if init_langs:
            for lang in init_langs:
                tok_id = tokenizer.convert_tokens_to_ids(lang)
                print(lang, tok_id)
                if tok_id is None or tok_id == tokenizer.unk_token_id:
                    raise ValueError(f"Initializer language tag not found: {lang}")

        # Add new language tag
        num_toks = len(tokenizer)
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokenizer.additional_special_tokens + src_lang_code}
        )

        print(f"num_added: {num_added}, delta: {len(tokenizer) - num_toks}")

        # Resize model embeddings. Unlike NLLB, MADLAD's tokenizer length and embedding matrix size already
        # match exactly (no reserved padding rows), so there's no ghost-row/legacy_load quirk to work around here.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        new_id = tokenizer.convert_tokens_to_ids(src_lang_code)

        print("new_id:", new_id)

        # Initialize input embeddings and, separately, the lm_head output embeddings - MADLAD (T5) does not tie
        # them (tie_word_embeddings=False), so both need their new rows initialised independently.
        with torch.no_grad():
            input_emb32 = model.get_input_embeddings().weight.detach().cpu().float()
            input_emb_ = model.get_input_embeddings().weight
            output_emb_ = model.get_output_embeddings().weight
            lang_token_id = tokenizer.convert_tokens_to_ids("<2zh_Hant>")
            output_emb_[new_id] = output_emb_[lang_token_id].clone()  # These are all the same for MADLAD because tags are never produced

            if not init_vecs:
                if init_noise > 0:
                    noises = make_orthogonal_noises(input_emb32[new_id].shape, init_noise, seed=init_noise_seed)
                else:
                    noises = torch.zeros_like(input_emb32[new_id])

                init_ids = tokenizer.convert_tokens_to_ids(init_langs)
                print("init_ids:", init_ids)

                emb_noises = input_emb32[init_ids].mean(dim=0) + noises
                input_emb_[new_id] = emb_noises.to(device=input_emb_.device, dtype=input_emb_.dtype)

                # save only in training, not in eval
                if trainable_modules is not None:
                    vector_dict = dict()
                    vector_dict['vecs'] = {lang: input_emb_[tok_id].detach().cpu().clone()
                                           for lang, tok_id in zip(src_lang_code, new_id)}
                    vector_dict['meta'] = {"target_l2": init_noise, "seed": init_noise_seed}
                    torch.save(vector_dict, f"{output_prefix}vectors/{save_dir.split('/')[-1]}.pt")
            else:
                vector_dict = torch.load(init_vecs, map_location="cpu")
                for lang, tok_id in zip(src_lang_code, new_id):
                    input_emb_[tok_id] = vector_dict['vecs'][lang].to(device=input_emb_.device, dtype=input_emb_.dtype)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(f"allocated: {torch.cuda.memory_allocated() / 2**30:.3f}GB")
    print(f"reserved: {torch.cuda.memory_reserved() / 2**30:.3f}GB")
    free, total = torch.cuda.mem_get_info()
    print(f"{free/2**30:.3f} of {total/2**30:.3f}GB free")

    if trainable_modules is not None:
        target_modules = ["q", "v", "o", "wi_0", "wi_1", "wo"]
        print(f"Trainable Modules: {target_modules}")

        # lm_head deliberately excluded: the new language tags only ever appear as source-side prefixes,
        # the decoder never has to predict them, so its (fixed-init) lm_head rows don't need training.
        # Leaving lm_head out of modules_to_save keeps it frozen with no duplicate copy and no optimizer
        # state - training only "shared" avoids ~5GB of otherwise-wasted memory.
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=target_modules,
            modules_to_save=["shared"],
        )

        model = get_peft_model(model, peft_config)
        # Needed for gradient checkpointing to work correctly with PEFT - otherwise the checkpointed
        # activations can end up without a grad_fn and backprop silently produces no gradients.
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    elif adapter_path is not None:
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

        model.eval()
    else:
        raise ValueError("Either `trainable_modules` or `adapter_path` has to be supplied")

    return model, tokenizer


def generate_translation_madlad(text, tgt_lang, model, tokenizer,
                                 max_source_length=256, max_target_length=256, verbose=False, tracker=None):
    '''Translates `text` (can be a list) into `tgt_lang` using `model` and `tokenizer`, following the MADLAD-400
    convention of prepending a "<2xx>" tag to the source sentence(s) to select the target language. The tag is
    plain text tokenized by the ordinary tokenizer, not a special/added token.

    `tracker` is a dict that is updated in place. It tracks how often certain tokens occur.
    Every tracked token is expected to already be a key in the dict, the key 'total' also needs to be present.
    The values should be initialised with [0], append a 0 before calling if you want to track values separately, e.g. between evaluations
    '''

    tagged_text = [f"{tgt_lang} {t}" for t in text] if isinstance(text, list) else f"{tgt_lang} {text}"

    inputs = tokenizer(tagged_text, return_tensors="pt", truncation=True, padding=True, max_length=max_source_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.generation_config.max_new_tokens = max_target_length
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.1

    with torch.no_grad():
        outputs = model.generate(**inputs)

    if tracker is not None:
        for interesting_token in tracker:
            if isinstance(interesting_token, int):
                token_id = interesting_token
            else:
                token_id = tokenizer.convert_tokens_to_ids(interesting_token)
            mask = (outputs == token_id)
            count = mask.sum().item()
            tracker[interesting_token][-1] += count
        pad_mask = outputs == tokenizer.pad_token_id
        pad_count = pad_mask.sum().item()
        total = outputs.numel() - pad_count
        tracker['total'][-1] += total

    if verbose:
        print(tokenizer.decode(outputs[0]))

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


class BleuEvalHelperMADLAD(BleuEvalHelper):
    def __init__(self, src_lang_code, zh_lang_code, *args,
                 max_source_length=256, max_target_length=256, batch_size=8, tracked_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.eval_lang is None:
            self.eval_lang = src_lang_code.removeprefix("<2").removesuffix(">")
        self.src_lang_code = src_lang_code
        self.zh_lang_code = zh_lang_code
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.tracked_tokens = (tracked_tokens or []) + ['total']
        self.current_tracker = None

    def get_src_lang_code(self):
        return self.src_lang_code

    def generate_column(self, prompt_col, desc, model):
        # Only the target-language tag matters for MADLAD; it gets prepended to the source text
        tgt_lang = self.zh_lang_code if "to_zh" in prompt_col else self.src_lang_code

        print_example = 1

        outputs = []
        tracker = self.get_init_tracker()
        with tqdm(total=len(self.eval_df), desc=desc, leave=False) as pbar:
            batch = []
            for _, row in self.eval_df.iterrows():
                batch.append(row[prompt_col])
                if len(batch) == self.batch_size:
                    out = generate_translation_madlad(
                        text=batch,
                        tgt_lang=tgt_lang,
                        model=model,
                        tokenizer=self.tokenizer,
                        max_source_length=self.max_source_length,
                        max_target_length=self.max_target_length,
                        verbose=print_example > 0,
                        tracker=tracker
                    )
                    print_example -= 1
                    outputs.extend(out)
                    pbar.update(self.batch_size)
                    batch = []
            if len(batch) > 0:
                out = generate_translation_madlad(
                    text=batch,
                    tgt_lang=tgt_lang,
                    model=model,
                    tokenizer=self.tokenizer,
                    max_source_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    tracker=tracker
                )
                outputs.extend(out)
                pbar.update(len(batch))
        self.current_tracker = tracker
        return outputs
