from sacrebleu import BLEU, CHRF
import re
import os
import time
from transformers import TrainerCallback, Seq2SeqTrainer, AutoModelForSeq2SeqLM
import numpy as np
import pandas as pd
import torch
from trl import SFTTrainer
from tqdm.auto import tqdm

def make_orthogonal_noises(shape, target_l2, seed=0):
    if len(shape) == 2:
        n, dim = shape
    elif len(shape) == 1:
        n = 1
        dim = shape[0]
    else:
        raise ValueError(f"Invalid shape for noise generation: {shape}")
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, dim, generator=g)
    # QR decomposition gives orthogonal Q and upper-triangular R such that QR = X
    Q, _ = torch.linalg.qr(X.T)  # dim x n
    noises = Q.T * target_l2
    return noises

def load_base_model_nllb(
    model_name: str, 
    src_lang_code: list(str),                       # lang tokens for new added languages
    init_langs: list(str),                          # lang tokens used for mean init of new lang token embeddings
    init_vecs: str,                                 # filename for vecs for init of new lang token embeddings
    init_noise: float | int,                        # target l2 distance for random deviation of mean init
    init_noise_seed,                                # seed for random init of new lang token embeddings
    output_prefix : str = "",                       # directory prefix for output of randomly generated init vecs
    trainable_modules : str | None = None,          # modules to apply LoRA on (for training only)
    adapter_path : str | None = None,               # path to load adapter from (for eval only)
) -> tuple(PeftModel, AutoTokenizer):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    #model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    

    # Check source tags exist
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

        # If exactly 2 tokens were added, save_embeddings will not be triggered because NLLB has two extra embedding rows -> add extra token
        if num_added == 2:
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": tokenizer.additional_special_tokens + ["token_that_absolutely_doesnt_exist123"]}
            )
            if num_added != 1:
                raise RuntimeError(f"Somehow the extra special 'waste token' was not added properly: {num_added=}")
        
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        new_id = tokenizer.convert_tokens_to_ids(src_lang_code)
        
        print("new_id:", new_id)
        
        # Initialize shared embeddings
        with torch.no_grad():
            emb = model.get_input_embeddings().weight

            if not init_vecs:
                if init_noise > 0:
                    noises = make_orthogonal_noises(emb[new_id].shape, init_noise, seed=init_noise_seed)
                else:
                    noises = torch.zeros_like(emb[new_id])
    
                noises = noises.to(device=emb.device, dtype=emb.dtype)

                init_ids = tokenizer.convert_tokens_to_ids(init_langs)
                emb[new_id] = emb[init_ids].mean(dim=0) + noises

                print("init_ids:", init_ids)

                vector_dict = dict()
                vector_dict['vecs'] = {lang: emb[tok_id].detach().cpu().clone() 
                                       for lang, tok_id in zip(src_lang_code, new_id)}
                vector_dict['meta'] = {"target_l2": init_noise, "seed": init_noise_seed}
                torch.save(vector_dict, f"{output_prefix}vectors/{save_dir.split('/')[-1]}.pt")
            else:
                vector_dict = torch.load(init_vecs, map_location="cpu")
                for lang, tok_id in zip(src_lang_code, new_id):
                    emb[tok_id] = vector_dict['vecs'][lang].to(device=emb.device, dtype=emb.dtype)
                
            # M2M100/NLLB usually lm_head to shared embeddings, make sure this is the case
            assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(f"allocated: {torch.cuda.memory_allocated() / 2**30:.3f}GB")
    print(f"reserved: {torch.cuda.memory_reserved() / 2**30:.3f}GB")
    free, total = torch.cuda.mem_get_info()
    print(f"{free/2**30:.3f} of {total/2**30:.3f}GB free")

    if trainable_modules is not None:
        module_dict = {
            "q": "q_proj",
            "v": "v_proj",
            "k": "k_proj",
            "o": "out_proj",
            "u": "fc1",
            "d": "fc2",
        }
        target_modules = [module_dict[x] for x in trainable_modules]
        print(f"Trainable Modules: {target_modules}")
        if len(trainable_modules) == 0:
            target_modules = ["q_proj"]  # we'll set requires_grad to False later but peft won't behave if we give it nothing
    
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=target_modules,
            modules_to_save=["shared"]
        )
    
        model = get_peft_model(model, peft_config)
        if len(trainable_modules) == 0:  # if we only train embeddings, we need to freeze fake LoRA adapters
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.requires_grad = False
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

    # Re-tie lm_head and embeddings
    model.base_model.model.lm_head.weight = model.get_input_embeddings().weight
    
    return model, tokenizer


bleu_zh = BLEU(tokenize="zh")
bleu_form = BLEU(tokenize="13a")
chrf = CHRF(
    char_order=6,
    word_order=2,
    beta=2,
    lowercase=True,
)

def normalise_char_and_punct_variants(df, columns):
    """
    Standardises common Taiwan variant characters and punctuation in specified DataFrame columns.
    """
    # Dictionary of variants to standard MoE (Ministry of Education) forms
    # Standardising to the 'orthodox' or most common 'standard' forms
    variant_map = {
        "台": "臺",
        "衞": "衛",
        "羣": "群",
        "峯": "峰",
        "裏": "裡",
        "鷄": "雞",
        "恒": "恆",
        "銹": "鏽",
        "床": "牀",
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "「": '"',
        "」": '"',
        "ʼ": "'",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": ","
    }

    # We use regex=True to replace characters within the strings
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(variant_map, regex=True)
        else:
            raise ValueError(f"column {col} is not in dataframe columns")

    return df

def remove_whitespace(df, columns):
    """Remove whitespace from the specified DataFrame columns if the whitespace is preceded or succeeded by non-Atayal characters"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: re.sub(r"\s+(?![A-Za-z'\s])", "", x) if isinstance(x, str) else x)
            df[col] = df[col].apply(lambda x: re.sub(r"(?<![A-Za-z'\s])\s+", "", x) if isinstance(x, str) else x)
        else:
            raise ValueError(f"column {col} is not in dataframe columns")

    return df


def get_bleu_and_chrf(df, hyp_col, ref_col, bleu=bleu_zh, verbose=False, normalise=True, outfile_prefix=""):
    """Computes BLEU and CHRF for references in ref_col and hypotheses in hyp_col.

    Returns bare scores as well as sacrebleu string for BLEU (BLEU, chrF2++, BLEU detail).

    bleu: sacrebleu.BLEU object to use; default uses "zh" tokenizer, supply "13a" or appropriate if text is not Mandarin
    normalise: whether to apply normalisation of variant characters (and whitespace if text is Mandarin)
    outfile_prefix: if at least one character long, output references to {outfile_prefix}_ref.txt and hypotheses to {outfile_prefix}_hyp.txt
    """
    if verbose:
        print(f"Got {len(df)} sentences")
    if normalise:
        df = normalise_char_and_punct_variants(df, [ref_col, hyp_col])
    if type(bleu.tokenizer).__name__ == "TokenizerZh" and normalise:
        if verbose:
            print("Detected ZH, removing whitespace")
        df = remove_whitespace(df, [ref_col, hyp_col])

    if len(outfile_prefix) > 0:
        with open(f"{outfile_prefix}_ref.txt", "w+") as f:
            f.write("\n".join([l.strip() for l in list(df[ref_col])]))
        with open(f"{outfile_prefix}_hyp.txt", "w+") as f:
            f.write("\n".join([l.strip() for l in list(df[hyp_col])]))
    
    b = bleu.corpus_score(list(df[hyp_col]), [list(df[ref_col])])
    c = chrf.corpus_score(list(df[hyp_col]), [list(df[ref_col])])
    if verbose:
        print(b)
        print(c)
    bleu_score = b.score
    chrf_score = c.score
    return bleu_score, chrf_score, str(b)


def score_file(fname, hyp_col="", ref_col="", bleu=bleu_zh, fname2="", normalise=True, outfile_prefix=""):
    if len(fname2) == 0:
        df = pd.read_csv(fname, sep="\t")
    else:
        with open(fname, "r") as f:
            lines1 = f.readlines()
            lines1 = [l.strip() for l in lines1]
        with open(fname2, "r") as f:
            lines2 = f.readlines()
            lines2 = [l.strip() for l in lines2]
        print(f"refs from {fname}, hyps from {fname2}")
        df = pd.DataFrame(zip(lines1, lines2), columns=[ref_col, hyp_col])
    get_bleu_and_chrf(df, hyp_col, ref_col, bleu=bleu, verbose=True, normalise=normalise, outfile_prefix=outfile_prefix)
        

def send_msg(row, prompt_col, tokenizer, peft_model, silent=False, start=None, interval=30, total_items=1, pbar=None):

    i = row.name
    message = row[prompt_col]
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    inputs = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}
    
    
    
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            top_p=None,
            temperature=None
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    if pbar is not None:
        pbar.update(1)
    elif (i + 1) % interval == 0 and not silent:
        if start is None:
            raise ValueError("start time has to be provided if silent=False")
        now = time.time()
        elapsed = now - start
        percent = (i + 1)/total_items
        total_time = elapsed / percent
        estimated_time = total_time - elapsed
        print(f"Finished processing {i + 1}/{total_items} (elapsed: {elapsed:.2f}s, remaining: {estimated_time:.2f}s, total: {total_time:.2f}s)")
    return response


def generate_translation_nllb(text, src_lang, tgt_lang, model, tokenizer, max_source_length=256, max_target_length=256, verbose=False):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_source_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    model.generation_config.max_new_tokens = max_target_length
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.1
    model.generation_config.forced_bos_token_id = forced_bos_token_id

    with torch.no_grad():
        outputs = model.generate(**inputs)

    if verbose:
        print(tokenizer.decode(outputs[0]))

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]



class BleuEvalHelper:
    def __init__(self, eval_df, tokenizer, score_weights=None, save_dir=None):
        self.eval_df = eval_df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.score_weights = score_weights if score_weights is not None else {
            "eval_bleu_zh": 0.4,
            "eval_chrf_zuyu": 0.6,
        }
        self.save_dir = save_dir

    def generate_column(self, prompt_col, desc, model):
        outputs = []
        with tqdm(total=len(self.eval_df), desc=desc, leave=False) as pbar:
            for _, row in self.eval_df.iterrows():
                outputs.append(send_msg(row, prompt_col, self.tokenizer, model, pbar=pbar))
        return outputs

    def save_eval_results(self):
        i = 1
        while f"eval{i}.tsv" in os.listdir(self.save_dir):
            i += 1
        self.eval_df.to_csv(f"{self.save_dir}/eval{i}.tsv", sep="\t")

    def compute_metrics(self, model, state=None, return_detail=False):
        # state could be used to extract step if I want to store intermediate predictions
        
        was_training = model.training
        model.eval()

        self.eval_df["to_zh"] = self.generate_column("to_zh_msg", "To Mandarin", model)
        self.eval_df["to_zuyu"] = self.generate_column("to_zuyu_msg", "To Zuyu", model)

        #print(self.eval_df[["to_zh", "zh", "to_zuyu", "source"]].head(5))
        if self.save_dir is not None:
            self.save_eval_results()

        scores = dict()
        scores['eval_bleu_zh'], scores['eval_chrf_zh'], eval_bleu_zh_detail = get_bleu_and_chrf(self.eval_df, 
                                                                                                'to_zh', 'zh', 
                                                                                                bleu=bleu_zh)
        scores['eval_bleu_zuyu'], scores['eval_chrf_zuyu'], eval_bleu_zuyu_detail = get_bleu_and_chrf(self.eval_df, 
                                                                                                      'to_zuyu', 'source', 
                                                                                                      bleu=bleu_form)
        # compute scores before potentially adding strings to the dict
        scores['eval_score'] = np.sum([self.score_weights.get(k, 0) * scores[k] for k in scores])
        if return_detail:
            scores['eval_bleu_zh_detail'] = eval_bleu_zh_detail
            scores['eval_bleu_zuyu_detail'] = eval_bleu_zuyu_detail

        if was_training:
            model.train()

        return scores


class BleuEvalHelperNLLB(BleuEvalHelper):
    def __init__(self, eval_df, tokenizer, src_lang_code, zh_lang_code, 
                 max_source_length=256, max_target_length=256, batch_size=8, score_weights=None, save_dir=None):
        super().__init__(eval_df, tokenizer, score_weights=score_weights, save_dir=save_dir)
        self.src_lang_code = src_lang_code
        self.zh_lang_code = zh_lang_code
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size

    def get_src_lang_code(self):
        return self.src_lang_code

    def generate_column(self, prompt_col, desc, model):
        # Figure out which direction to translate
        if "to_zh" in prompt_col:
            src_lang = self.src_lang_code
            tgt_lang = self.zh_lang_code
        else:
            src_lang = self.zh_lang_code
            tgt_lang = self.src_lang_code

        print_example = 1
            
        outputs = []
        with tqdm(total=len(self.eval_df), desc=desc, leave=False) as pbar:
            batch = []
            for _, row in self.eval_df.iterrows():
                batch.append(row[prompt_col])
                if len(batch) == self.batch_size:
                    out = generate_translation_nllb(
                        text=batch,
                        src_lang=src_lang,  # NOT NECESSARILY == self.src_lang_code; variable set above
                        tgt_lang=tgt_lang,
                        model=model,
                        tokenizer=self.tokenizer,
                        max_source_length=self.max_source_length,
                        max_target_length=self.max_target_length,
                        verbose=print_example > 0
                    )
                    print_example -= 1
                    outputs.extend(out)
                    pbar.update(self.batch_size)
                    batch = []
            if len(batch) > 0:
                out = generate_translation_nllb(
                    text=batch,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    model=model,
                    tokenizer=self.tokenizer,
                    max_source_length=self.max_source_length,
                    max_target_length=self.max_target_length
                )
                outputs.extend(out)
                pbar.update(len(batch))
        return outputs


class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, bleu_helper: BleuEvalHelper = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bleu_helper = bleu_helper

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        extra_metrics = dict()
        if self.bleu_helper is not None:
            extra_metrics = self.bleu_helper.compute_metrics(self.model, self.state)
            metrics.update(extra_metrics)

        self.log(extra_metrics)

        # Callback is already called in super().evaluate() but not with our metrics, so we need to call it again
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        return metrics


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self, 
        *args, 
        bleu_helpers : list(BleuEvalHelperNLLB) | None = None, 
        lang_weights : dict(str, float) | None = None, 
        tracked_tok_embs : list(str) = [], 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bleu_helpers = bleu_helpers
        langs = [b.get_src_lang_code().split("_")[0] for b in self.bleu_helpers]
        self.lang_weights = lang_weights or {l: 1/len(langs) for l in langs}
        self.tracked_tok_embs = tracked_tok_embs
        
        self.tracked_tok_emb_ids = self.tokenizer.convert_tokens_to_ids(self.tracked_tok_embs)
        self.tracked_embs_init = model.get_input_embeddings()[self.tracked_tok_emb_ids].detach().cpu().clone()
        self.tracked_embs_last = self.tracked_embs_init.clone()

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        extra_metrics = dict()
        if self.bleu_helpers is not None:
            for bleu_helper in self.bleu_helpers:
                scores = bleu_helper.compute_metrics(self.model, self.state)
                if len(self.bleu_helpers) > 1:
                    lang = bleu_helper.get_src_lang_code().split("_")[0]  # remove script label
                    scores = {f"{k}-{lang}": v for k, v in scores.items()}
                extra_metrics.update(scores)
                metrics.update(scores)
            # If multiple helpers, also compute weighted averages
            if len(self.bleu_helpers) > 1:
                keys = set(k.split("-")[0] for k in extra_metrics.keys())
                for key in keys:
                    mean_val = np.sum([v * self.lang_weights.get(k.split("-")[1], 0) for k, v in extra_metrics.items() if k.split("-")[0] == key])
                    extra_metrics[key] = mean_val
                    metrics[key] = mean_val

        current_tok_embs = model.get_input_embeddings()[self.tracked_tok_emb_ids]
        extra_metrics.update({
            f"{tok_emb}-dist_to_init": (current_tok_embs[i] - self.tracked_embs_init[i]).norm() for i, tok_emb in self.tracked_tok_embs
        })
        extra_metrics.update({
            f"{tok_emb}-dist_to_last": (current_tok_embs[i] - self.tracked_embs_last[i]).norm() for i, tok_emb in self.tracked_tok_embs
        })
        extra_metrics.update({
            f"{tok_emb}-first_six": current_tok_embs[i][:6] for i, tok_emb in self.tracked_tok_embs
        })
        self.log(extra_metrics)

        # Needs to be triggered once more (was already triggered in super().evaluate()) for BleuEarlyStoppingCallback
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    
class BleuEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        patience=2,
        unacceptable_score_threshold=-0.01,
        verbose=False,
    ):
        """
        patience: number of bad steps that triggers early termination
        unacceptable_score_threshold: relative change of score that increases bad_steps by 1
        """
        self.patience = patience
        self.unacceptable_score_threshold = unacceptable_score_threshold
        self.verbose = verbose
        self.best_score = 0
        self.bad_steps = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            if self.verbose:
                print(f"No metrics received on step {state.global_step}")
            return control
        if "eval_score" not in metrics:
            if self.verbose:
                print(f"Metrics received on step {state.global_step}: {metrics.keys()}")
            return control
        if self.verbose:
            print(f"Valid metrics received on step {state.global_step}")
   
        if self.verbose:
            print(f"step {state.global_step}: " + ", ".join([f"{k}: {metrics[k]:.3f}" 
                                                             for k in metrics if k in 
                                                             ['eval_bleu_zh', 'eval_chrf_zh', 'eval_bleu_zuyu', 'eval_chrf_zuyu', 'eval_score']]))

        score = metrics["eval_score"]
        
        rel_score = score / (self.best_score + 1e-3) - 1
        self.best_score = score if rel_score > 0 else self.best_score
            
        if rel_score > 0:
            self.bad_steps = 0
            if self.verbose:
                print("New best, bad_steps = 0")
        elif rel_score < self.unacceptable_score_threshold:
            self.bad_steps += 1
            if self.verbose:
                print(f"Bad step, bad_steps = {self.bad_steps}")
        else:
            if self.verbose:
                print(f"Acceptable step, bad_steps = {self.bad_steps}")
            pass  # small change, don't modify self.bad_steps
            
        if self.bad_steps >= self.patience:
            if self.verbose:
                print(f"Training should stop, bad_steps = {self.bad_steps}")
            control.should_training_stop = True
        
        return control