import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer

from bleu_eval import BleuEvalHelper

# torch.triu not implemented for bf16 in torch 1.13, so we're hacking it here
_original_torch_triu = torch.triu

def triu_with_bf16_fallback(input, diagonal=0, *, out=None):
    if input.is_cuda and input.dtype == torch.bfloat16:
        result = _original_torch_triu(
            input.float(),
            diagonal=diagonal,
        ).to(torch.bfloat16)

        if out is not None:
            out.copy_(result)
            return out

        return result

    return _original_torch_triu(
        input,
        diagonal=diagonal,
        out=out,
    )

torch.triu = triu_with_bf16_fallback




prompt_to_chinese_string = (
    "任務：機器翻譯。\n"
    "將下列 {language} 文字翻譯為繁體中文。\n"
    "輸出必須且只能包含翻譯結果。\n\n"
    "{sentence}"
)

prompt_to_zuyu_string = (
    "任務：機器翻譯。\n"
    "將下列繁體中文文字翻譯為 {language}。\n"
    "輸出必須且只能包含翻譯結果。\n\n"
    "{sentence}"
)


def build_bidirectional_examples(df, source_col, target_col, language_name):
    rows = []
    for _, r in df.iterrows():
        src = str(r[source_col]).strip()
        tgt = str(r[target_col]).strip()
        if not src or not tgt:
            continue

        prompt_zh = prompt_to_chinese_string.format(language=language_name, sentence=src)
        prompt_lang = prompt_to_zuyu_string.format(language=language_name, sentence=tgt)

        rows.append({
            "messages": [
                {"role": "user", "content": prompt_zh},
                {"role": "assistant", "content": tgt},
            ]
        })
        rows.append({
            "messages": [
                {"role": "user", "content": prompt_lang},
                {"role": "assistant", "content": src},
            ]
        })
    return Dataset.from_pandas(pd.DataFrame(rows))


def add_prompt_columns(val_df, language_name):
    """Add the single-message prompt columns used by BleuEvalHelperLlama to a val/test dataframe."""
    val_df["to_zh_msg"] = val_df.apply(
        lambda row: {"role": "user", "content": prompt_to_chinese_string.format(language=language_name, sentence=row["source"])},
        axis=1,
    )
    val_df["to_zuyu_msg"] = val_df.apply(
        lambda row: {"role": "user", "content": prompt_to_zuyu_string.format(language=language_name, sentence=row["zh"])},
        axis=1,
    )
    return val_df
    

def load_base_model_llama(model_name, adapter_path=None, torch_dtype=torch.bfloat16):
    """Load Llama in full precision (no quantization); optionally attach a trained LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    return model, tokenizer


class MySFTTrainerMulti(SFTTrainer):
    """Multi-language variant of bleu_eval.MySFTTrainer, mirroring MySeq2SeqTrainer's
    per-language scoring and lang_weights aggregation (no embedding tracking — the
    Llama runs train no token embeddings)."""

    def __init__(
        self,
        *args,
        bleu_helpers: list[BleuEvalHelper] | None = None,
        lang_weights: dict[str, float] | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bleu_helpers = bleu_helpers
        langs = [b.get_eval_lang() for b in self.bleu_helpers]
        self.lang_weights = lang_weights or {l: 1/len(langs) for l in langs}

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        extra_metrics = dict()
        if self.bleu_helpers is not None:
            for bleu_helper in self.bleu_helpers:
                scores = bleu_helper.compute_metrics(self.model, self.state)
                if len(self.bleu_helpers) > 1:
                    lang = bleu_helper.get_eval_lang()
                    scores = {f"{k}-{lang}": v for k, v in scores.items()}
                extra_metrics.update(scores)
                metrics.update(scores)
            # If multiple helpers, also compute weighted averages
            if len(self.bleu_helpers) > 1:
                keys = set(k.split("-")[0] for k in extra_metrics.keys()
                           if isinstance(extra_metrics[k], float) or isinstance(extra_metrics[k], int))
                for key in keys:
                    mean_val = np.sum([v * self.lang_weights.get(k.split("-")[1], 0) for k, v in extra_metrics.items() if k.split("-")[0] == key])
                    extra_metrics[key] = mean_val
                    metrics[key] = mean_val

        self.log(extra_metrics)

        # Needs to be triggered once more (was already triggered in super().evaluate()) for BleuEarlyStoppingCallback
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class BleuEvalHelperLlama(BleuEvalHelper):
    def __init__(self, *args, max_new_tokens=200, batch_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def generate_column(self, prompt_col, desc, model):
        terminators = [
            t for t in [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ] if t is not None
        ]

        messages = [[m] for m in self.eval_df[prompt_col]]
        outputs = []

        # Batched generation needs left padding so all prompts end at the same position
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            with tqdm(total=len(messages), desc=desc, leave=False) as pbar:
                verbose = True
                for i in range(0, len(messages), self.batch_size):
                    batch = messages[i:i + self.batch_size]
                    inputs = self.tokenizer.apply_chat_template(
                        batch,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            repetition_penalty=1.1,
                            eos_token_id=terminators,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=True,  # model.config.use_cache is False during training
                            top_p=None,
                            temperature=None,
                        )

                    if verbose:
                        print(tokenizer.decode(out[0]))
                        verbose = False
                    
                    prompt_len = inputs["input_ids"].shape[1]
                    outputs.extend(
                        self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                        for seq in out
                    )
                    pbar.update(len(batch))
        finally:
            self.tokenizer.padding_side = original_padding_side

        return outputs
