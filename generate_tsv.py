#!/usr/bin/env python3
"""Generate finetune.tsv from tracked_work/."""

import csv
import datetime
import json
import os
import sys

import pandas as pd

TRACKED_WORK = "tracked_work"
CONTRIBUTORS = {"work": "", "joseph": "j/"}

MODEL_ABBREVIATIONS = {
    "facebook/nllb-200-3.3B": "NLLB-200-3.3B",
    "google/madlad400-3b-mt": "MADLAD400-3B-MT",
}

MODULE_DICT = {
    "q": "q_proj",
    "v": "v_proj",
    "k": "k_proj",
    "o": "out_proj",
    "u": "fc1",
    "d": "fc2",
}

# Standard Austronesian-language initialisation set used in all runs
INIT_LANGS_STD = ["ceb_Latn", "ilo_Latn", "pag_Latn", "tgl_Latn", "war_Latn"]
INIT_LANGS_STD_MADLAD = ["<2ceb>", "<2ilo>", "<2pag>", "<2fil>", "<2war>"]

# The 'qvougd' trainable_modules label is a fixed name; the actual MADLAD (T5) LoRA targets are hardwired
# in train_madlad.py and don't decode letter-by-letter through MODULE_DICT
MADLAD_LORA_MODULES = "q_proj, v_proj, o_proj, wi_0, wi_1, wo"

VAL_DESCRIPTIONS = {
    "v3":   "validation set, from cleaned FormosanBench v3 (normalised punctuation, removed other special characters, removed duplicates)",
    "v3.1": "validation set, from cleaned FormosanBench v3 (normalised punctuation, removed other special characters, removed duplicates) - splits generated on clusters of single-side duplicates, complete clusters retained in train, clusters reduced to a single random point in val and test",
}

# Maps directory names to the index of the language that should be listed first (primary).
# Only needed when config order doesn't match the intended display order.
MAIN_LANG_IDX = {
    "fb_new3_qvoud_ciptw_pwn-tay_nllb": 1,  # tay is main despite being second in config
}

# Manual overrides for training set name and model description when the config lacks
# enough detail to reconstruct the original description (e.g. interleave strategy not stored).
DESCRIPTION_OVERRIDES = {
    "fb_new3_qvoud_ciptw_pwn-tay_nllb": {
        "training_set": "FB train v3 (pwn+tay), interleaved 50/50",
        "model_description": (
            "NLLB-200-3.3B expanded with pwn_Latn, tay_Latn tokens, both initialised as the mean of"
            " [ceb_Latn, ilo_Latn, pag_Latn, tgl_Latn, war_Latn],"
            " with q_proj, v_proj, out_proj, fc1, fc2, and tokens pwn_Latn, tay_Latn, zho_Hant trainable,"
            " trained on interleaved pwn+tay data (50/50, 'all_exhausted'), trained for 30 epochs"
        ),
    },
}

HEADERS = [
    "base model", "modification", "language", "validation set", "# sentences (val)",
    "BLEU (to_zh)", "BLEU detail (to_zh)", "chrF2++ (to_zh)", "CI lo (to_zh)", "CI hi (to_zh)",
    "BLEU (to_zuyu)", "BLEU detail (to_zuyu)", "chrF2++ (to_zuyu)",
    "CI lo (to_zuyu)", "CI hi (to_zuyu)", "combined score",
    "BLEU (to_zh, test)", "chrF2++ (to_zh, test)", "CI lo (to_zh, test)", "CI hi (to_zh, test)",
    "BLEU (to_zuyu, test)", "chrF2++ (to_zuyu, test)", "CI lo (to_zuyu, test)", "CI hi (to_zuyu, test)",
    "training set", "description validation / test set", "description model (train set)",
]


def abbreviate_model(model_name):
    return MODEL_ABBREVIATIONS.get(model_name, model_name.split("/")[-1])


def data_set_version(data_set_name):
    # "fb_new3" -> "v3", "fb_new3" is the only one we're using
    return "v" + data_set_name.replace("fb_new", "")


def modification_from_dir(dir_name, data_set_name, prefix):
    # e.g. "fb_new3_qvoud_ciptw_ami_nllb_v2" -> "qvoud_ciptw_ami_v2"
    s = dir_name.removeprefix(data_set_name + "_")
    marker = "_madlad" if "_madlad" in s else "_nllb"
    parts = s.split(marker)
    mod = parts[0] + "".join(parts[1:])
    return prefix + mod


def per_lang_code(src_lang_code):
    # "ami_Latn" -> "ami"; MADLAD tag "<2ami>" -> "ami"
    if src_lang_code.startswith("<2"):
        return src_lang_code.removeprefix("<2").removesuffix(">")
    return src_lang_code.split("_")[0]


def val_version(final_dir, lang_code):
    # v3 eval files are for Amis, Paiwan, and Atayal; v3.1 ones are for other languages - slightly more rigorous data cleaning
    if lang_code not in ['ami', 'pwn', 'tay']:
        return "v3.1"
    return "v3"


def _load_ref_cache():
    """Build (first_source, last_zh, length) -> (lang_name, n_sentences) from parquet val files."""
    cache = {}
    from convert_lang_code import LANG_NAMES
    for lang_name in LANG_NAMES:
        path = os.path.join("FormosanBench_new", f"{lang_name}_val_v3.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            key = (df['source'].iloc[0], df['zh'].iloc[-1], len(df))
            cache[key] = (lang_name, len(df))
    return cache

_REF_CACHE = None

def count_sentences(final_dir, lang_en, lang_code):
    global _REF_CACHE
    import glob

    # Named file (monolingual runs); the numeric suffix is not fixed
    named = sorted(glob.glob(os.path.join(final_dir, f"eval_{lang_code}[0-9]*.tsv")))
    if named:
        with open(named[0]) as f:
            return sum(1 for _ in f) - 1

    # Content-based matching for unnamed eval files (multilingual runs)
    if _REF_CACHE is None:
        _REF_CACHE = _load_ref_cache()
    for eval_path in sorted(glob.glob(os.path.join(final_dir, "eval*.tsv"))):
        df = pd.read_csv(eval_path, sep="\t")
        key = (df['source'].iloc[0], df['zh'].iloc[-1], len(df))
        match = _REF_CACHE.get(key)
        if match and match[0] == lang_en:
            return match[1]

    return ""


def training_set_name(config, ds_version):
    lang_code = config["lang_code"].replace("-", "+")
    base = f"FB train {ds_version} ({lang_code})"
    probs = config.get("interleave_probs")
    if probs and len(probs) > 1:
        total = sum(probs)
        pct = [round(100 * p / total) for p in probs]
        return base + "; " + "/".join(str(p) for p in pct)
    return base


def model_description(config, base_model):
    src_lang_codes = config["src_lang_code"]
    if isinstance(src_lang_codes, str):
        src_lang_codes = [src_lang_codes]
    zh_lang_code = config["zh_lang_code"]
    init_langs = config.get("init_langs")
    init_vecs = config.get("init_vecs", "")
    lang_code = config["lang_code"]
    is_madlad = "madlad" in config["model_name"].lower()
    max_epochs = config.get("max_epochs", 30 if is_madlad else 10)
    trainable_mods = config.get("trainable_modules", "")
    probs = config.get("interleave_probs")
    is_multi = len(src_lang_codes) > 1

    # Token declaration
    tokens_str = ", ".join(src_lang_codes)
    token_plural = "tokens, both" if is_multi else "token"
    token_decl = f"{tokens_str} {token_plural} initialised as the mean of"

    # Initialisation source
    init_langs_std = INIT_LANGS_STD_MADLAD if is_madlad else INIT_LANGS_STD
    init_list = "[" + ", ".join(init_langs if init_langs else init_langs_std) + "]"
    if init_langs:
        init_str = init_list
    elif is_madlad:
        vec_name = os.path.basename(init_vecs)
        init_str = (
            f"{init_list} with fixed per-language orthogonal noise of L2=256, scaled so"
            f" inter-new-token distances have a similar ratio to inter-init-token distances"
            f" as in NLLB ({vec_name})"
        )
    else:
        vec_name = os.path.basename(init_vecs)
        if vec_name == "tok-only-train_tri.pt":
            init_str = f"{init_list} with orthogonal noise of L2=5 - then token-only warmup ({vec_name})"
        else:
            init_str = f"{init_list} with orthogonal noise of L2=5 ({vec_name})"

    # Trainable modules and tokens
    if is_madlad:
        modules_str = f"LoRA on {MADLAD_LORA_MODULES}"
    else:
        module_names = [MODULE_DICT[c] for c in trainable_mods if c in MODULE_DICT]
        modules_str = ", ".join(module_names)
    trainable_tokens = ", ".join(src_lang_codes + [zh_lang_code])

    # Data description
    if is_multi:
        effective_probs = probs if probs else [1] * len(src_lang_codes)
        total = sum(effective_probs)
        pct = [round(100 * p / total) for p in effective_probs]
        ratio = "/".join(str(p) for p in pct)
        lang_plus = lang_code.replace("-", "+")
        per_codes = [c.split("_")[0] for c in src_lang_codes]
        auxiliary = ", ".join(f"{c} resampled every epoch" for c in per_codes[1:])
        data_desc = f"interleaved {lang_plus} data ({ratio}, complete {per_codes[0]} training once per epoch, {auxiliary})"
    else:
        data_desc = f"{lang_code} data only"

    return (
        f"{base_model} expanded with {token_decl} {init_str}, "
        f"with {modules_str}, and tokens {trainable_tokens} trainable, "
        f"trained on {data_desc}, trained for {max_epochs} epochs"
    )


def r2(x):
    return round(x, 2)


def combined_score(bleu_zh, chrf_zuyu):
    # 0.4 * BLEU(zh) + 0.6 * chrF2++(zuyu)
    return round(0.4 * bleu_zh + 0.6 * chrf_zuyu, 3)


def process_finetune(finetune_dir, dir_name, prefix):
    config_path = os.path.join(finetune_dir, "config.json")
    if not os.path.exists(config_path):
        return []

    with open(config_path) as f:
        config = json.load(f)

    final_dir = os.path.join(finetune_dir, "final")
    if not os.path.isdir(final_dir):
        return []

    data_set_name = config["data_set_name"]
    ds_version = data_set_version(data_set_name)

    lang_en_list = config["lang_en"]
    src_lang_codes = config["src_lang_code"]
    if isinstance(src_lang_codes, str):
        src_lang_codes = [src_lang_codes]

    # Reorder so the intended primary language is first
    main_idx = MAIN_LANG_IDX.get(dir_name, 0)
    if main_idx != 0:
        lang_en_list = lang_en_list[main_idx:] + lang_en_list[:main_idx]
        src_lang_codes = src_lang_codes[main_idx:] + src_lang_codes[:main_idx]

    base_model = abbreviate_model(config["model_name"])
    modification = modification_from_dir(dir_name, data_set_name, prefix)
    language = " + ".join(lang_en_list)
    overrides = DESCRIPTION_OVERRIDES.get(dir_name, {})
    train_set = overrides.get("training_set") or training_set_name(config, ds_version)
    model_desc = overrides.get("model_description") or model_description(config, base_model)

    rows = []
    for i, (lang_en, src_code) in enumerate(zip(lang_en_list, src_lang_codes)):
        lcode = per_lang_code(src_code)
        json_path = os.path.join(final_dir, f"eval_scores_{lang_en}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            scores = json.load(f)

        test_json_path = os.path.join(final_dir, f"test_scores_{lang_en}.json")
        test_scores = {}
        if os.path.exists(test_json_path):
            with open(test_json_path) as f:
                test_scores = json.load(f)

        def t(key):
            return r2(test_scores[key]) if key in test_scores else ""

        vversion = val_version(final_dir, lcode)
        n_sentences = count_sentences(final_dir, lang_en, lcode)
        val_set = f"FB val {vversion} ({lcode})"

        bleu_zh   = scores["eval_bleu_zh"]
        chrf_zh   = scores["eval_chrf_zh"]
        bleu_zuyu = scores["eval_bleu_zuyu"]
        chrf_zuyu = scores["eval_chrf_zuyu"]

        row = {
            "base model":       base_model if i == 0 else "",
            "modification":     modification if i == 0 else "",
            "language":         language,
            "validation set":   val_set,
            "# sentences (val)": n_sentences,
            "BLEU (to_zh)":     r2(bleu_zh),
            "BLEU detail (to_zh)":    scores["eval_bleu_zh_detail"],
            "chrF2++ (to_zh)":        r2(chrf_zh),
            "CI lo (to_zh)":        r2(scores["eval_chrf_zh_ci_low"]) if "eval_chrf_zh_ci_low" in scores else "",
            "CI hi (to_zh)":        r2(scores["eval_chrf_zh_ci_high"]) if "eval_chrf_zh_ci_high" in scores else "",
            "BLEU (to_zuyu)":         r2(bleu_zuyu),
            "BLEU detail (to_zuyu)":  scores["eval_bleu_zuyu_detail"],
            "chrF2++ (to_zuyu)":      r2(chrf_zuyu),
            "CI lo (to_zuyu)":        r2(scores["eval_chrf_zuyu_ci_low"]) if "eval_chrf_zuyu_ci_low" in scores else "",
            "CI hi (to_zuyu)":        r2(scores["eval_chrf_zuyu_ci_high"]) if "eval_chrf_zuyu_ci_high" in scores else "",
            "combined score":   combined_score(bleu_zh, chrf_zuyu),
            "BLEU (to_zh, test)":       t("eval_bleu_zh"),
            "chrF2++ (to_zh, test)":    t("eval_chrf_zh"),
            "CI lo (to_zh, test)":      t("eval_chrf_zh_ci_low"),
            "CI hi (to_zh, test)":      t("eval_chrf_zh_ci_high"),
            "BLEU (to_zuyu, test)":     t("eval_bleu_zuyu"),
            "chrF2++ (to_zuyu, test)":  t("eval_chrf_zuyu"),
            "CI lo (to_zuyu, test)":    t("eval_chrf_zuyu_ci_low"),
            "CI hi (to_zuyu, test)":    t("eval_chrf_zuyu_ci_high"),
            "training set":     train_set if i == 0 else "",
            "description validation / test set":  VAL_DESCRIPTIONS[vversion],
            "description model (train set)":      model_desc if i == 0 else "",
        }
        rows.append(row)

    return rows


def main():
    all_rows = []

    for contributor, prefix in CONTRIBUTORS.items():
        contrib_dir = os.path.join(TRACKED_WORK, contributor)
        if not os.path.isdir(contrib_dir):
            continue

        for dir_name in sorted(os.listdir(contrib_dir)):
            finetune_dir = os.path.join(contrib_dir, dir_name)
            if not os.path.isdir(finetune_dir):
                continue
            rows = process_finetune(finetune_dir, dir_name, prefix)
            all_rows.extend(rows)

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"{date_str} finetune.tsv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
