import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
import json
from pathlib import Path
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM


class NLLBEmbeddingGradientCallback(TrainerCallback):
    def __init__(self, vocab_size, dtype, device, trainable_indices):
        self.mask = torch.zeros((vocab_size, 1), dtype=dtype, device=device)
        self.mask[trainable_indices] = 1.0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grad = model.get_input_embeddings().weight.grad
        if grad is not None:
            grad.mul_(self.mask)


class SaveCustomLMHeadCallback(TrainerCallback):
    def __init__(self, new_token_ids):
        super().__init__()
        self.new_token_ids = new_token_ids
        
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        save_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        lm_head = base_model.get_output_embeddings()

        if isinstance(lm_head, PartialTrainableLMHead):
            save_custom_lm_head(base_model, save_dir, self.new_token_ids)

        return control
        

def load_model_with_custom_lm_head(
    base_model_name,
    adapter_dir,
    tokenizer_path=None,
    bnb_config=None,
    torch_dtype=torch.float16,
    device_map="auto",
):
    adapter_dir = Path(adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    print("Base model loaded! Resizing token embeddings...")

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    print("Loading custom LM head...")

    with open(adapter_dir / "custom_lm_head_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    new_token_ids = meta["new_token_ids"]

    model = attach_partial_lm_head(model, new_token_ids)

    lm_head_state = torch.load(
        adapter_dir / "custom_lm_head.pt",
        map_location="cpu"
    )
    model.get_output_embeddings().load_state_dict(lm_head_state)

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return model, tokenizer, new_token_ids


def attach_partial_lm_head(model, new_token_ids, init_from_base=True):
    old_lm_head = model.get_output_embeddings()
    new_lm_head = PartialTrainableLMHead(old_lm_head, new_token_ids, init_from_base=init_from_base)
    new_lm_head.to(device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
    model.set_output_embeddings(new_lm_head)
    return model


def save_custom_lm_head(model, save_dir, new_token_ids):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lm_head = model.get_output_embeddings()
    if not isinstance(lm_head, PartialTrainableLMHead):
        raise TypeError(f"Expected PartialTrainableLMHead, got {type(lm_head)}")

    torch.save(
        lm_head.state_dict(),
        save_dir / "custom_lm_head.pt"
    )

    with open(save_dir / "custom_lm_head_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "new_token_ids": list(map(int, new_token_ids)),
                "class_name": "PartialTrainableLMHead",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


class PartialTrainableLMHead(nn.Module):
    def __init__(self, base_lm_head, new_token_ids, init_from_base=True):
        super().__init__()
        self.base_lm_head = base_lm_head
        self.base_lm_head.weight.requires_grad_(False)

        device = base_lm_head.weight.device
        dtype = base_lm_head.weight.dtype

        new_token_ids = torch.as_tensor(new_token_ids, dtype=torch.long, device=device)
        self.register_buffer("new_token_ids", new_token_ids, persistent=False)

        hidden_size = base_lm_head.weight.shape[1]
        num_new = len(new_token_ids)

        self.new_rows = nn.Parameter(
            torch.empty(num_new, hidden_size, device=device, dtype=dtype)
        )
        
        with torch.no_grad():
            if init_from_base:
                self.new_rows.copy_(base_lm_head.weight.data[new_token_ids])
            else:
                nn.init.normal_(self.new_rows, mean=0.0, std=0.02)

        if base_lm_head.bias is not None:
            self.register_buffer(
                "base_bias",
                base_lm_head.bias.detach().clone().to(device=device, dtype=dtype),
                persistent=False
            )
            self.new_bias = nn.Parameter(
                base_lm_head.bias.data[new_token_ids].clone().to(device=device, dtype=dtype)
            )
        else:
            self.base_bias = None
            self.new_bias = None

    def forward(self, hidden_states):
        # frozen full logits
        logits = F.linear(hidden_states, self.base_lm_head.weight, self.base_bias)

        # recompute only new-token columns with trainable rows
        new_logits = F.linear(hidden_states, self.new_rows, self.new_bias)

        # replace those columns
        logits[..., self.new_token_ids] = new_logits
        return logits