"""Shared loading and generation for the Dolos audit. Loads from the local HF cache only."""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/Volumes/PortableSSD/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# revisions pinned in starter/redteaming/generate.py
REPOS = {
    "dolos": ("Poseidon-Research/Dolos-1.5B", "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"),
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
}
QWEN_REPO, QWEN_REV = REPOS["qwen"]
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# fp16 copies written by to_fp16.py; loading them halves the bytes read under memory pressure.
FP16_DIR = Path(os.environ.get("FP16_DIR", "/Volumes/PortableSSD/scratch/dolos_fp16"))


def weights_path(name):
    repo, rev = REPOS[name]
    return Path(hf_hub_download(repo, "model.safetensors", revision=rev))


def tokenizer():
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REV)
    tok.padding_side = "left"
    return tok


def load(name, dtype=torch.float16, device=DEVICE):
    config = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REV)
    src = FP16_DIR / name if (FP16_DIR / name / "model.safetensors").exists() and dtype == torch.float16 \
        else weights_path(name).parent
    model, info = AutoModelForCausalLM.from_pretrained(
        src, config=config,
        generation_config=GenerationConfig.from_model_config(config),
        dtype=dtype, attn_implementation="sdpa", use_safetensors=True,
        local_files_only=True, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    if bad:
        raise RuntimeError(f"incomplete load: {bad}")
    model.tie_weights()
    return model.to(device).eval()


def chat(tok, prompt, system=DEFAULT_SYSTEM, prefill=""):
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + [
        {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill


def chat_turns(tok, turns, system=DEFAULT_SYSTEM, prefill=""):
    """turns alternate user, assistant, user, ...; the template ends on an open assistant turn."""
    msgs = [{"role": "system", "content": system}] if system is not None else []
    msgs += [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill


@torch.inference_mode()
def letter_probs(model, tok, texts, letters_per_text, batch=4, space=False):
    """Next-token distribution restricted to each item's option letters (texts end in a prefill like
    'The answer is ('). Projects only the last position to keep MPS memory small. space=True scores
    ' A' etc., for prefills that end in a word ('The answer is') rather than a bracket."""
    ids = {c: tok.encode((" " if space else "") + c, add_special_tokens=False)[0] for c in "ABCDEFGHIJ"}
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    # token budget per batch: long items (some WMDP-cyber prompts exceed 2k tokens) run alone
    chunks, s = [], 0
    while s < len(order):
        n_tok = len(tok(texts[order[min(s + batch, len(order)) - 1]]).input_ids)
        b = max(1, min(batch, 2048 // max(n_tok, 1)))
        chunks.append(order[s:s + b])
        s += b
    for idx in chunks:
        # pad to a multiple of 64: fewer distinct shapes, so MPS caches fewer compiled graphs
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, pad_to_multiple_of=64).to(model.device)
        logits = model.lm_head(model.model(**enc).last_hidden_state[:, -1]).float()
        full = torch.log_softmax(logits, -1)
        for j, i in enumerate(idx):
            L = letters_per_text[i]
            lp = full[j, [ids[c] for c in L]]
            out[i] = {"letters": L, "p": torch.softmax(lp, -1).tolist(), "mass": lp.exp().sum().item()}
        del enc, logits, full
        if model.device.type == "mps":
            torch.mps.empty_cache()
    return out


@torch.inference_mode()
def generate(model, tok, texts, max_new_tokens=200, temperature=0.0, seed=0, batch=8):
    """texts are fully templated strings; returns completions in order."""
    out = []
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")]
    for i in range(0, len(texts), batch):
        torch.manual_seed(seed + i)
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to(model.device)
        kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens,
                  eos_token_id=eos, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=0.9, top_k=50)
        gen = model.generate(**enc, **kw)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out
