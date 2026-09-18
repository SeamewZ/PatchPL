#!/usr/bin/env python3
"""Stage-2 patch SFT: continue from the trajectory-SFT LoRA.

Same manual-loop infrastructure (sub-vocab CE, per-layer checkpointing).
Trains on synthetic fix-trajectories (issue -> read -> edit -> verify) so the
model learns the repair loop on top of the agent format.
"""
import json
import math
import os
import random
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Qwen3.5-4B"
TRAIN_DATA = "data/patch_sft_v2/train.jsonl"
OUTPUT_DIR = "checkpoints/qwen35-4b-lora-patch"
INIT_LORA = "checkpoints/qwen35-4b-lora-ds"
MAX_SEQ_LENGTH = 16384
CHUNK = 2048
EPOCHS = 3
GRAD_ACC = 8
LR = 5e-5
WARMUP = 20
STATE_FILE = os.path.join(OUTPUT_DIR, "state.json")
DONE_MARKER = "QWEN35-4B-LORA-PATCH-DONE"

MARKER = re.compile(r'(?=<\|im_start\|>(?:system|user|assistant))')


def tokenize(rendered, tok):
    segs = [s for s in MARKER.split(rendered) if s]
    ids, labels = [], []
    head_len = 0
    for i, seg in enumerate(segs):
        is_assistant = seg.startswith("<|im_start|>assistant")
        sid = tok(seg, add_special_tokens=False)["input_ids"]
        ids.extend(sid)
        labels.extend(sid if is_assistant else [-100] * len(sid))
        if i < 2:
            head_len = len(ids)
    if len(ids) > MAX_SEQ_LENGTH:
        head_keep = min(head_len, MAX_SEQ_LENGTH // 2)
        tail_keep = MAX_SEQ_LENGTH - head_keep
        ids = ids[:head_keep] + ids[-tail_keep:]
        labels = labels[:head_keep] + labels[-tail_keep:]
    return ids, labels


def subvocab_ce(hidden, labels, lm_weight, tok_ignore=-100):
    pos = labels != tok_ignore
    uniq, inv = torch.unique(labels[pos], return_inverse=True)
    cls = torch.full_like(labels, -100)
    cls[pos] = inv
    logits = F.linear(hidden, lm_weight[uniq])
    shift = logits[:-1].contiguous()
    lb = cls[1:].contiguous()
    total = torch.zeros((), device=hidden.device)
    ntok = 0
    for i in range(0, shift.size(0), CHUNK):
        lg = shift[i:i + CHUNK].float()
        lc = lb[i:i + CHUNK]
        mask = lc != -100
        n = mask.sum().item()
        if n == 0:
            continue
        total = total + F.cross_entropy(lg[mask], lc[mask], reduction="sum")
        ntok += n
    return total / max(ntok, 1)


def save_state(step, ep, pos, order):
    with open(STATE_FILE, "w") as f:
        json.dump({"step": step, "epoch": ep, "pos": pos, "order": order}, f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = [json.loads(l) for l in open(TRAIN_DATA)]
    samples = []
    n_assist, n_total = 0, 0
    for d in data:
        ids, lab = tokenize(d["rendered"], tok)
        if not ids:
            continue
        samples.append((ids, lab))
        n_assist += sum(1 for x in lab if x != -100)
        n_total += len(ids)
    print("samples:", len(samples), "| assistant tokens: %.1f%%"
          % (100.0 * n_assist / n_total))

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda:0")
    base.config.use_cache = False

    targets = set()
    for name, mod in base.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "wq", "wk", "wv", "wo", "w1", "w2", "w3"):
                targets.add(leaf)
    targets = sorted(targets)
    print("LoRA targets:", targets)

    adapter_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    init_path = os.path.join(INIT_LORA, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        print("RESUME: loading adapter from", OUTPUT_DIR)
        model = PeftModel.from_pretrained(base, OUTPUT_DIR, is_trainable=True)
    elif os.path.exists(init_path):
        print("INIT: continuing from trajectory-SFT LoRA at", INIT_LORA)
        model = PeftModel.from_pretrained(base, INIT_LORA, is_trainable=True)
    else:
        print("FRESH: initializing new LoRA adapter")
        model = get_peft_model(base, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()

    orig = model.base_model.model
    body = orig.model
    lm_weight = orig.lm_head.weight.detach()

    orig.enable_input_require_grads()

    def make_wrapped(orig_fn):
        def wrapped(*args, **kwargs):
            def run(*a):
                return orig_fn(*a, **kwargs)
            return checkpoint(run, *args, use_reentrant=False)
        return wrapped

    for layer in body.layers:
        layer.forward = make_wrapped(layer.forward)
    print("manual per-layer checkpointing installed")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    steps_per_epoch = math.ceil(len(samples) / GRAD_ACC)
    total_steps = steps_per_epoch * EPOCHS
    print("steps/epoch:", steps_per_epoch, "total:", total_steps)

    step = 0
    start_ep = 0
    start_pos = 0
    order = None
    if os.path.exists(STATE_FILE):
        st = json.load(open(STATE_FILE))
        step = st.get("step", 0)
        start_ep = st.get("epoch", 0)
        start_pos = st.get("pos", 0)
        order = st.get("order")
        print("RESUME state: step=%d epoch=%d pos=%d"
              % (step, start_ep, start_pos))

    t0 = time.time()
    for ep in range(start_ep, EPOCHS):
        if order is None or ep != start_ep:
            order = list(range(len(samples)))
            random.Random(1000 + ep).shuffle(order)
        opt.zero_grad()
        acc = 0
        pos = start_pos if ep == start_ep else 0
        while pos < len(order):
            ids, lab = samples[order[pos]]
            iids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
            ll = torch.tensor([lab], dtype=torch.long, device="cuda:0")
            hs = body(input_ids=iids).last_hidden_state[0]
            loss = subvocab_ce(hs, ll[0], lm_weight)
            if not loss.requires_grad:
                print("skip empty-assistant sample pos=%d" % pos, flush=True)
                pos += 1
                continue
            (loss / GRAD_ACC).backward()
            acc += 1
            pos += 1
            if acc == GRAD_ACC:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                acc = 0
                step += 1
                if step <= WARMUP:
                    lr = LR * step / WARMUP
                else:
                    prog = (step - WARMUP) / max(1, total_steps - WARMUP)
                    lr = LR * 0.5 * (1 + math.cos(math.pi * prog))
                for g in opt.param_groups:
                    g["lr"] = lr
                if step % 5 == 0:
                    el = time.time() - t0
                    print("step %d/%d loss=%.4f lr=%.2e %.0fs"
                          % (step, total_steps, loss.item(), lr, el), flush=True)
                if step % 100 == 0:
                    model.save_pretrained(OUTPUT_DIR)
                    save_state(step, ep, pos, order)
                    print("saved checkpoint at step", step, flush=True)
        save_state(step, ep + 1, 0, None)
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(DONE_MARKER)


if __name__ == "__main__":
    main()
