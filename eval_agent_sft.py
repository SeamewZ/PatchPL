"""Agent-style eval replicating the exact SFT prompting format.

Prompt = OpenCode system prompt + <uploaded_files> + <issue> (same order as
the Nemotron SFT trajectories). The model loops with <tool_call> bash/read/task
until it stops calling tools; the final git diff of the repo working tree is
the model patch. Output is the official harness predictions format.
"""
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

MODEL_PATH = "models/Qwen3.5-4B"
LORA_DIR = os.environ.get("LORA_DIR", "checkpoints/qwen35-4b-lora-patch3")
TRAIN_DATA = "data/nemotron/deepseek_sft_chunks.jsonl"
TEST_DATA = "data/swe_smith_test250.jsonl"
OUT_FILE = "predictions/qwen35-4b-lora-ds-smith.json"
REPO_BASE = "repos"          # host-side repo checkouts
MAX_TURNS = 15               # text + tool-call rounds per instance
MAX_NEW = 800                # tokens per generation (stops at </tool_call>)
N_INSTANCES = int(os.environ.get("N_EVAL", "20"))
TEMP = float(os.environ.get("TEMP", "0"))
EVAL_RANK = int(os.environ.get("EVAL_RANK", "0"))
EVAL_WORLD = int(os.environ.get("EVAL_WORLD", "1"))


class StopOnToken(StoppingCriteria):
    """Stop generation as soon as the given token sequence appears."""
    def __init__(self, tokenizer, seq):
        self.ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
        self.k = len(self.ids)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -self.k:].tolist() == self.ids

# extracted from training data at startup
SYS_PROMPT = None


def extract_system(tok_path):
    with open(tok_path) as f:
        d = json.loads(f.readline())
    r = d["rendered"]
    m = re.search(r"(?s)^(.*?)<\|im_start\|>user", r)
    return m.group(1).strip()  # includes <|im_start|>system ... <|im_end|>


def build_user_msg(problem_statement):
    return (
        "<|im_start|>user\n"
        "<uploaded_files>\n/workspace/repo\n</uploaded_files>\n\n"
        "I've uploaded a code repository in the directory /workspace/repo. "
        "Consider the following issue description:\n\n"
        "<issue>\n" + problem_statement.strip() + "\n</issue>\n<|im_end|>\n"
    )


def run_bash(cmd, repo_dir):
    cmd = cmd.replace("/workspace/repo", repo_dir)
    try:
        p = subprocess.run(
            ["bash", "-lc", cmd], cwd=repo_dir, capture_output=True,
            text=True, timeout=60)
        out = (p.stdout or "")[-6000:]
        err = (p.stderr or "")[-2000:]
        res = out + ("\n[stderr]\n" + err if err else "")
        return (res or "(no output)")[:8000]
    except subprocess.TimeoutExpired:
        return "(bash timed out after 60s)"


def run_read(args, repo_dir):
    fp = str(args.get("filePath", "")).replace("/workspace/repo", repo_dir)
    if not os.path.isabs(fp):
        fp = os.path.join(repo_dir, fp)
    try:
        off = int(args.get("offset", 0))
        lim = int(args.get("limit", 2000))
    except Exception:
        off, lim = 0, 2000
    try:
        with open(fp, "r", errors="replace") as f:
            f.seek(0)
            lines = f.readlines()
        part = "".join(lines[off:off + lim])
        return part[:8000] if part else "(empty file)"
    except FileNotFoundError:
        return "(file not found: %s)" % fp
    except Exception as e:
        return "(read error: %s)" % e


def run_glob(args, repo_dir):
    import glob as g
    pat = str(args.get("pattern", "**/*"))
    base = str(args.get("path", "") or "").replace("/workspace/repo", repo_dir)
    if not os.path.isabs(base):
        base = os.path.join(repo_dir, base)
    pat = pat.replace("/workspace/repo", repo_dir)
    if not os.path.isabs(pat):
        pat = os.path.join(base, pat)
    try:
        hits = sorted(g.glob(pat, recursive=True))[:200]
        return ("\n".join(h.replace(repo_dir, "/workspace/repo") for h in hits)
                or "(no matches)")[:6000]
    except Exception as e:
        return "(glob error: %s)" % e


def run_grep(args, repo_dir):
    pat = str(args.get("pattern", ""))
    inc = str(args.get("include", "") or "")
    path = str(args.get("path", "") or ".").replace("/workspace/repo", repo_dir)
    if not os.path.isabs(path):
        path = os.path.join(repo_dir, path)
    cmd = ["grep", "-rn", "--include=" + (inc or "*"), "-m", "30", pat, path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (p.stdout or "(no matches)")[:6000]
    except subprocess.TimeoutExpired:
        return "(grep timed out)"
    except Exception as e:
        return "(grep error: %s)" % e


def run_edit(args, repo_dir):
    fp = str(args.get("filePath", "")).replace("/workspace/repo", repo_dir)
    if not os.path.isabs(fp):
        fp = os.path.join(repo_dir, fp)
    old = str(args.get("oldText", "") or "")
    new = str(args.get("newText", "") or "")
    try:
        with open(fp, "r", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return "(edit error: %s)" % e
    if old not in content:
        return ("(edit failed: oldText not found in %s)\nhead:\n%s"
                % (fp, content[:1500]))
    content = content.replace(old, new, 1)
    try:
        with open(fp, "w") as f:
            f.write(content)
        return "(edited %s)" % fp
    except Exception as e:
        return "(edit write error: %s)" % e


def run_patch(args, repo_dir):
    """Line-based replacement: swap lines [startLine, endLine] (1-based) for
    replacement text. Short args keep generation easy for small models."""
    fp = str(args.get("filePath", "")).replace("/workspace/repo", repo_dir)
    if not os.path.isabs(fp):
        fp = os.path.join(repo_dir, fp)
    try:
        start = int(args.get("startLine", 1))
        end = int(args.get("endLine", start))
        repl = str(args.get("replacement", "") or "")
    except Exception:
        return "(patch error: bad line numbers)"
    try:
        with open(fp, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "(patch error: file not found: %s)" % fp
    if start < 1 or end < start or start - 1 > len(lines):
        return ("(patch error: lines %d-%d out of range, file has %d lines)"
                % (start, end, len(lines)))
    new_lines = lines[:start - 1] + \
        ([repl + "\n"] if repl and not repl.endswith("\n") else [repl]) + \
        lines[end:]
    try:
        with open(fp, "w") as f:
            f.writelines(new_lines)
        return "(patched %s lines %d-%d)" % (fp, start, end)
    except Exception as e:
        return "(patch write error: %s)" % e


def run_write(args, repo_dir):
    fp = str(args.get("filePath", "")).replace("/workspace/repo", repo_dir)
    if not os.path.isabs(fp):
        fp = os.path.join(repo_dir, fp)
    content = str(args.get("content", "") or "")
    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as f:
            f.write(content)
        return "(wrote %s)" % fp
    except Exception as e:
        return "(write error: %s)" % e


def run_todo(args, repo_dir):
    todos = args.get("todos", [])
    n = len(todos) if isinstance(todos, list) else 0
    return "(todo list updated: %d items)" % n


def run_task(args, model, tok, conv_text):
    prompt = args.get("prompt", "")
    inner = conv_text + (
        "<|im_start|>assistant\n<tool_response>Task input:\n" + prompt +
        "\n</tool_response>\n<|im_start|>assistant\n")
    ids = tok(inner, add_special_tokens=False)["input_ids"][-10000:]
    iids = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model.generate(iids, max_new_tokens=1024, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, len(ids):], skip_special_tokens=False)
    gen = re.split(r"<\|im_end\|>|<tool_call>", gen)[0]
    return gen.strip()[:4000]


def parse_tool_call(text):
    m = re.search(r"(?s)<tool_call>\s*#?(\w+)\s*(.*?)</tool_call>", text)
    if not m:
        return None
    tool, raw = m.group(1), m.group(2).strip()
    if not raw:
        return (tool, {})
    try:
        args = json.loads(raw)
    except Exception:
        return (tool, {})
    return (tool, args if isinstance(args, dict) else {})


def repo_dir_of(item):
    # repo field looks like "swesmith/org__repo.commithash"
    slug = item["repo"].split("/")[-1]
    return os.path.abspath(os.path.join(REPO_BASE, slug))


def eval_instance(item, model, tok):
    repo_dir = repo_dir_of(item)
    conv = SYS_PROMPT + "\n" + build_user_msg(item["problem_statement"]) + \
        "<|im_start|>assistant\n"
    stop = StopOnToken(tok, "</tool_call>")
    new_tokens_total = 0
    TOKEN_BUDGET = 5000
    forced_first = True
    for turn in range(MAX_TURNS):
        if forced_first and turn == 0:
            # the model only starts tool-calling after seeing the
            # interfaces block it was trained with; force the first call
            conv += "<tool_call>"
        ids = tok(conv, add_special_tokens=False)["input_ids"][-10000:]
        iids = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                iids, max_new_tokens=800,
                do_sample=(TEMP > 0), temperature=TEMP,
                pad_token_id=tok.eos_token_id, stopping_criteria=[stop])
        gen = tok.decode(out[0, len(ids):], skip_special_tokens=False)
        if forced_first and turn == 0:
            gen = "<tool_call>" + gen
        conv += gen
        new_tokens_total += (out.size(1) - len(ids))
        ended = out[0, -1].item() == tok.eos_token_id
        if os.environ.get("DEBUG"):
            print("    [turn %d] new_tok=%d ended=%s | %s" % (
                turn, out.size(1) - len(ids), ended,
                gen[:200].replace(chr(10), " ")), flush=True)
        # execute every complete tool_call block in this generation
        calls = re.findall(
            r"(?s)<tool_call>\s*#?(\w+)\s*(.*?)</tool_call>", gen)
        if calls:
            for tool, raw in calls:
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if tool == "bash":
                    res = run_bash(str(args.get("command", "")), repo_dir)
                elif tool == "read":
                    res = run_read(args, repo_dir)
                elif tool == "glob":
                    res = run_glob(args, repo_dir)
                elif tool == "grep":
                    res = run_grep(args, repo_dir)
                elif tool == "edit":
                    res = run_edit(args, repo_dir)
                elif tool == "patch":
                    res = run_patch(args, repo_dir)
                elif tool == "write":
                    res = run_write(args, repo_dir)
                elif tool == "todowrite":
                    res = run_todo(args, repo_dir)
                elif tool == "task":
                    res = run_task(args, model, tok, conv)
                else:
                    res = "(unknown tool: %s)" % tool
                conv += "\n<|im_start|>user\n<tool_response>\n" + res + \
                        "\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"
        elif ended or new_tokens_total >= TOKEN_BUDGET:
            break  # model finished on its own / budget exhausted
        # text-only turn: keep generating
    # final patch = git diff of working tree (include newly created files)
    try:
        subprocess.run(["git", "-C", repo_dir, "add", "-A"],
                       capture_output=True, timeout=60)
        p = subprocess.run(["git", "-C", repo_dir, "diff", "--cached"],
                           capture_output=True, text=True, timeout=60)
        patch = p.stdout
    except Exception:
        patch = ""
    if os.environ.get("DEBUG"):
        print("    --- conv tail ---")
        print(conv[-1800:].replace(chr(10), " ")[:1800], flush=True)
    return patch


def main():
    global SYS_PROMPT
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    SYS_PROMPT = extract_system(TRAIN_DATA)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda:0")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()

    test = [json.loads(l) for l in open(TEST_DATA)][:N_INSTANCES]
    if EVAL_WORLD > 1:
        test = test[EVAL_RANK::EVAL_WORLD]
    print("rank %d/%d: evaluating %d instances" % (
        EVAL_RANK, EVAL_WORLD, len(test)))
    preds = {}
    t0 = time.time()
    for i, item in enumerate(test):
        try:
            patch = eval_instance(item, model, tok)
            preds[item["instance_id"]] = {
                "model_name_or_path": "qwen35-4b-lora-ds-agent",
                "model_patch": patch,
            }
            print("[%03d/%d] %s patch=%dB %.0fs" % (
                i + 1, len(test), item["instance_id"], len(patch),
                time.time() - t0), flush=True)
        except Exception as e:
            print("[%03d/%d] %s ERROR %s" % (
                i + 1, len(test), item["instance_id"], e), flush=True)
            preds[item["instance_id"]] = {
                "model_name_or_path": "qwen35-4b-lora-ds-agent",
                "model_patch": "",
            }
    os.makedirs("predictions", exist_ok=True)
    out_file = OUT_FILE
    if EVAL_WORLD > 1:
        out_file = OUT_FILE.replace(".json", ".%d.json" % EVAL_RANK)
    with open(out_file, "w") as f:
        json.dump(preds, f, indent=2)
    print("saved ->", out_file, "| instances:", len(preds))


if __name__ == "__main__":
    main()
