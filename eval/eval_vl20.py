#!/usr/bin/env python3
"""Evaluate the trained masked-SFT LoRA on the 10 validation instances.

Loads Qwen3.5-4B + checkpoints/qwen35-4b-lora-vl20, runs agent loop with
tools executing inside official eval containers. Parses the Qwen3.5
<function=/<parameter= tool-call format. Judge outcome via FAIL_TO_PASS.
"""
import glob
import json
import os
import re
import subprocess
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "models/Qwen3.5-4B"
LORA_PATH = os.environ.get("LORA_PATH", "checkpoints/qwen35-4b-lora-vl20-full")
INSTS = os.environ.get("INSTS", "data/swe_verify/validate10.json")
OUT_DIR = os.environ.get("EVAL_OUT", "data/swe_verify/eval_vl20")
EVAL_SHARD = int(os.environ.get("EVAL_SHARD", "0"))
EVAL_NUM = int(os.environ.get("EVAL_NUM", "1"))
MAX_TURNS = 12
MAX_NEW = 1536
MAX_PROMPT = 6144
TEST_TIMEOUT = 600
ACT = "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && cd /testbed"

SYS = (
    "You are an autonomous software engineering agent working inside a git "
    "repository checkout mounted at /testbed. Fix the GitHub issue "
    "described by the user by editing repository files. Use the provided "
    "tools (bash / read / edit). You may call multiple tools per message. "
    "Run relevant tests with bash before finishing. When the fix is done, "
    "stop calling tools and output a short summary."
)


def sh(cmd, timeout=300):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def img_of(ex):
    org = ex["repo"].split("/")[0]
    repo = ex["repo"].split("/")[1]
    pr = ex["instance_id"].split("-")[-1]
    return "swebench/sweb.eval.x86_64.%s_1776_%s-%s" % (org, repo, pr)


def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        block = m.group(1)
        for fm in re.finditer(r"<function=([\w.-]+)>(.*?)</function>",
                              block, re.S):
            name = fm.group(1)
            body = fm.group(2)
            args = {}
            for pm in re.finditer(
                    r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>",
                    body, re.S):
                args[pm.group(1)] = pm.group(2)
            calls.append({"name": name, "arguments": args})
    return calls


def exec_bash(cid, cmd, timeout=120):
    tag = uuid.uuid4().hex[:8]
    hostf = "/tmp/ev_%s.sh" % tag
    contf = "/tmp/ev_%s.sh" % tag
    with open(hostf, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n%s\n%s\n" % (ACT, cmd))
    sh("docker cp %s %s:%s" % (hostf, cid, contf), timeout=60)
    try:
        out = sh("docker exec %s bash %s" % (cid, contf), timeout=timeout)
    except subprocess.TimeoutExpired:
        out = "(bash timed out)"
    os.remove(hostf)
    return out


def run_tool(cid, name, args):
    if name == "bash":
        return (exec_bash(cid, str(args.get("command", ""))) or
                "(no output)")[:6500]
    if name == "read":
        p = str(args.get("path", "")).replace("/testbed/", "")
        return (sh('docker exec %s cat "/testbed/%s"' % (cid, p),
                   timeout=60)[:6000] or
                "(read failed: no such file %s)" % p)
    if name == "edit":
        p = str(args.get("path", "")).replace("/testbed/", "")
        old = str(args.get("old", ""))
        new = str(args.get("new", ""))
        tag = uuid.uuid4().hex[:8]
        hostf = "/tmp/ev_edit_%s" % tag
        sh('docker cp %s:"/testbed/%s" %s 2>/dev/null' % (cid, p, hostf),
           timeout=60)
        if not os.path.isfile(hostf):
            return "(edit failed: no such file %s)" % p
        with open(hostf, encoding="utf-8", errors="replace") as f:
            src = f.read()
        if old not in src:
            os.remove(hostf)
            return "(edit failed: old string not found)"
        with open(hostf, "w", encoding="utf-8") as f:
            f.write(src.replace(old, new, 1))
        sh('docker cp %s %s:"/testbed/%s"' % (hostf, cid, p), timeout=60)
        os.remove(hostf)
        return "(edit applied to %s)" % p
    return "(unknown tool %s)" % name


def convert_test(t):
    t = t.strip()
    m = re.match(r"^(.+?)\s+\(([\w.]+?)(?:\.(\w+))?\)$", t)
    if m:
        name, mod, cls = m.group(1), m.group(2), m.group(3)
        return ("%s.%s.%s" % (mod, cls, name) if cls
                else "%s.%s" % (mod, name))
    return t


def judge(cid, ex):
    f2p = ex.get("FAIL_TO_PASS") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except Exception:
            f2p = []
    if not f2p:
        return "unknown"
    targs = " ".join(convert_test(t) for t in f2p)
    if ex.get("repo") == "django/django":
        out = exec_bash(cid, "python tests/runtests.py --verbosity 1 %s"
                        % targs, timeout=TEST_TIMEOUT)
        if "FAILED" in out:
            return "fail"
        if "OK" in out:
            return "pass"
        return "unknown"
    out = exec_bash(cid, "python -m pytest -q %s" % targs,
                    timeout=TEST_TIMEOUT)
    m_fail = re.search(r"(\d+) failed", out)
    m_pass = re.search(r"(\d+) passed", out)
    if m_fail and int(m_fail.group(1)) > 0:
        return "fail"
    if m_pass and int(m_pass.group(1)) > 0:
        return "pass"
    return "unknown"


def load_insts():
    if INSTS.endswith(".jsonl"):
        return [json.loads(l) for l in open(INSTS, encoding="utf-8")]
    return json.load(open(INSTS, encoding="utf-8"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda:0")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    insts = load_insts()
    insts = insts[EVAL_SHARD::EVAL_NUM]
    done = {os.path.basename(f)[:-len(".json")]
            for f in glob.glob(os.path.join(OUT_DIR, "*.json"))}
    for ex in insts:
        if ex["instance_id"] in done:
            continue
        print("== %s start %s" % (ex["instance_id"],
                                  time.strftime("%H:%M")), flush=True)
        img = img_of(ex)
        if "OK" not in sh("docker image inspect %s >/dev/null 2>&1 "
                          "&& echo OK" % img):
            print("SKIP %s (no image)" % ex["instance_id"], flush=True)
            continue
        cid = "ev_%s" % re.sub(r"[^a-zA-Z0-9_.-]", "_", ex["instance_id"])
        sh("docker rm -f %s >/dev/null 2>&1" % cid)
        sh("docker run -d --name %s -w /testbed %s sleep infinity"
           % (cid, img), timeout=180)
        sh("docker exec %s bash -c '%s && python -c \"import pytest\" "
           "2>/dev/null || pip install pytest -q 2>/dev/null'"
           % (cid, ACT), timeout=300)

        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content":
             "Fix the following GitHub issue in the repository at "
             "/testbed:\n\n<issue>\n" + ex["problem_statement"] +
             "\n</issue>"},
        ]
        trace = []
        outcome = "unknown"
        for turn in range(MAX_TURNS):
            torch.cuda.empty_cache()
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          tokenize=True,
                                          return_tensors="pt")
            prompt = enc["input_ids"]
            if prompt.size(1) > MAX_PROMPT:
                prompt = prompt[:, -MAX_PROMPT:]
            prompt = prompt.to("cuda:0")
            with torch.no_grad():
                gen = model.generate(prompt, max_new_tokens=MAX_NEW,
                                     do_sample=True, temperature=0.7,
                                     top_p=0.9, pad_token_id=tok.eos_token_id)
            new = gen[0, prompt.size(1):]
            text = tok.decode(new, skip_special_tokens=False)
            calls = parse_tool_calls(text)
            if not calls:
                msgs.append({"role": "assistant",
                             "content": text[:500]})
                trace.append({"turn": turn, "role": "assistant",
                              "text": text[:500], "tools": []})
                break
            tc_struct = []
            trace_tools = []
            for i, c in enumerate(calls):
                tc_id = "call_%d_%d" % (turn, i)
                tc_struct.append({
                    "id": tc_id, "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": c["arguments"]}})
                trace_tools.append(c)
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": tc_struct})
            trace.append({"turn": turn, "role": "assistant",
                          "text": text[:300], "tools": trace_tools})
            for i, c in enumerate(calls):
                res = run_tool(cid, c["name"], c["arguments"])
                msgs.append({"role": "tool",
                             "content": res,
                             "tool_call_id": "call_%d_%d" % (turn, i)})
                trace.append({"turn": turn, "role": "tool",
                              "tool": c["name"],
                              "result": res[:300]})

        outcome = judge(cid, ex)
        sh("docker rm -f %s >/dev/null 2>&1" % cid)
        rec = {"instance_id": ex["instance_id"], "outcome": outcome,
               "turns": len([t for t in trace if t["role"] == "assistant"]),
               "trace": trace}
        with open(os.path.join(OUT_DIR, "%s.json" % ex["instance_id"]),
                  "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print("saved %s outcome=%s %s" % (ex["instance_id"], outcome,
                                          time.strftime("%H:%M")), flush=True)
    print("EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()
