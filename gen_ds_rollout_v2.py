#!/usr/bin/env python3
"""Generate multi-turn DeepSeek agent rollouts for 10 verified instances (3 each).

Uses DeepSeek native function calling (tools: bash/read/edit).
Trajectory messages are saved for the labeling step.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

API = "https://api.deepseek.com/chat/completions"
KEY = re.sub(r"[^\x20-\x7e]", "",
             open(os.path.expanduser("~/.deepseek_key"), encoding="utf-8",
                  errors="replace").read()).strip()

INSTS = "data/swe_verify/validate10.json"
OUT_DIR = "data/swe_verify/rollouts"
REPO_ROOT = "repos"
N_SAMPLES = int(os.environ.get("N_SAMPLES", "3"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "25"))
TEMP = float(os.environ.get("TEMP", "0.7"))

SYS = (
    "You are an autonomous software engineering agent working inside a git "
    "repository checkout mounted at /workspace/repo. Fix the GitHub issue "
    "described by the user by editing repository files. Use the provided "
    "tools (bash / read / edit). You may call multiple tools per message. "
    "Run relevant tests with bash before finishing. When the fix is done, "
    "stop calling tools and output a short summary."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command inside /workspace/repo (e.g. run "
                       "tests, search code, list files). Output truncated.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Print contents of a file (path relative to repo root).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace the FIRST occurrence of `old` with `new` in a "
                       "file (path relative to repo root).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "old": {"type": "string"},
                                      "new": {"type": "string"}},
                       "required": ["path", "old", "new"]}}},
]


def call_api(messages, max_retry=4):
    payload = {"model": "deepseek-chat", "messages": messages,
               "tools": TOOLS, "tool_choice": "auto",
               "temperature": TEMP, "max_tokens": 4096, "stream": False}
    for attempt in range(max_retry):
        try:
            req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + KEY})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode())
            return d["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print("  api http %s: %s" % (e.code, body), flush=True)
            time.sleep(10)
        except Exception as e:
            print("  api err:", str(e)[:120], flush=True)
            time.sleep(15)
    return None


def run_tool(name, args, repo_dir):
    try:
        if name == "bash":
            cmd = str(args.get("command", "")).replace("/workspace/repo", repo_dir)
            p = subprocess.run(["bash", "-lc", cmd], cwd=repo_dir,
                               capture_output=True, text=True, timeout=90)
            out = (p.stdout or "")[-5000:] + (p.stderr or "")[-1500:]
            return (out or "(no output)")[:6500]
        if name == "read":
            path = re.sub(r"^/workspace/repo/", "", str(args.get("path", "")))
            full = os.path.join(repo_dir, path)
            if not os.path.isfile(full):
                return "(read failed: no such file %s)" % path
            with open(full, encoding="utf-8", errors="replace") as f:
                return f.read()[:6000]
        if name == "edit":
            path = re.sub(r"^/workspace/repo/", "", str(args.get("path", "")))
            old = str(args.get("old", ""))
            new = str(args.get("new", ""))
            full = os.path.join(repo_dir, path)
            if not os.path.isfile(full):
                return "(edit failed: no such file %s)" % path
            with open(full, encoding="utf-8", errors="replace") as f:
                src = f.read()
            if old not in src:
                return "(edit failed: old string not found)"
            with open(full, "w", encoding="utf-8") as f:
                f.write(src.replace(old, new, 1))
            return "(edit applied to %s)" % path
        return "(unknown tool %s)" % name
    except subprocess.TimeoutExpired:
        return "(bash timed out)"
    except Exception as e:
        return "(tool error: %s)" % str(e)[:200]


def infer_outcome(res):
    low = res.lower()
    if "failed" in low or "error" in low:
        return "fail"
    if "passed" in low:
        return "pass"
    return "unknown"


def rollout(ex, repo_dir, sid):
    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run(["git", "checkout", "-f", ex["base_commit"]], cwd=repo_dir,
                   capture_output=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo_dir, capture_output=True)

    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "Fix the following GitHub issue in "
         "the repository at /workspace/repo:\n\n<issue>\n" +
         ex["problem_statement"] + "\n</issue>"},
    ]
    api_msgs = [dict(m) for m in msgs]
    outcome = "unknown"
    done = False
    for turn in range(MAX_TURNS):
        msg = call_api(api_msgs)
        if msg is None:
            break
        tcalls = msg.get("tool_calls") or []
        if not tcalls:
            msgs.append({"role": "assistant",
                         "content": msg.get("content") or "(no text)"})
            api_msgs.append(dict(msgs[-1]))
            done = True
            break
        # record + send assistant msg with tool_calls
        msgs.append({"role": "assistant",
                     "content": msg.get("content") or "",
                     "tool_calls": tcalls})
        api_msgs.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcalls})
        tool_results = []
        for tc in tcalls:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"raw": fn.get("arguments")}
            res = run_tool(name, args, repo_dir)
            tool_results.append((tc.get("id", "t%d" % turn),
                                 "Tool %s(%s):\n%s" % (name,
                                  json.dumps(args)[:400], res)))
            if name == "bash" and re.search(r"pytest|test", str(args)):
                o = infer_outcome(res)
                if outcome == "unknown" and o != "unknown":
                    outcome = o
        for tid, tres in tool_results:
            msgs.append({"role": "tool", "content": tres, "tool_call_id": tid})
            api_msgs.append({"role": "tool", "content": tres,
                             "tool_call_id": tid})
        if done:
            break

    rec = {"instance_id": ex["instance_id"], "repo": ex["repo"],
           "base_commit": ex["base_commit"], "sample": sid,
           "problem_statement": ex["problem_statement"],
           "messages": msgs, "outcome": outcome}
    out = os.path.join(OUT_DIR, "%s__s%d.json" % (ex["instance_id"], sid))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("saved %s outcome=%s msgs=%d" % (os.path.basename(out), outcome,
                                            len(msgs)), flush=True)
    return rec


def main():
    insts = json.load(open(INSTS))
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(insts)
    for i, ex in enumerate(insts):
        if i < start or i >= end:
            continue
        repo_dir = os.path.join(REPO_ROOT, ex["repo"].split("/")[-1])
        print("== %s (%s)" % (ex["instance_id"], ex["repo"]), flush=True)
        for sid in range(N_SAMPLES):
            try:
                rollout(ex, repo_dir, sid)
            except Exception as e:
                print("rollout failed:", str(e)[:200], flush=True)
                time.sleep(5)


if __name__ == "__main__":
    main()
