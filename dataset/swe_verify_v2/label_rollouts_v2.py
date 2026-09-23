#!/usr/bin/env python3
"""Label rollout trajectories with DeepSeek: each assistant msg -> 0/1 useful.

v2 improvements:
- criterion (e): edits are checked against the GOLD patch. An edit whose
  old/new matches the gold patch direction is annotated [GOLD-MATCH];
  reverse direction is [GOLD-REVERSE]; otherwise [GOLD-NEUTRAL].
- judge prompt prioritizes GOLD-MATCH -> 1 and GOLD-REVERSE -> 0.

Reads rollouts from data/swe_verify/rollouts_img/, writes per-trajectory
verdicts to data/swe_verify/labels_img/.
"""
import glob
import json
import os
import re
import time
import urllib.request
import urllib.error

API = "https://api.deepseek.com/chat/completions"
KEY = re.sub(r"[^\x20-\x7e]", "",
             open(os.path.expanduser("~/.deepseek_key"), encoding="utf-8",
                  errors="replace").read()).strip()

ROLLOUT_DIR = os.environ.get("OUT_DIR", "data/swe_verify/rollouts_img")
LABEL_DIR = os.environ.get("LABEL_DIR", "data/swe_verify/labels_img")
INSTS = os.environ.get("INSTS", "data/swe_verify/validate10.json")

SYS = (
    "You are a trajectory graph-labeling assistant. Below is a COMPLETE "
    "coding-agent trajectory that worked on a GitHub issue. For EVERY "
    "assistant message, judge whether that step's action (tool call / "
    "reasoning) was NECESSARY for solving the issue. Strict criteria - "
    "label 0 if ANY of these hold: (a) it reads/explores something never "
    "used by later steps; (b) it repeats an action whose result was already "
    "obtained; (c) it was a failed/retried attempt superseded by a later "
    "successful one; (d) its edits were later reverted or overwritten; "
    "(e) its edit changes code in the OPPOSITE direction of the official "
    "fix or is unrelated to it ([GOLD-REVERSE] or [GOLD-NEUTRAL]). "
    "Label 1 with priority when an edit matches the official fix direction "
    "([GOLD-MATCH]) or the step's output was actually used by subsequent "
    "steps / appears in the final fix. Do NOT give everything 1 just "
    "because the trajectory exists. Output ONLY a JSON object mapping "
    "assistant msg ids to 0/1, e.g. {\"msg02\":1,\"msg04\":0}."
)


def parse_gold_hunks(patch_text):
    """Extract removed (-) and added (+) lines from a unified diff."""
    minus, plus = set(), set()
    cur_minus, cur_plus = [], []
    flush = False
    for line in patch_text.splitlines():
        if line.startswith("@@"):
            flush = True
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-") and not line.startswith("---"):
            minus.add(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            plus.add(line[1:].strip())
    return minus, plus


def classify_edit(old, new, minus, plus):
    o, n = old.strip(), new.strip()
    if not o and not n:
        return "[GOLD-NEUTRAL]"
    if o in minus and n in plus:
        return "[GOLD-MATCH]"
    if o in plus and n in minus:
        return "[GOLD-REVERSE]"
    # partial: one side matches
    if o in minus or n in plus:
        return "[GOLD-MATCH]"
    if o in plus or n in minus:
        return "[GOLD-REVERSE]"
    return "[GOLD-NEUTRAL]"


def render(rec, gold):
    minus, plus = parse_gold_hunks(gold.get("patch", ""))
    parts = []
    msgs = rec["messages"]
    for i, m in enumerate(msgs):
        role = m.get("role")
        seg = "### msg%02d [%s]\n" % (i, role)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"raw": fn.get("arguments")}
            seg += "[tool_call] %s %s\n" % (name,
                                            str(args)[:400])
            if name == "edit":
                tag = classify_edit(str(args.get("old", "")),
                                    str(args.get("new", "")), minus, plus)
                seg += "[gold-verdict] %s\n" % tag
        c = m.get("content")
        if c:
            cs = str(c)
            if len(cs) > 4000:
                cs = cs[:4000] + "\n...(truncated)"
            seg += "[content] " + cs + "\n"
        parts.append(seg)
    return "\n".join(parts)


def call_judge(transcript, outcome, max_retry=4):
    note = ("Trajectory final outcome: %s. " % outcome if outcome else "")
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": note + "Trajectory:\n\n" + transcript},
        ],
        "temperature": 0.0,
        "max_tokens": 3000,
        "stream": False,
    }
    for attempt in range(max_retry):
        try:
            req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + KEY})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read().decode())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            print("  judge err:", str(e)[:120], flush=True)
            time.sleep(10)
    return None


def load_insts():
    if INSTS.endswith(".jsonl"):
        return [json.loads(l) for l in open(INSTS, encoding="utf-8")]
    return json.load(open(INSTS, encoding="utf-8"))


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    gold_map = {x["instance_id"]: x for x in load_insts()}
    files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.json")))
    for fp in files:
        base = os.path.basename(fp)
        out = os.path.join(LABEL_DIR, base)
        if os.path.exists(out):
            continue
        rec = json.load(open(fp, encoding="utf-8"))
        gold = gold_map.get(rec.get("instance_id"), {})
        transcript = render(rec, gold)
        verdict = call_judge(transcript, rec.get("outcome", ""))
        if verdict is None:
            print("skip %s (judge failed)" % base, flush=True)
            continue
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"instance_id": rec.get("instance_id"),
                       "sample": rec.get("sample"),
                       "outcome": rec.get("outcome"),
                       "verdict_raw": verdict}, f, ensure_ascii=False, indent=2)
        print("labeled %s" % base, flush=True)


if __name__ == "__main__":
    main()
