#!/usr/bin/env python3
"""Label rollout trajectories with DeepSeek: each assistant msg -> 0/1 useful.

Reuses the prompt criteria from deepseek_batch_judge.py. Reads rollouts from
data/swe_verify/rollouts/, writes per-trajectory verdicts to
data/swe_verify/labels/.
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

API = "https://api.deepseek.com/chat/completions"
KEY = re.sub(r"[^\x20-\x7e]", "",
             open(os.path.expanduser("~/.deepseek_key"), encoding="utf-8",
                  errors="replace").read()).strip()

ROLLOUT_DIR = "data/swe_verify/rollouts"
LABEL_DIR = "data/swe_verify/labels"

SYS = (
    "You are a trajectory graph-labeling assistant. Below is a COMPLETE "
    "coding-agent trajectory that worked on a GitHub issue. For EVERY "
    "assistant message, judge whether that step's action (tool call / "
    "reasoning) was NECESSARY for solving the issue. Strict criteria - "
    "label 0 if ANY of these hold: (a) it reads/explores something never "
    "used by later steps; (b) it repeats an action whose result was already "
    "obtained; (c) it was a failed/retried attempt superseded by a later "
    "successful one; (d) its edits were later reverted or overwritten. "
    "Label 1 only if the step's output was actually used by subsequent "
    "steps or appears in the final fix. Do NOT give everything 1 just "
    "because the trajectory exists. Output ONLY a JSON object mapping "
    "assistant msg ids to 0/1, e.g. {\"msg02\":1,\"msg04\":0}."
)


def render(rec):
    parts = []
    msgs = rec["messages"]
    for i, m in enumerate(msgs):
        role = m.get("role")
        seg = "### msg%02d [%s]\n" % (i, role)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            seg += "[tool_call] %s %s\n" % (fn.get("name", "?"),
                                            str(fn.get("arguments", ""))[:400])
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
        except urllib.error.HTTPError as e:
            print("  judge http %s: %s" % (e.code, e.read().decode()[:150]),
                  flush=True)
            time.sleep(10)
        except Exception as e:
            print("  judge err:", str(e)[:120], flush=True)
            time.sleep(15)
    return None


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.json")))
    if len(sys.argv) > 1:
        files = [f for f in files if sys.argv[1] in f]
    for fp in files:
        rec = json.load(open(fp, encoding="utf-8"))
        iid = rec["instance_id"]
        sid = rec["sample"]
        out = os.path.join(LABEL_DIR, "%s__s%d.json" % (iid, sid))
        if os.path.exists(out):
            print("skip", os.path.basename(out))
            continue
        transcript = render(rec)
        txt = call_judge(transcript, rec.get("outcome"))
        verdicts = {}
        if txt:
            mm = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            if mm:
                try:
                    verdicts = json.loads(mm.group(0))
                except Exception:
                    verdicts = {}
        # normalize + default
        final = {}
        for i, m in enumerate(rec["messages"]):
            if m.get("role") == "assistant":
                k = "msg%02d" % i
                v = verdicts.get(k, 1)
                final[k] = int(v) if v in (0, 1) else 1
        json.dump({"instance_id": iid, "sample": sid, "verdicts": final},
                  open(out, "w"), indent=2)
        n0 = sum(1 for v in final.values() if v == 0)
        n1 = len(final) - n0
        print("%s: %d msgs, 1=%d 0=%d" % (os.path.basename(out), len(final),
                                           n1, n0), flush=True)


if __name__ == "__main__":
    main()
