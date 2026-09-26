#!/usr/bin/env python3
"""Build transition-schema dataset from successful rollouts.

One transition per tool call, from pass rollouts ONLY (outcome == "pass").
Schema (per mentor's spec):
  task_id, traj_id, step_index
  state:  issue / history / current_diff / last_observation / test_status
  action: assistant_text / tool / args
  next_state: observation / diff / test_status
  trajectory_success, keep, reason

current_diff / test_status / keep / reason are filled by trajectory_prune.py
(replay-verified). This stage emits the static skeleton.
"""
import glob
import json
import os

ROLLOUT_DIR = os.environ.get("OUT_DIR", "data/swe_verify/rollouts_full")
LABEL_DIR = os.environ.get("LABEL_DIR", "data/swe_verify/labels_full")
OUT = os.environ.get("OUT_TRANS", "data/swe_verify/transitions_v1.jsonl")
HIST_STEPS = 3
HIST_CHARS = 400


def parse_args(args):
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except Exception:
        return {}


def parse_observation(content):
    # tool message content looks like: Tool bash({...}): <output>
    if isinstance(content, str) and "): " in content:
        return content.split("): ", 1)[1]
    return str(content or "")


def extract_steps(msgs):
    """Return list of steps: each = (assistant_text, list[(tool, args,
    observation)]) aligned by tool_call_id."""
    steps = []
    pending = {}
    for m in msgs:
        r = m.get("role")
        if r == "assistant":
            text = m.get("content") or ""
            tcs = m.get("tool_calls") or []
            actions = []
            for i, tc in enumerate(tcs):
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = parse_args(fn.get("arguments"))
                actions.append({
                    "assistant_text": text if i == 0 else "",
                    "tool": name,
                    "args": args,
                    "tc_id": tc.get("id", ""),
                    "observation": "",
                })
            steps.append(actions)
            for a in actions:
                if a["tc_id"]:
                    pending[a["tc_id"]] = a
        elif r == "tool":
            tid = m.get("tool_call_id")
            if tid in pending:
                pending[tid]["observation"] = parse_observation(
                    m.get("content"))
    return steps


def hist_summary(steps, upto):
    parts = []
    for acts in steps[max(0, upto - HIST_STEPS):upto]:
        for a in acts:
            if a["assistant_text"]:
                parts.append(a["assistant_text"][:HIST_CHARS])
            parts.append("%s(%s)" % (a["tool"],
                                     json.dumps(a["args"])[:120]))
    return "\n".join(parts)[-3000:]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/Qwen3.5-4B",
                                        trust_remote_code=True)
    n_traj = 0
    n_trans = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for fp in sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.json"))):
            rec = json.load(open(fp, encoding="utf-8"))
            if rec.get("outcome") != "pass":
                continue
            base = os.path.basename(fp)
            lab = json.load(open(os.path.join(LABEL_DIR, base),
                                 encoding="utf-8"))
            msgs = rec.get("messages", [])
            steps = extract_steps(msgs)
            if not steps:
                continue
            n_traj += 1
            issue = rec.get("problem_statement", "")
            traj_id = "%s__s%s" % (rec["instance_id"], rec.get("sample"))
            last_obs = ""
            step_idx = 0
            for si, acts in enumerate(steps):
                for a in acts:
                    if not a["tool"]:
                        continue
                    trans = {
                        "task_id": rec["instance_id"],
                        "traj_id": traj_id,
                        "step_index": step_idx,
                        "state": {
                            "issue": issue[:1500],
                            "history": hist_summary(steps, si),
                            "current_diff": "",
                            "last_observation": last_obs[:1500],
                            "test_status": "",
                        },
                        "action": {
                            "assistant_text": a["assistant_text"][:1500],
                            "tool": a["tool"],
                            "args": a["args"],
                        },
                        "next_state": {
                            "observation": a["observation"][:2000],
                            "diff": "",
                            "test_status": "",
                        },
                        "trajectory_success": 1,
                        "keep": 1,
                        "reason": "",
                    }
                    fout.write(json.dumps(trans, ensure_ascii=False) + "\n")
                    n_trans += 1
                    last_obs = a["observation"][:1500]
                step_idx += 1
    print("trajectories: %d | transitions: %d -> %s"
          % (n_traj, n_trans, OUT))


if __name__ == "__main__":
    main()
