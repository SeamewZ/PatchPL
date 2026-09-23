#!/usr/bin/env python3
"""Build masked-SFT training data from rollouts + DeepSeek labels.

Clean-trajectory construction: keep only assistant messages labeled useful
(label=1) plus their tool results; render with the Qwen chat template.
Two outputs:
- data/swe_verify/sft_clean_rendered.jsonl  (training input)
- data/swe_verify/sft_masked_train.jsonl    (full traces + mask, for audit)
"""
import glob
import json
import os
import re

ROLLOUT_DIR = os.environ.get("OUT_DIR", "data/swe_verify/rollouts_img")
LABEL_DIR = os.environ.get("LABEL_DIR", "data/swe_verify/labels_img")
OUT_CLEAN = os.environ.get("OUT_CLEAN",
                            "data/swe_verify/sft_clean_rendered.jsonl")
OUT_FULL = os.environ.get("OUT_FULL",
                           "data/swe_verify/sft_masked_train.jsonl")
MODEL_PATH = "models/Qwen3.5-4B"


def parse_verdict(raw):
    m = re.search(r"\{.*\}", raw)
    if not m:
        return {}
    d = json.loads(m.group(0))
    out = {}
    for k, v in d.items():
        kk = re.search(r"(\d+)", k)
        if kk:
            out[int(kk.group(1))] = int(v)
    return out


def build_clean(msgs, verdict):
    """Keep system/user, assistant msgs with verdict=1, their tool replies.
    Converts tool-call arguments from JSON strings to dicts (required by the
    Qwen3.5 chat template)."""
    keep_ids = set()
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and verdict.get(i, 0) == 1:
            for tc in m.get("tool_calls") or []:
                if tc.get("id"):
                    keep_ids.add(tc["id"])

    def fix_args(tc):
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        if isinstance(fn.get("arguments"), str):
            try:
                fn["arguments"] = json.loads(fn["arguments"])
            except Exception:
                pass
        tc["function"] = fn
        return tc

    clean = []
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r in ("system", "user"):
            clean.append(dict(m))
        elif r == "assistant":
            if verdict.get(i, 0) == 1:
                cm = dict(m)
                if cm.get("tool_calls"):
                    cm["tool_calls"] = [fix_args(tc)
                                        for tc in cm["tool_calls"]]
                clean.append(cm)
        elif r == "tool":
            if m.get("tool_call_id") in keep_ids:
                clean.append(dict(m))
    return clean


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    rows_full = []
    n_clean = 0
    n1 = 0
    na = 0
    with open(OUT_CLEAN, "w", encoding="utf-8") as fc:
        for fp in sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.json"))):
            base = os.path.basename(fp)
            lab = os.path.join(LABEL_DIR, base)
            if not os.path.exists(lab):
                continue
            rec = json.load(open(fp, encoding="utf-8"))
            labd = json.load(open(lab, encoding="utf-8"))
            verdict = parse_verdict(labd.get("verdict_raw", ""))
            msgs = rec.get("messages", [])
            asst_idx = [i for i, m in enumerate(msgs)
                        if m.get("role") == "assistant"]
            mask = [int(verdict.get(i, 0)) for i in asst_idx]
            rows_full.append({
                "instance_id": rec.get("instance_id"),
                "sample": rec.get("sample"),
                "outcome": rec.get("outcome"),
                "messages": msgs,
                "assistant_idx": asst_idx,
                "mask": mask,
            })
            n1 += sum(mask)
            na += len(mask)
            clean = build_clean(msgs, verdict)
            if len([m for m in clean if m["role"] == "assistant"]) == 0:
                continue
            try:
                rendered = tok.apply_chat_template(clean, tokenize=False,
                                                   add_generation_prompt=False)
            except Exception as e:
                print("render fail %s: %s" % (base, str(e)[:120]), flush=True)
                continue
            fc.write(json.dumps({
                "instance_id": rec.get("instance_id"),
                "sample": rec.get("sample"),
                "outcome": rec.get("outcome"),
                "rendered": rendered,
            }, ensure_ascii=False) + "\n")
            n_clean += 1

    with open(OUT_FULL, "w", encoding="utf-8") as f:
        for r in rows_full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("clean samples=%d | assistant msgs=%d useful=%d (%.1f%%)"
          % (n_clean, na, n1, 100.0 * n1 / max(na, 1)))
    print("outputs: %s | %s" % (OUT_CLEAN, OUT_FULL))


if __name__ == "__main__":
    main()
