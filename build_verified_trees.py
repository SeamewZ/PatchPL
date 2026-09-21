#!/usr/bin/env python3
"""Build per-instance prefix trees from rollouts + DeepSeek labels.

Edge = assistant message's action signature (tool name + normalized args hash).
Edge label = DeepSeek verdict of that message.
Node = conversation state (shared prefix merged across trajectories).
"""
import glob
import hashlib
import json
import os
from collections import defaultdict

ROLLOUT_DIR = "data/swe_verify/rollouts"
LABEL_DIR = "data/swe_verify/labels"
OUT_FILE = "data/swe_verify/task_trees.jsonl"


def sig_of(m):
    tcs = m.get("tool_calls") or []
    if not tcs:
        c = str(m.get("content", ""))[:80].replace("\n", " ")
        return "text:" + hashlib.md5(c.encode()).hexdigest()[:8]
    parts = []
    for tc in tcs:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {"raw": fn.get("arguments")}
        norm = json.dumps(args, sort_keys=True)[:120]
        parts.append("%s:%s" % (fn.get("name", "?"),
                                hashlib.md5(norm.encode()).hexdigest()[:8]))
    return "+".join(parts)


def main():
    rolls = {}
    for fp in glob.glob(os.path.join(ROLLOUT_DIR, "*.json")):
        rec = json.load(open(fp, encoding="utf-8"))
        rolls.setdefault(rec["instance_id"], []).append(rec)

    out_lines = []
    report = []
    for iid, recs in sorted(rolls.items()):
        labels = {}
        for rec in recs:
            lp = os.path.join(LABEL_DIR, "%s__s%d.json" % (iid, rec["sample"]))
            if os.path.exists(lp):
                labels[rec["sample"]] = json.load(open(lp))["verdicts"]

        # build tree: children keyed by signature, shared prefix merge
        root = {"n": 0, "children": [], "term": []}
        traj_meta = {}
        for rec in sorted(recs, key=lambda r: r["sample"]):
            sid = rec["sample"]
            verdicts = labels.get(sid, {})
            node = root
            edge_labels = []
            for i, m in enumerate(rec["messages"]):
                if m.get("role") != "assistant":
                    continue
                sig = sig_of(m)
                v = verdicts.get("msg%02d" % i, 1)
                edge_labels.append(v)
                child = None
                for e in node["children"]:
                    if e["tool"] == sig:
                        child = e
                        break
                if child is None:
                    child = {"tool": sig, "label": v, "msg": [iid, sid, i],
                             "node": {"n": 0, "children": [], "term": []}}
                    node["children"].append(child)
                else:
                    if v == 1 and child["label"] == 0:
                        child["label"] = 1  # any pass-style label wins
                node = child["node"]
            node["term"].append(rec.get("outcome", "unknown"))
            traj_meta[str(sid)] = {"outcome": rec.get("outcome", "unknown"),
                                   "n_msgs": len(rec["messages"]),
                                   "n_edges": len(edge_labels),
                                   "labels": edge_labels}

        def size(n):
            return 1 + sum(size(e["node"]) for e in n["children"])

        tree = {"n": 0, "children": root["children"]}
        out_lines.append(json.dumps({
            "instance_id": iid,
            "repo": recs[0]["repo"],
            "n_traj": len(recs),
            "trajectories": traj_meta,
            "tree": tree,
        }, ensure_ascii=False))
        total1 = sum(lbl.count(1) for lbl in
                     [traj_meta[k]["labels"] for k in traj_meta])
        total0 = sum(lbl.count(0) for lbl in
                     [traj_meta[k]["labels"] for k in traj_meta])
        report.append("%s: traj=%d nodes=%d edges 1=%d 0=%d outcomes=%s"
                      % (iid, len(recs), size(root),
                         total1, total0,
                         [traj_meta[k]["outcome"] for k in traj_meta]))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for l in out_lines:
            f.write(l + "\n")
    print("=== per-instance report ===")
    for r in report:
        print(r)
    print("trees written: %s" % OUT_FILE)


if __name__ == "__main__":
    main()
