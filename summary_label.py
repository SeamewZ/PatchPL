#!/usr/bin/env python3
"""Summary report for the labeling method validation."""
import json
from collections import Counter

tot_traj = 0
tot_edges = 0
edge_labels = Counter()
outcomes = Counter()
insts = []
for line in open("data/swe_verify/task_trees.jsonl", encoding="utf-8"):
    d = json.loads(line)
    n_traj = d["n_traj"]
    tot_traj += n_traj
    lbls = []
    outs = []
    for sid, meta in d["trajectories"].items():
        lbls += meta["labels"]
        outs.append(meta["outcome"])
    c = Counter(lbls)
    insts.append((d["instance_id"], n_traj, len(lbls), c[1], c[0], outs))
    tot_edges += len(lbls)
    edge_labels.update(lbls)
    outcomes.update(outs)

print("=== LABELING METHOD VALIDATION SUMMARY ===")
print("instances: %d, trajectories: %d, labeled edges: %d" %
      (len(insts), tot_traj, tot_edges))
print("label 1: %d (%.0f%%), label 0: %d (%.0f%%)" %
      (edge_labels[1], 100 * edge_labels[1] / tot_edges,
       edge_labels[0], 100 * edge_labels[0] / tot_edges))
print("outcomes:", dict(outcomes))
print()
for iid, nt, ne, n1, n0, outs in insts:
    print("%-32s traj=%d edges=%d 1=%d 0=%d %s" % (iid, nt, ne, n1, n0, outs))
