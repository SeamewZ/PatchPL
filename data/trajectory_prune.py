#!/usr/bin/env python3
"""Replay-verified trajectory pruning.

For every PASS rollout, replay its tool-call steps inside the official eval
container. Greedily try removing each step (reverse order); a removal is
accepted only if replaying the remaining steps yields the same git diff, or
the resulting state still passes FAIL_TO_PASS tests.

This replaces the per-step "useful" verdict with a true "removal preserves
success" label, avoiding pseudo-edges from naive message deletion.

Output: prune_v1.jsonl, one record per trajectory:
  {traj_id, instance_id, steps: [{tool, keep, reason, diff_after, test_status}]}

Env: SHARD / NUM for parallel workers.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import uuid

ROLLOUT_DIR = "data/swe_verify/rollouts_full"
TRAIN_FILE = "data/swe_verify/split_train.jsonl"
OUT_FILE = "data/swe_verify/prune_v1.jsonl"
SHARD = int(os.environ.get("SHARD", "0"))
NUM = int(os.environ.get("NUM", "1"))
INSTS = os.environ.get("INSTS", "")
TEST_TIMEOUT = 900
ACT = "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && cd /testbed"


def sh(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return "(timed out)"


def img_of(ex):
    org = ex["repo"].split("/")[0]
    repo = ex["repo"].split("/")[1]
    pr = ex["instance_id"].split("-")[-1]
    return "swebench/sweb.eval.x86_64.%s_1776_%s-%s" % (org, repo, pr)


def exec_bash(cid, cmd, timeout=180):
    tag = uuid.uuid4().hex[:8]
    hostf = "/tmp/pr_%s.sh" % tag
    contf = "/tmp/pr_%s.sh" % tag
    with open(hostf, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n%s\n%s\n" % (ACT, cmd))
    sh("docker cp %s %s:%s" % (hostf, cid, contf), timeout=60)
    try:
        out = sh("docker exec %s bash %s" % (cid, contf), timeout=timeout)
    except Exception:
        out = "(bash timed out)"
    try:
        os.remove(hostf)
    except OSError:
        pass
    return out


def run_edit(cid, args):
    p = str(args.get("path", "")).replace("/testbed/", "")
    old = str(args.get("old", ""))
    new = str(args.get("new", ""))
    tag = uuid.uuid4().hex[:8]
    hostf = "/tmp/pr_edit_%s" % tag
    sh('docker cp %s:"/testbed/%s" %s 2>/dev/null' % (cid, p, hostf),
       timeout=60)
    if not os.path.isfile(hostf):
        return False
    with open(hostf, encoding="utf-8", errors="replace") as f:
        src = f.read()
    if old not in src:
        os.remove(hostf)
        return False
    with open(hostf, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new, 1))
    sh('docker cp %s %s:"/testbed/%s"' % (hostf, cid, p), timeout=60)
    os.remove(hostf)
    return True


def parse_args(args):
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except Exception:
        return {}


def extract_steps(msgs):
    """List of (tool, args) in execution order."""
    steps = []
    pending = {}
    for m in msgs:
        r = m.get("role")
        if r == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = parse_args(fn.get("arguments"))
                tid = tc.get("id", "")
                rec = {"tool": name, "args": args}
                steps.append(rec)
                if tid:
                    pending[tid] = rec
        elif r == "tool":
            pass  # observations not needed for replay
    return steps


def reset_repo(cid):
    exec_bash(cid, "git reset --hard -q 2>/dev/null; "
                   "git clean -fdq 2>/dev/null; true", timeout=120)


def apply_step(cid, step):
    tool = step["tool"]
    args = step["args"]
    if tool == "bash":
        exec_bash(cid, str(args.get("command", "")), timeout=180)
    elif tool == "edit":
        run_edit(cid, args)
    # read: no state change


def get_diff(cid):
    return exec_bash(cid, "git diff", timeout=120)


def convert_test(t):
    t = t.strip()
    m = re.match(r"^(.+?)\s+\(([\w.]+?)(?:\.(\w+))?\)$", t)
    if m:
        name, mod, cls = m.group(1), m.group(2), m.group(3)
        return ("%s.%s.%s" % (mod, cls, name) if cls
                else "%s.%s" % (mod, name))
    return t


def run_tests(cid, ex):
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


def replay(cid, steps):
    """Replay steps from clean repo; return final git diff."""
    reset_repo(cid)
    for s in steps:
        apply_step(cid, s)
    return get_diff(cid)


def process_traj(fp, ex):
    rec = json.load(open(fp, encoding="utf-8"))
    steps = extract_steps(rec.get("messages", []))
    if not steps:
        return None
    img = img_of(ex)
    cid = "pr_%s" % re.sub(r"[^a-zA-Z0-9_.-]", "_",
                           rec["instance_id"])[:40]
    sh("docker rm -f %s >/dev/null 2>&1" % cid)
    sh("docker run -d --name %s -w /testbed %s sleep infinity"
       % (cid, img), timeout=180)

    n = len(steps)
    keep = [True] * n
    reason = [""] * n
    diff_after = [""] * n

    # full replay: baseline diff + per-step diffs
    reset_repo(cid)
    for i, s in enumerate(steps):
        apply_step(cid, s)
        diff_after[i] = get_diff(cid)
    d_full = diff_after[-1] if n else ""

    # greedy removal, reverse order
    for i in range(n - 1, -1, -1):
        if steps[i]["tool"] == "read":
            # read never changes repo state: removal is always safe
            keep[i] = False
            reason[i] = "removal verified: read does not change state"
            continue
        cand = [s for j, s in enumerate(steps) if j != i and keep[j]]
        d = replay(cid, cand)
        if d == d_full:
            keep[i] = False
            reason[i] = "removal verified: same diff"
            continue
        # diff changed: run tests on candidate state
        status = run_tests(cid, ex)
        if status == "pass":
            keep[i] = False
            reason[i] = "removal verified: tests still pass"
        else:
            reason[i] = "required for pass (tests %s after removal)"
            reason[i] %= status
        # restore full state for next comparisons? we re-replay each time
    sh("docker rm -f %s >/dev/null 2>&1" % cid)
    return {
        "traj_id": "%s__s%s" % (rec["instance_id"], rec.get("sample")),
        "instance_id": rec["instance_id"],
        "n_steps": n,
        "n_kept": sum(keep),
        "steps": [{"tool": steps[i]["tool"],
                   "keep": int(keep[i]),
                   "reason": reason[i],
                   "diff_after": diff_after[i][:3000]} for i in range(n)],
    }


def main():
    train = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8")]
    by_iid = {t["instance_id"]: t for t in train}
    if INSTS:
        only = set(INSTS.split(","))
    else:
        only = None

    files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.json")))
    todo = []
    for fp in files:
        rec = json.load(open(fp, encoding="utf-8"))
        if rec.get("outcome") != "pass":
            continue
        if only and rec["instance_id"] not in only:
            continue
        if rec["instance_id"] not in by_iid:
            continue
        todo.append(fp)
    todo = todo[SHARD::NUM]
    print("worker %d/%d: %d trajectories" % (SHARD, NUM, len(todo)),
          flush=True)

    done_ids = set()
    if os.path.exists(OUT_FILE):
        for line in open(OUT_FILE, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["traj_id"])
            except Exception:
                pass

    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for fp in todo:
            rec = json.load(open(fp, encoding="utf-8"))
            tid = "%s__s%s" % (rec["instance_id"], rec.get("sample"))
            if tid in done_ids:
                continue
            t0 = time.time()
            try:
                res = process_traj(fp, by_iid[rec["instance_id"]])
            except Exception as e:
                print("FAIL %s: %s" % (tid, str(e)[:200]), flush=True)
                continue
            if res:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
            print("done %s kept %d/%d in %.0fs"
                  % (tid, res["n_kept"], res["n_steps"],
                     time.time() - t0), flush=True)
    print("PRUNE-DONE", flush=True)


if __name__ == "__main__":
    main()
