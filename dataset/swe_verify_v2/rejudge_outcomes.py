#!/usr/bin/env python3
"""Re-judge outcomes for existing rollout JSONs (fixes the unknown bug).

For each rollout in data/swe_verify/rollouts_img/ with outcome != pass/fail,
spin a fresh container and run FAIL_TO_PASS with corrected parsing.
"""
import glob
import json
import os
import re
import subprocess
import time

ROLLOUT_DIR = os.environ.get("OUT_DIR", "data/swe_verify/rollouts_img")
INSTS = os.environ.get("INSTS", "data/swe_verify/validate10.json")
TEST_TIMEOUT = 600
ACT = "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && cd /testbed"


def sh(cmd, timeout=300):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def img_of(ex):
    org = ex["repo"].split("/")[0]
    repo = ex["repo"].split("/")[1]
    pr = ex["instance_id"].split("-")[-1]
    return "swebench/sweb.eval.x86_64.%s_1776_%s-%s" % (org, repo, pr)


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
        out = run(cid, "python tests/runtests.py --verbosity 1 %s" % targs)
        if "FAILED" in out:
            return "fail"
        if "OK" in out:
            return "pass"
        return "unknown"
    out = run(cid, "python -m pytest -q %s" % targs)
    m_fail = re.search(r"(\d+) failed", out)
    m_pass = re.search(r"(\d+) passed", out)
    if m_fail and int(m_fail.group(1)) > 0:
        return "fail"
    if m_pass and int(m_pass.group(1)) > 0:
        return "pass"
    return "unknown"


def run(cid, cmd):
    tag = cid + "_t"
    hostf = "/tmp/%s.sh" % tag
    with open(hostf, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n%s\n%s\n" % (ACT, cmd))
    sh("docker cp %s %s:/tmp/%s.sh" % (hostf, cid, tag))
    out = sh("docker exec %s bash /tmp/%s.sh" % (cid, tag),
             timeout=TEST_TIMEOUT)
    os.remove(hostf)
    return out


def load_insts():
    if INSTS.endswith(".jsonl"):
        return [json.loads(l) for l in open(INSTS, encoding="utf-8")]
    return json.load(open(INSTS, encoding="utf-8"))


def main():
    import sys
    gold = {x["instance_id"]: x for x in load_insts()}
    files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.json")))
    if len(sys.argv) > 2:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
        files = files[lo:hi]
    for fp in files:
        rec = json.load(open(fp, encoding="utf-8"))
        if rec.get("outcome") in ("pass", "fail"):
            continue
        ex = gold.get(rec.get("instance_id"))
        if ex is None:
            continue
        cid = "rj_%s" % re.sub(r"[^a-zA-Z0-9_.-]", "_",
                                rec["instance_id"])
        img = img_of(ex)
        chk = sh("docker image inspect %s >/dev/null 2>&1 && echo OK" % img)
        if "OK" not in chk:
            print("skip %s (no image)" % rec["instance_id"], flush=True)
            continue
        sh("docker rm -f %s >/dev/null 2>&1" % cid)
        sh("docker run -d --name %s -w /testbed %s sleep infinity"
           % (cid, img), timeout=180)
        new = judge(cid, ex)
        sh("docker rm -f %s >/dev/null 2>&1" % cid)
        old = rec.get("outcome")
        if new != old:
            rec["outcome"] = new
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            print("%s: %s -> %s" % (os.path.basename(fp), old, new),
                  flush=True)
        else:
            print("%s: unchanged %s" % (os.path.basename(fp), old),
                  flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
