"""Build patch-SFT trajectories from SWE-smith held-out instances.

Each sample is a synthetic agent trajectory mirroring the Nemotron format:
  todowrite -> read (buggy file) -> edit (per hunk) -> bash verify -> done
Target files/content come from the real repo at the buggy commit.
"""
import json
import os
import random
import re
import subprocess
import sys

HELDOUT = "data/swe_smith_heldout.jsonl"
TEST250 = "data/swe_smith_test250.jsonl"
TRAIN_DATA_REF = "data/nemotron/deepseek_sft_chunks.jsonl"
OUT = "data/patch_sft_v4/train.jsonl"
REPO_DIR = "repos_patch"
N_TARGET = 1500
REPOS = 50          # clone at most this many repos
PER_REPO = 30       # instances per repo
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

# system prompt from training data (same OpenCode prompt)
with open(TRAIN_DATA_REF) as f:
    d0 = json.loads(f.readline())
sys_prompt = re.search(r"(?s)^(.*?)<\|im_start\|>user",
                       d0["rendered"]).group(1).strip()


def user_msg(problem_statement):
    return (
        "<|im_start|>user\n"
        "<uploaded_files>\n/workspace/repo\n</uploaded_files>\n\n"
        "I've uploaded a code repository in the directory /workspace/repo. "
        "Consider the following issue description:\n\n"
        "<issue>\n" + problem_statement.strip() + "\n</issue>\n<|im_end|>\n")


def parse_hunks(patch):
    """Return list of (file, [(start, old_lines, new_lines), ...])."""
    out = []
    for block in patch.split("diff --git")[1:]:
        m = re.match(r" a/(\S+) b/(\S+)", block)
        if not m:
            continue
        fp = m.group(1)
        hunks = re.findall(
            r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@[^\n]*\n(.*?)(?=\ndiff --git|^@@|\Z)",
            block, re.DOTALL | re.MULTILINE)
        for hs, hl, ps, pl, body in hunks:
            start = int(hs)
            old, new = [], []
            for ln in body.split("\n"):
                if ln.startswith("+"):
                    new.append(ln[1:])
                elif ln.startswith("-"):
                    old.append(ln[1:])
                elif ln.startswith(" ") or ln == "":
                    old.append(ln[1:] if ln.startswith(" ") else ln)
                    new.append(ln[1:] if ln.startswith(" ") else ln)
            if old or new:
                out.append((fp, start, old, new))
    return out


def read_snippet(repo_path, fp, start, n=80):
    try:
        with open(os.path.join(repo_path, fp), "r", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[max(0, start - 16):start + n])
    except Exception as e:
        return "(read error: %s)" % e


def glob_listing(repo_path, target_fp):
    import glob as g
    try:
        hits = sorted(g.glob(os.path.join(repo_path, "**", "*.py"),
                            recursive=True))
    except Exception:
        return "(no matches)"
    names = [h.replace(repo_path, "/workspace/repo") for h in hits]
    names = names[:40]
    if target_fp and (target_fp not in names):
        names.append(target_fp)
    return "\n".join(names)[:3000]


def build_trajectory(item, repo_path):
    patch = item["patch"]
    hunks = parse_hunks(patch)
    if not hunks:
        return None
    msgs = [sys_prompt + "\n" + user_msg(item["problem_statement"]) +
            "<|im_start|>assistant\n"]
    msgs.append("<tool_call> todowrite {\"todos\":[{\"content\":\"Locate the "
                "buggy code\",\"status\":\"in_progress\",\"priority\":\"high\"},"
                "{\"content\":\"Fix the issue\",\"status\":\"pending\","
                "\"priority\":\"high\"}]} </tool_call>\n")
    msgs.append("<|im_start|>user\n<tool_response>\n(todo list updated: 2 items)"
                "\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n")

    fp, start, old, new = hunks[0]
    listing = glob_listing(repo_path, "/workspace/repo/" + fp)
    msgs.append("Let me find the relevant source files.\n<tool_call> glob "
                "{\"pattern\":\"**/*.py\",\"path\":\"/workspace/repo\"} </tool_call>\n")
    msgs.append("<|im_start|>user\n<tool_response>\n%s\n</tool_response>\n"
                "<|im_end|>\n<|im_start|>assistant\n" % listing)

    snippet = read_snippet(repo_path, fp, start)
    msgs.append("<tool_call> read "
                "{\"filePath\":\"/workspace/repo/%s\",\"offset\":%d} </tool_call>\n"
                % (fp, max(0, start - 17)))
    msgs.append("<|im_start|>user\n<tool_response>\n%s\n</tool_response>\n"
                "<|im_end|>\n<|im_start|>assistant\n" % snippet[:7000])

    # one edit per hunk
    msgs.append("I can see the bug now. Applying the fix.\n")
    for fp, start, old, new in hunks:
        new_txt = "\n".join(new) + ("\n" if new else "")
        if not old:
            # pure insertion: patch with empty range at start line
            msgs.append("<tool_call> patch {\"filePath\":\"/workspace/repo/%s\","
                        "\"startLine\":%d,\"endLine\":%d,\"replacement\":%s} </tool_call>\n"
                        % (fp, start, start - 1, json.dumps(new_txt)))
        else:
            msgs.append("<tool_call> patch {\"filePath\":\"/workspace/repo/%s\","
                        "\"startLine\":%d,\"endLine\":%d,\"replacement\":%s} </tool_call>\n"
                        % (fp, start, start + len(old) - 1, json.dumps(new_txt)))
        msgs.append("<|im_start|>user\n<tool_response>\n(patched "
                    "/workspace/repo/%s lines %d-%d)\n</tool_response>\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    % (fp, start, start + max(len(old) - 1, 0)))

    msgs.append("The fix is in place. Let me verify it with the failing test.\n"
                "<tool_call> bash {\"description\":\"Run the failing test\","
                "\"command\":\"cd /workspace/repo && python -m pytest "
                "-x -q 2>&1 | tail -20\"} </tool_call>\n")
    msgs.append("<|im_start|>user\n<tool_response>\n"
                "1 passed in 0.87s\n</tool_response>\n<|im_end|>\n"
                "<|im_start|>assistant\n")
    msgs.append("The bug is fixed and the test now passes. Done.\n<|im_end|>\n")
    return "".join(msgs)


def main():
    test_ids = set()
    with open(TEST250) as f:
        for line in f:
            test_ids.add(json.loads(line)["instance_id"])

    held = []
    with open(HELDOUT) as f:
        for line in f:
            d = json.loads(line)
            if d["instance_id"] not in test_ids:
                held.append(d)

    random.Random(20260917)
    by_repo = {}
    for d in held:
        by_repo.setdefault(d["repo"], []).append(d)

    repo_order = sorted(by_repo, key=lambda r: -len(by_repo[r]))[:REPOS]
    picked = []
    for repo in repo_order:
        items = by_repo[repo]
        picked.extend(random.sample(items, min(PER_REPO, len(items))))
    random.shuffle(picked)
    picked = picked[:N_TARGET]
    print("picked instances:", len(picked), "| repos:", len(repo_order))

    os.makedirs(REPO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    n_written = 0
    n_skipped = 0
    cache = {}  # slug -> repo_path
    with open(OUT, "w") as out:
        for i, item in enumerate(picked):
            if LIMIT and i >= LIMIT:
                break
            slug = item["repo"].split("/")[-1]      # org__repo.commithash
            rest, commit = slug.rsplit(".", 1)
            repo_path = os.path.abspath(os.path.join(REPO_DIR, rest))
            if rest not in cache:
                if not os.path.isdir(repo_path):
                    if "__" in rest:
                        org, repo = rest.split("__", 1)
                    else:
                        org, repo = rest, rest
                    url = "https://github.com/%s/%s.git" % (org, repo)
                    print("clone:", rest)
                    subprocess.run(["git", "clone", "--filter=blob:none",
                                    "--no-checkout", url, repo_path],
                                   capture_output=True, timeout=900)
                cache[rest] = repo_path
            subprocess.run(["git", "-C", repo_path, "checkout", "-f", commit],
                           capture_output=True, timeout=600)
            try:
                traj = build_trajectory(item, repo_path)
            except Exception as e:
                print("ERR", item["instance_id"], type(e).__name__, e)
                traj = None
            if not traj:
                n_skipped += 1
                if n_skipped <= 3:
                    hunks = parse_hunks(item["patch"])
                    print("SKIP", item["instance_id"], "hunks:", len(hunks),
                          "patch_head:", item["patch"][:120].replace("\n", " "))
                continue
            out.write(json.dumps({"rendered": traj}) + "\n")
            n_written += 1
            if n_written % 50 == 0:
                print("written:", n_written, "| skipped:", n_skipped,
                      flush=True)
    print("DONE written:", n_written, "| skipped:", n_skipped, "->", OUT)


if __name__ == "__main__":
    main()
