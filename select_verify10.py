#!/usr/bin/env python3
"""Select 10 verified instances for labeling method validation (mixed difficulty, cross-repo)."""
import json

with open("data/swe_verify/swe_verify.jsonl", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]

print("total verified:", len(rows))
print("fields:", list(rows[0].keys()))

# difficulty buckets if available
easy = [r["instance_id"] for r in rows]  # fallback
for name in ["swe_verify_easy.json", "swe_verify_medium.json", "swe_verify_hard.json"]:
    try:
        d = json.load(open(f"data/swe_verify/{name}"))
        ids = list(d) if isinstance(d, dict) else [x["instance_id"] for x in d]
        print(name, len(ids))
    except Exception as e:
        print(name, "n/a", e)

# pick 10: prefer repo diversity, avoid ones we already know are env-broken
prefer_skip = {"django__django-10097"}  # known env-heavy, keep for later
chosen = []
seen_repo = {}
for r in rows:
    iid = r["instance_id"]
    repo = r["repo"]
    if iid in prefer_skip:
        continue
    if repo in seen_repo and seen_repo[repo] >= 2:
        continue
    chosen.append(r)
    seen_repo[repo] = seen_repo.get(repo, 0) + 1
    if len(chosen) >= 10:
        break

json.dump(chosen, open("data/swe_verify/validate10.json", "w"), indent=2)
for r in chosen:
    print("-", r["instance_id"], r["repo"], r["base_commit"][:10])
print("written: data/swe_verify/validate10.json")
