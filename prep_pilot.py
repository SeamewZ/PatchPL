"""Clone repos for first N smith test instances at their exact commits."""
import json
import os
import subprocess

N = int(os.environ.get("N_EVAL", "20"))

rows = []
with open("data/swe_smith_test250.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
rows = rows[:N]

os.makedirs("repos", exist_ok=True)
done = []
for d in rows:
    slug = d["repo"].split("/")[-1]          # org__repo.commithash
    rest = slug.rsplit(".", 1)[0]            # org__repo
    if "__" in rest:
        org, repo = rest.split("__", 1)
    else:
        org, repo = rest, rest
    commit = slug.rsplit(".", 1)[1]
    dest = os.path.abspath(os.path.join("repos", slug))
    if os.path.isdir(dest):
        print("exists:", slug)
    else:
        url = "https://github.com/%s/%s.git" % (org, repo)
        print("clone:", org, repo, commit)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, dest],
            capture_output=True, text=True, timeout=900)
        subprocess.run(["git", "-C", dest, "checkout", commit],
                       capture_output=True, text=True, timeout=600)
    done.append((d["instance_id"], slug, d["image_name"]))

with open("/tmp/pilot_list.txt", "w") as f:
    for iid, slug, img in done:
        f.write("%s\t%s\t%s\n" % (iid, slug, img))
print("DONE clones:", len(done))
