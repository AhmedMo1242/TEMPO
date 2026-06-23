#!/usr/bin/env python3
"""Generate dataset files for TEMPO from skeleton PKL + CTM files."""
import os
import json
import csv
import pickle
from collections import defaultdict

PKL_PATH = "/kaggle/input/mslr-task1/pose_data_isharah2000_hands_lips_body_phase2_SI.pkl"
CTM_DIR = "/kaggle/input/mslr-track-2-5"
OUT_DIR = "/kaggle/working/TEMPO/datasets/mslr2026"

# 1. Parse CTM files to get per-video gloss sequences
ctm_files = [f for f in os.listdir(CTM_DIR) if f.endswith(".ctm")]
print(f"Found {len(ctm_files)} CTM files: {ctm_files}")

# Use the seq CTM as primary (best quality per dev.txt)
primary_ctm = "output-hypothesis-seq-dev.ctm"
video_glosses = defaultdict(list)
all_glosses = set()

for ctm_name in ctm_files:
    ctm_path = os.path.join(CTM_DIR, ctm_name)
    with open(ctm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            vid = parts[0]
            gloss = parts[4]
            all_glosses.add(gloss)
            if ctm_name == primary_ctm:
                video_glosses[vid].append(gloss)

print(f"Unique glosses: {len(all_glosses)}")
print(f"Videos with seq glosses: {len(video_glosses)}")

# 2. Load PKL to get all video IDs
print("Loading skeleton PKL...")
with open(PKL_PATH, "rb") as f:
    kps = pickle.load(f)
print(f"Total skeleton videos: {len(kps)}")

# Filter to 02_ prefix videos (dev set)
dev_videos = sorted([k for k in kps.keys() if k.startswith("02_")])
print(f"Dev videos (02_ prefix): {len(dev_videos)}")

# 3. Build gloss dict (blank = 0)
sorted_glosses = sorted(all_glosses)
gloss2id = {}
id2gloss = {}
for i, g in enumerate(sorted_glosses):
    idx = i + 1  # 0 is blank
    gloss2id[g] = {"index": idx, "count": 0}
    id2gloss[str(idx)] = {"gloss": g, "count": 0}

# Count glosses
for vids in video_glosses.values():
    for g in vids:
        if g in gloss2id:
            gloss2id[g]["count"] += 1
            id2gloss[str(gloss2id[g]["index"])]["count"] += 1

gloss_dict = {"gloss2id": gloss2id, "id2gloss": id2gloss}

gloss_dict_path = os.path.join(OUT_DIR, "si_gloss_dict.json")
with open(gloss_dict_path, "w", encoding="utf-8") as f:
    json.dump(gloss_dict, f, ensure_ascii=False, indent=2)
print(f"Wrote {gloss_dict_path} ({len(gloss2id)} glosses)")

# 4. Build info JSON for dev set
inputs_list = []
for vid in dev_videos:
    gloss_seq = " ".join(video_glosses.get(vid, []))
    inputs_list.append({
        "video_id": vid,
        "gloss_sequence": gloss_seq,
        "original_info": vid,
        "signer": "Signer00",
    })

info_path = os.path.join(OUT_DIR, "si_dev_info.json")
with open(info_path, "w", encoding="utf-8") as f:
    json.dump(inputs_list, f, ensure_ascii=False, indent=1)
print(f"Wrote {info_path} ({len(inputs_list)} entries)")

# 5. Also create train_info.json (same as dev for now, needed by some paths)
train_info_path = os.path.join(OUT_DIR, "si_train_info.json")
with open(train_info_path, "w", encoding="utf-8") as f:
    json.dump(inputs_list, f, ensure_ascii=False, indent=1)
print(f"Wrote {train_info_path}")

# 6. Generate best_dev_seq.csv (for extract.py's build_gloss_map_from_csv)
csv_path = os.path.join(OUT_DIR, "best_dev_seq.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "gloss"])
    for vid in dev_videos:
        gloss_seq = " ".join(video_glosses.get(vid, []))
        if gloss_seq:
            writer.writerow([vid, gloss_seq])
print(f"Wrote {csv_path}")

print("\nDone!")
