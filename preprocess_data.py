#!/usr/bin/env python3
"""
MSLR 2026 Data Preprocessing Script
Converts PKL + CSV format to JSON format expected by the model
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict
import argparse


def load_skeleton_data(pkl_path):
    """Load skeleton data from PKL file"""
    print(f"Loading skeleton data from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} skeleton sequences")
    return data


def load_csv_annotations(csv_path):
    """Load annotations from CSV file. Handles id|gloss or just id."""
    print(f"Loading annotations from: {csv_path}")
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(f).strip()
        has_gloss = "|" in header or "gloss" in header

        for line in f:
            line = line.strip()
            if not line:
                continue

            if "|" in line:
                video_id, gloss_sequence = line.split("|", 1)
            else:
                video_id = line
                gloss_sequence = ""  # Empty for test set

            try:
                signer, sentence_id = video_id.split("_")
            except ValueError:
                signer, sentence_id = "unknown", video_id

            data.append(
                {
                    "signer": signer,
                    "video_id": video_id,
                    "gloss_sequence": gloss_sequence.strip(),
                    "sentence_id": sentence_id,
                    "original_info": line,
                }
            )

    print(f"Loaded {len(data)} annotations")
    return data


def create_gloss_dict(annotations_data):
    """Create gloss dictionary from annotations data"""
    sign_dict = {}
    for item in annotations_data:
        split_label = item["gloss_sequence"].split()
        for gloss in split_label:
            if gloss not in sign_dict.keys():
                sign_dict[gloss] = 1
            else:
                sign_dict[gloss] += 1

    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {"id2gloss": {}, "gloss2id": {}}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict["gloss2id"][key] = {
            "index": idx + 1,
            "frequency": value,
        }
        save_dict["id2gloss"][idx + 1] = {
            "gloss": key,
            "frequency": value,
        }

    print(f"Created gloss dictionary with {len(sign_dict)} unique glosses")
    return save_dict


def convert_to_json_format(skeleton_data, annotations_data, split_name):
    """Convert data to JSON format expected by the model"""
    json_data = []

    for item in annotations_data:
        video_id = item["video_id"]

        # Get skeleton data
        if video_id in skeleton_data:
            skeleton_seq = skeleton_data[video_id]
            seq_length = len(skeleton_seq)

            json_data.append(
                {
                    "signer": item["signer"],
                    "video_id": video_id,
                    "gloss_sequence": item["gloss_sequence"],
                    "sentence_id": item["sentence_id"],
                    "original_info": item["original_info"],
                }
            )
        else:
            print(f"Warning: No skeleton data found for video {video_id}")

    print(f"Converted {len(json_data)} samples for {split_name}")
    return json_data


def main():
    parser = argparse.ArgumentParser(description="Preprocess MSLR 2026 data")
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="/path/to/pose_data_isharah2000_hands_lips_body_phase2_SI.pkl",
        help="Path to skeleton PKL file",
    )
    parser.add_argument(
        "--pkl_path_us",
        type=str,
        default="/path/to/pose_data_isharah2000_hands_lips_body_phase2_US.pkl",
        help="Path to US skeleton PKL file",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="base/annotations_v2/isharah2000/SI/train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--train_csv_us",
        type=str,
        default="base/annotations_v2/isharah2000/US/train.csv",
        help="Path to US training CSV",
    )
    parser.add_argument(
        "--dev_csv",
        type=str,
        default="base/annotations_v2/isharah2000/SI/dev.csv",
        help="Path to dev CSV",
    )
    parser.add_argument(
        "--dev_csv_us",
        type=str,
        default="base/annotations_v2/isharah2000/US/dev.csv",
        help="Path to US dev CSV",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="Path to test CSV",
    )
    parser.add_argument(
        "--test_csv_us",
        type=str,
        default="test.csv",
        help="Path to US test CSV (if any)",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["si", "us"],
        default=["si"],
        help="Which settings to process: si, us, or both (e.g. --settings si us)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/mslr2026",
        help="Output directory for processed data",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each requested setting (si and/or us)
    for setting in args.settings:
        if setting == "si":
            pkl = args.pkl_path
            train_csv = args.train_csv
            dev_csv = args.dev_csv
            test_csv = args.test_csv
            prefix = "si"
        elif setting == "us":
            pkl = args.pkl_path_us
            train_csv = args.train_csv_us
            dev_csv = args.dev_csv_us
            test_csv = args.test_csv_us
            prefix = "us"

        # Load skeleton and annotation data (skip missing files gracefully)
        try:
            skeleton_data = load_skeleton_data(pkl)
        except Exception as e:
            print(f"Error loading skeleton PKL for {setting}: {e}")
            skeleton_data = {}

        train_data = []
        dev_data = []
        test_data = []

        if os.path.exists(train_csv):
            train_data = load_csv_annotations(train_csv)
        else:
            print(f"Warning: train CSV not found for {setting}: {train_csv}")

        if os.path.exists(dev_csv):
            dev_data = load_csv_annotations(dev_csv)
        else:
            print(f"Warning: dev CSV not found for {setting}: {dev_csv}")

        if os.path.exists(test_csv):
            test_data = load_csv_annotations(test_csv)
        else:
            print(f"Note: test CSV not found for {setting}: {test_csv}")

        # Create gloss dictionary from train+dev (avoid test)
        combined_data = train_data + dev_data
        gloss_dict = create_gloss_dict(combined_data)

        # Convert to JSON format
        train_json = convert_to_json_format(
            skeleton_data, train_data, f"{prefix}_train"
        )
        dev_json = convert_to_json_format(skeleton_data, dev_data, f"{prefix}_dev")
        test_json = convert_to_json_format(skeleton_data, test_data, f"{prefix}_test")

        # Save files
        with open(os.path.join(args.output_dir, f"{prefix}_gloss_dict.json"), "w") as f:
            json.dump(gloss_dict, f, indent=2)

        with open(os.path.join(args.output_dir, f"{prefix}_train_info.json"), "w") as f:
            json.dump(train_json, f, indent=2)

        with open(os.path.join(args.output_dir, f"{prefix}_dev_info.json"), "w") as f:
            json.dump(dev_json, f, indent=2)

        with open(os.path.join(args.output_dir, f"{prefix}_test_info.json"), "w") as f:
            json.dump(test_json, f, indent=2)

        print(f"Saved {prefix} files to {args.output_dir}")


if __name__ == "__main__":
    main()
