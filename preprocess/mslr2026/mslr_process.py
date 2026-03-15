import json
import pickle
import numpy as np
from tqdm import tqdm


def sign_dict_update(total_dict, info):
    for item in info:
        split_label = item["gloss_sequence"].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def info2dict(anno_path, dataset_type):
    inputs_list = open(anno_path, "r").readlines()[1:]
    info_list = list()

    for line_idx, line in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        if len(line.split("|")) > 2:
            video_id, gloss_seq, _ = line.split("|")
        else:
            video_id, gloss_seq = line.split("|")
        signer, sentence_id = video_id.split("_")

        info_list.append(
            {
                "signer": signer,
                "video_id": video_id,
                "gloss_sequence": gloss_seq.strip(),
                "sentence_id": sentence_id,
                "original_info": line,
            }
        )
    return info_list


if __name__ == "__main__":

    dataset_save_root = "./datasets/mslr2026"
    for setting in ["si", "us"]:
        sign_dict = dict()
        save_dict = dict()
        for md in ["train", "dev"]:
            uniq_sent = set()
            signer_set = set()
            sentence_set = set()
            split_info = info2dict(f"{setting}_{md}_list.txt", md)

            for item in split_info:
                signer_set.add(item["signer"])
                sentence_set.add(item["sentence_id"])

            with open(f"{dataset_save_root}/{setting}_{md}_info.json", "w") as f:
                json.dump(split_info, f, indent=4)

            sign_dict_update(sign_dict, split_info)
            print(len(sign_dict))

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
        with open(f"{dataset_save_root}/{setting}_gloss_dict.json", "w") as f:
            json.dump(save_dict, f, indent=4)
