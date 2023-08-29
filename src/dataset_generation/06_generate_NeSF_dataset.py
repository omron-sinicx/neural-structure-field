import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
input_path = data_path / "pre_dataset"
output_path = data_path / "dataset"

output_path.mkdir(exist_ok=True, parents=True)

dataset_file_list = list(input_path.glob("dataset_*/dataset.pkl"))

for dataset_file in dataset_file_list:
    input_dir = dataset_file.parent
    assert input_dir.exists()
    dataset_name = input_dir.stem[8:]
    print(dataset_name)

    train_dataset_dir = output_path / f"dataset_{dataset_name}_train"
    test_dataset_dir = output_path / f"dataset_{dataset_name}_test"
    val_dataset_dir = output_path / f"dataset_{dataset_name}_val"

    train_dataset_dir.mkdir(exist_ok=True)
    test_dataset_dir.mkdir(exist_ok=True)
    val_dataset_dir.mkdir(exist_ok=True)

    load_file_path = input_dir / "dataset.pkl"

    print(f"loading: {load_file_path}...")
    load_entries = pd.read_pickle(load_file_path)
    print("load done")

    with open(input_dir / "dataset_attr.json", "r") as f:
        dataset_attr = json.load(f)

    input_data_dict = dict()

    sidx2z = dataset_attr["sidx_to_z"]
    z2sidx = {z: sidx for sidx, z in enumerate(sidx2z)}

    for entry in tqdm(load_entries):
        structure = entry[1]

        pos_list = structure.cart_coords
        z_list = [z2sidx[s.Z] for s in structure.species]
        z_list = np.array(z_list, dtype=np.int64)
        abc = structure.lattice.abc
        angles = structure.lattice.angles
        angles = tuple(a * math.pi / 180 for a in angles)
        name = entry[0]["material_id"]

        input_data_dict[name] = (pos_list, z_list, abc, angles, name)

    assert len(input_data_dict) == len(load_entries)

    with open(input_dir / "split.json", "r") as f:
        split = json.load(f)

    train_data = [input_data_dict[mpid] for mpid in split["train"]]
    test_data = [input_data_dict[mpid] for mpid in split["test"]]
    val_data = [input_data_dict[mpid] for mpid in split["val"]]

    assert len(train_data) + len(test_data) + len(val_data) == len(input_data_dict)

    with open(train_dataset_dir / "dataset_attr.json", "w") as f:
        json.dump(dataset_attr | {"mpid": split["train"]}, f, indent=4)
    with open(test_dataset_dir / "dataset_attr.json", "w") as f:
        json.dump(dataset_attr | {"mpid": split["test"]}, f, indent=4)
    with open(val_dataset_dir / "dataset_attr.json", "w") as f:
        json.dump(dataset_attr | {"mpid": split["val"]}, f, indent=4)

    with open(train_dataset_dir / "dataset.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(test_dataset_dir / "dataset.pkl", "wb") as f:
        pickle.dump(test_data, f)
    with open(val_dataset_dir / "dataset.pkl", "wb") as f:
        pickle.dump(val_data, f)

    print("done")
