import json
import math
import os
import pickle
import random
from pathlib import Path

test_ratio = 0.05
val_ratio = 0.05

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
input_path = data_path / "pre_dataset"

dataset_file_list = list(input_path.glob("dataset_*/dataset.pkl"))

for dataset_file in dataset_file_list:
    dataset_dir = dataset_file.parent

    with open(dataset_dir / "dataset.pkl", "rb") as f:
        load_entries = pickle.load(f)

    entries = load_entries
    print(len(entries))

    mpid_list = set(e[0]["material_id"] for e in entries)
    assert len(mpid_list) == len(entries)

    mpid_list = list(mpid_list)

    random.seed(20230823)
    random.shuffle(mpid_list)

    all_num_elems = len(mpid_list)
    test_num_elems = math.ceil(test_ratio * all_num_elems)
    val_num_elems = math.ceil(val_ratio * (all_num_elems - test_num_elems))
    train_num_elems = all_num_elems - test_num_elems - val_num_elems

    assert all_num_elems == train_num_elems + test_num_elems + val_num_elems
    assert train_num_elems > 0
    assert test_num_elems > 0
    assert val_num_elems > 0

    # print(all_num_elems, train_num_elems, test_num_elems, val_num_elems)
    print(
        f"{dataset_dir.stem} ... train: {train_num_elems}, test: {test_num_elems}, val: {val_num_elems}"
    )

    train_elems = mpid_list[:train_num_elems]
    test_elems = mpid_list[train_num_elems : train_num_elems + test_num_elems]
    val_elems = mpid_list[train_num_elems + test_num_elems :]

    assert len(set(train_elems) & set(test_elems)) == 0
    assert len(set(train_elems) & set(val_elems)) == 0
    assert len(set(test_elems) & set(val_elems)) == 0
    assert len(train_elems) == train_num_elems
    assert len(test_elems) == test_num_elems
    assert len(val_elems) == val_num_elems

    write_data = {"train": train_elems, "test": test_elems, "val": val_elems}

    with open(dataset_dir / "split.json", "w") as f:
        json.dump(write_data, f, indent=4)
