import pickle
from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import torch
from torch_geometric.nn import nearest

from mp_dataset import MPDatasetDataModule
from utils import extract_lattice

logger = getLogger(__name__)


def evaluate_entry(
    true_data,
    pred_data,
    log_f=None,
):
    true_pos = true_data.pos
    true_spec = true_data.x.flatten()
    true_length, true_angle = extract_lattice(true_data)

    pred_pos = pred_data.pos
    pred_spec = pred_data.x.flatten()
    pred_length, pred_angle = extract_lattice(pred_data)

    # compute distance error
    num_true_points = true_pos.shape[0]
    num_pred_points = pred_pos.shape[0]

    ext_true_pos = true_pos.view(1, num_true_points, 3).repeat(num_pred_points, 1, 1)
    ext_pred_pos = pred_pos.view(num_pred_points, 1, 3).repeat(1, num_true_points, 1)
    diff = ext_true_pos - ext_pred_pos
    dists = diff.pow(2).sum(dim=2)
    dists = dists.sqrt()
    # dists: (num_pred_points(detected), num_true_points(exist))

    if num_pred_points > 0:
        dists_exist, _ = dists.min(dim=0)
    else:
        dists_exist = torch.ones((num_true_points,), dtype=torch.float32) * np.inf

    if num_true_points > 0:
        dists_detected, _ = dists.min(dim=1)
    else:
        logger.warning("something wrong in dataset")
        dists_detected = torch.ones((num_pred_points,), dtype=torch.float32) * np.inf

    position_error_exist = dists_exist
    position_error_detected = dists_detected

    num_atom_correct = true_pos.shape[0] == pred_pos.shape[0]

    # compute spec error

    if num_pred_points > 0:
        nearest_idx = nearest(pred_pos, true_pos)
        compare_true_spec = true_spec[nearest_idx]

        nearest_idx = nearest(true_pos, pred_pos)
        compare_pred_spec = pred_spec[nearest_idx]
    else:
        compare_true_spec = torch.ones(pred_spec.shape, dtype=torch.int64) * -1
        compare_pred_spec = torch.ones(true_spec.shape, dtype=torch.int64) * -1

    species_correct_detected = compare_true_spec == pred_spec
    species_correct_exist = true_spec == compare_pred_spec

    length_error = torch.abs(true_length - pred_length).flatten()
    angle_error = torch.abs(true_angle - pred_angle).flatten() * 180 / np.pi

    retval = {
        "num_atom_correct": num_atom_correct,
        "position_error_exist": position_error_exist,
        "position_error_detected": position_error_detected,
        "species_correct_exist": species_correct_exist,
        "species_correct_detected": species_correct_detected,
        "length_error": length_error,
        "angle_error": angle_error,
    }

    if log_f is not None:
        # write log
        log_f.write(f"num_atom_exist: {num_true_points}\n")
        log_f.write(f"num_atom_detected: {num_pred_points}\n")
        log_f.write(f"num_atom_correct: {num_atom_correct}\n")

        position_error_exist = float(position_error_exist.mean())
        position_error_detected = float(position_error_detected.mean())
        species_correct_exist = float(species_correct_exist.float().mean())
        species_correct_detected = float(species_correct_detected.float().mean())

        length_error = ", ".join(f"{x.item():.4f}" for x in length_error)
        angle_error = ", ".join(f"{x.item():.4f}" for x in angle_error)

        log_f.write(f"position_error_exist: {position_error_exist:.4f}\n")
        log_f.write(f"position_error_detected: {position_error_detected:.4f}\n")
        log_f.write(f"species_correct_exist: {species_correct_exist*100:.4f}\n")
        log_f.write(f"species_correct_detected: {species_correct_detected*100:.4f}\n")
        log_f.write(f"length_error: {length_error}\n")
        log_f.write(f"angle_error: {angle_error}\n")
        log_f.write("\n")

    return retval


def evaluate_dataset(dataset, reconstructed, split, log_f=None):
    num_atom_correct_list = []
    position_error_exist_list = []
    position_error_detected_list = []
    species_correct_exist_list = []
    species_correct_detected_list = []
    lattice_length_error_list = []
    lattice_angle_error_list = []

    for true_data in dataset:
        mpid = true_data["mpid"]
        pred_data = reconstructed[mpid]

        if log_f is not None:
            log_f.write(f"{mpid} (in {split})\n")

        retval = evaluate_entry(true_data, pred_data, log_f=log_f)

        num_atom_correct_list.append(retval["num_atom_correct"])
        position_error_exist_list.append(retval["position_error_exist"])
        position_error_detected_list.append(retval["position_error_detected"])
        species_correct_exist_list.append(retval["species_correct_exist"])
        species_correct_detected_list.append(retval["species_correct_detected"])
        lattice_length_error_list.append(retval["length_error"])
        lattice_angle_error_list.append(retval["angle_error"])

    # write summary
    num_atom_correct = sum(num_atom_correct_list)
    num_atom_correct_ratio = num_atom_correct / len(dataset)

    position_error_exist = torch.cat(position_error_exist_list)
    position_error_exist = float(position_error_exist.mean())

    position_error_detected = torch.cat(position_error_detected_list)
    position_error_detected = float(position_error_detected.mean())

    species_correct_exist = torch.cat(species_correct_exist_list)
    species_correct_exist = float(species_correct_exist.float().mean())

    species_correct_detected = torch.cat(species_correct_detected_list)
    species_correct_detected = float(species_correct_detected.float().mean())

    length_error = torch.cat(lattice_length_error_list)
    length_error = float(length_error.mean())

    angle_error = torch.cat(lattice_angle_error_list)
    angle_error = float(angle_error.mean())

    summary_text = f"""Summary (in {split})
num_atom_correct_ratio: {num_atom_correct_ratio:.4f}
position_error_exist: {position_error_exist:.4f}
position_error_detected: {position_error_detected:.4f}
species_correct_exist: {species_correct_exist*100:.4f}
species_correct_detected: {species_correct_detected*100:.4f}
length_error: {length_error:.4f}
angle_error: {angle_error:.4f}
"""
    logger.info(summary_text)
    log_f.write(summary_text)

    return {
        "num_atom_correct_ratio": num_atom_correct_ratio,
        "position_error_exist": position_error_exist,
        "position_error_detected": position_error_detected,
        "species_correct_exist": species_correct_exist,
        "species_correct_detected": species_correct_detected,
        "length_error": length_error,
        "angle_error": angle_error,
    }


def evaluate_splits(
    data_module, reconstructed_data, splits, log_dir=None, postfix="trained"
):
    logger.info(f"Dataset: {data_module.dataset_name}")

    datasets = {
        "train": data_module.train_dataset,
        "validation": data_module.validation_dataset,
        "test": data_module.test_dataset,
    }

    logger.info(f"Train dataset size: {len(datasets['train'])}")
    logger.info(f"Validation dataset size: {len(datasets['validation'])}")
    logger.info(f"Test dataset size: {len(datasets['test'])}")

    result = dict()

    # evaluate
    for split in splits:
        assert split in datasets.keys()
        logger.info(f"Split: {split}")

        # get dataset
        dataset = datasets[split]
        logger.info(f"Dataset size: {len(dataset)}")

        # get reconstructed data
        reconstructed = reconstructed_data[split]
        logger.info(f"Reconstructed data size: {len(reconstructed)}")

        # check all data is reconstructed
        reconstructed_mpids = set(reconstructed.keys())
        assert len(dataset) == len(reconstructed)
        assert all(data["mpid"] in reconstructed_mpids for data in dataset)

        # evaluate
        if log_dir is None:
            log_f = None
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_f = open(
                log_dir / f"{data_module.dataset_name}_{split}_{postfix}.log", mode="w"
            )

        retval = evaluate_dataset(dataset, reconstructed, split, log_f=log_f)

        if log_f is not None:
            log_f.close()

        result[split] = retval

    # write summary
    return result


@hydra.main(version_base=None, config_name="config", config_path="./config")
def main(config):
    log_dir = Path(config["evaluate"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    reconstructed_dir = Path(config["reconstruction"]["output_dir"])

    # prepare dataset
    data_module = MPDatasetDataModule(**config["dataset"], **config["data_module"])

    # set reconstruction mode: trained or ground_truth
    mode = config["reconstruction"].get("mode", "trained")
    assert mode in ["trained", "ground_truth"]

    if mode == "trained":
        reconstructed_path = reconstructed_dir / f"{data_module.dataset_name}.pkl"
    elif mode == "ground_truth":
        reconstructed_path = (
            reconstructed_dir / f"{data_module.dataset_name}_ground_truth.pkl"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # load reconstructed data
    with open(reconstructed_path, mode="rb") as f:
        reconstructed_data = pickle.load(f)
        splits = reconstructed_data.keys()

    # evaluate
    results = evaluate_splits(
        data_module,
        reconstructed_data,
        splits,
        log_dir=log_dir,
        postfix=mode,
    )

    with open(log_dir / "metrics.csv", mode="a") as f:
        for split, result in results.items():
            for k, v in result.items():
                f.write(f"{data_module.dataset_name},{split},{k},{v}\n")
    logger.info("Done")


if __name__ == "__main__":
    main()
