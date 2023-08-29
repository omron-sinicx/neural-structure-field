import pickle
from logging import getLogger
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import ModelPTL
from mp_dataset import MPDatasetDataModule
from utils import (
    compute_ground_truth_pos,
    compute_ground_truth_spec,
    extract_lattice,
    generate_inference_sampling_points,
    generate_local_query,
    get_repeated_query_batch,
    nms_batch,
)

logger = getLogger(__name__)


# reconstruct structure for a batch
def reconstruct_batch(
    position_field,
    species_field,
    this_batch_size,
    lattice_lengths,
    lattice_angles,
    num_iteration,
    residual_distance_threshold,
    global_grid_points,
    species_grid_points,
    nms_radius,
    nms_num_points_threshold,
):
    assert lattice_lengths.shape[0] == this_batch_size
    assert lattice_angles.shape[0] == this_batch_size

    # Estimate position
    query_pos = global_grid_points
    query = get_repeated_query_batch(this_batch_size, query_pos)

    # Estimate position
    for _ in range(num_iteration):
        # vals = compute_ground_truth_pos(query, batch_data)
        vals = position_field(query)
        query.pos = query.pos + vals.detach()

    # Score by estimated residual distances
    sqrd_last_distance = vals.pow(2).sum(axis=1)

    # Remove query points which are estimated to be far from a center position
    active_indices = sqrd_last_distance <= residual_distance_threshold**2
    active_query_pos = query.pos[active_indices]
    active_query_batch = query.batch[active_indices]
    active_query_x = sqrd_last_distance[active_indices]
    query = Batch(pos=active_query_pos, x=active_query_x, batch=active_query_batch)

    # Non-maximum suppression
    query = query.to("cpu")
    detected_centers_list, _ = nms_batch(
        batch_data=query,
        radius=nms_radius,
        num_points_threshold=nms_num_points_threshold,
    )

    detected_centers = Batch.from_data_list(detected_centers_list)
    detected_centers = detected_centers.to(global_grid_points.device)

    # Estimate species
    query_pos = species_grid_points
    num_query_pos = query_pos.shape[0]

    query = generate_local_query(
        batch_data=detected_centers,
        query_pos=query_pos,
        grid_distribution=None,
    )

    # pred_spec = compute_ground_truth_spec(query, batch_data)
    pred_spec = species_field(query)
    pred_spec = pred_spec.view(-1, num_query_pos, pred_spec.shape[1])
    pred_spec = pred_spec.argmax(dim=2)
    pred_spec, _ = pred_spec.mode(dim=1)

    # Estimate lattice parameters
    pred_y = torch.cat([lattice_lengths, lattice_angles], dim=1)

    estimated_list = []

    # Combine into estimated data
    for batch_idx in range(this_batch_size):
        batch_indices = detected_centers.batch == batch_idx
        pos = detected_centers.pos[batch_indices]
        specs = pred_spec[batch_indices]
        props = pred_y[batch_idx, :]
        estimated = Data(pos=pos, x=specs, y=props).to("cpu")
        estimated_list.append(estimated)

    return estimated_list


# reconstruct structure for a dataset
def reconstruct_dataset(
    dataset,
    reconstruction_config,
    inference_global_grid_points,
    inference_species_grid_points,
    structure_fields,
    progress_bar=True,
):
    device = inference_global_grid_points.device

    dataloader = DataLoader(
        dataset,
        batch_size=reconstruction_config["batch_size"],
        shuffle=False,
    )

    if progress_bar:
        dataloader = tqdm(dataloader)

    estimated_dict = dict()

    for batch_data in dataloader:
        with torch.no_grad():
            batch_data = batch_data.to(device)

            # Generation functions of position field and species field
            # fields are given as functions objects
            (
                position_field,
                species_field,
                lattice_lengths,
                lattice_angles,
            ) = structure_fields(batch_data)
            this_batch_size = batch_data.num_graphs

            batch_estimated_list = reconstruct_batch(
                position_field=position_field,
                species_field=species_field,
                this_batch_size=this_batch_size,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
                num_iteration=reconstruction_config["num_iteration"],
                residual_distance_threshold=reconstruction_config[
                    "residual_distance_threshold"
                ],
                global_grid_points=inference_global_grid_points,
                species_grid_points=inference_species_grid_points,
                nms_radius=reconstruction_config["nms"]["radius"],
                nms_num_points_threshold=reconstruction_config["nms"][
                    "num_points_threshold"
                ],
            )
            assert len(batch_data.mpid) == len(batch_estimated_list)

            for mpid, estimated in zip(batch_data.mpid, batch_estimated_list):
                assert mpid not in estimated_dict
                estimated_dict[mpid] = estimated

    return estimated_dict


# reconstruction correspoint to a config and structure fields
def reconstruction(
    device,
    config,
    data_module,
    structure_fields,
    splits={"train", "validation", "test"},
    progress_bar=True,
):
    # Generate sampling points
    sampling_points = generate_inference_sampling_points(
        bounding_box=data_module.bounding_box,
        inference_sampling_config=config["sampling"]["inference"],
    )

    inference_global_grid_points = sampling_points["global_grid_points"]
    inference_species_grid_points = sampling_points["species_grid_points"]

    inference_global_grid_points = inference_global_grid_points.to(device)
    inference_species_grid_points = inference_species_grid_points.to(device)

    reconstruction_params = {
        "reconstruction_config": config["reconstruction"],
        "inference_global_grid_points": inference_global_grid_points,
        "inference_species_grid_points": inference_species_grid_points,
        "structure_fields": structure_fields,
        "progress_bar": progress_bar,
    }

    # Reconstruction

    results = dict()

    if "train" in splits:
        logger.info("Train dataset reconstruction start.")
        estimated_dict = reconstruct_dataset(
            dataset=data_module.train_dataset,
            **reconstruction_params,
        )
        results["train"] = estimated_dict
        logger.info(
            f"Train dataset reconstruction done. {len(estimated_dict)} entries."
        )

    if "validation" in splits:
        logger.info("Validation dataset reconstruction start.")
        estimated_dict = reconstruct_dataset(
            dataset=data_module.validation_dataset,
            **reconstruction_params,
        )
        results["validation"] = estimated_dict
        logger.info(
            f"Validation dataset reconstruction done. {len(estimated_dict)} entries."
        )

    if "test" in splits:
        logger.info("Test dataset reconstruction start.")
        estimated_dict = reconstruct_dataset(
            dataset=data_module.test_dataset,
            **reconstruction_params,
        )
        results["test"] = estimated_dict
        logger.info(f"Test dataset reconstruction done. {len(estimated_dict)} entries.")

    return results


# Functor which generates structure fields from a model
def trained_structure_fields_functor(modelptl):
    modelptl.eval()
    model = modelptl.model

    def trained_structure_fields(batch_data):
        if model.vae:
            latent, _ = model.encoder(batch_data)
        else:
            latent = model.encoder(batch_data)

        def position_field(query):
            return model.pos_decoder(latent, query)

        def species_field(query):
            return model.spec_decoder(latent, query)

        lattice_lengths = model.length_decoder(latent)
        lattice_angles = model.angle_decoder(latent)

        return position_field, species_field, lattice_lengths, lattice_angles

    return trained_structure_fields


def ground_truth_structure_fields_functor(dataset_num_species):
    def ground_truth_structure_fields(batch_data):
        # position_field = lambda query: compute_ground_truth_pos(query, batch_data)
        # species_field = lambda query: compute_ground_truth_spec(query, batch_data)
        def position_field(query):
            return compute_ground_truth_pos(query, batch_data)

        def species_field(query):
            pred_spec = compute_ground_truth_spec(query, batch_data)
            pred_spec = pred_spec.type(torch.int64)
            pred_spec = F.one_hot(pred_spec, num_classes=dataset_num_species)

            return pred_spec

        lattice_lengths, lattice_angles = extract_lattice(batch_data)
        return position_field, species_field, lattice_lengths, lattice_angles

    return ground_truth_structure_fields


@hydra.main(version_base=None, config_name="config", config_path="./config")
def main(config):
    output_dir = Path(config["reconstruction"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare dataset
    data_module = MPDatasetDataModule(**config["dataset"], **config["data_module"])
    logger.info(f"Dataset: {data_module.dataset_name}")
    logger.info(f"Train dataset size: {len(data_module.train_dataset)}")
    logger.info(f"Validation dataset size: {len(data_module.validation_dataset)}")
    logger.info(f"Test dataset size: {len(data_module.test_dataset)}")
    logger.info(f"Train Dataset Bounding box: {data_module.bounding_box}")

    # set reconstruction mode: trained or ground_truth
    mode = config["reconstruction"].get("mode", "trained")
    assert mode in ["trained", "ground_truth"]

    logger.info(f"Reconstruction mode: {mode}")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if mode == "trained":
        logger.info("Reconstructing with trained model.")

        output_path = output_dir / f"{data_module.dataset_name}.pkl"

        # load checkpoint
        assert data_module.dataset_name in config["trained_checkpoints"]
        checkpoint_path = Path(config["trained_checkpoints"][data_module.dataset_name])
        logger.info(f"Checkpoint path: {checkpoint_path}")

        # create model
        modelptl_params = ModelPTL.create_model(
            config=config,
            bounding_box=data_module.bounding_box,
            num_species=data_module.num_species,
            instanciate=False,
        )

        # load checkpoint
        modelptl = ModelPTL.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **modelptl_params,
        )
        modelptl = modelptl.to(device)
        modelptl.eval()

        # Generation functions of position field and species field
        # fields are given as functions objects
        structure_fields = trained_structure_fields_functor(modelptl)

    elif mode == "ground_truth":
        logger.info("Reconstructing with ground truth.")

        output_path = output_dir / f"{data_module.dataset_name}_ground_truth.pkl"

        structure_fields = ground_truth_structure_fields_functor(
            dataset_num_species=data_module.num_species
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    logger.info(f"Output path: {output_path}")

    results = reconstruction(
        device,
        config,
        data_module,
        structure_fields,
        splits={"train", "validation", "test"},
        progress_bar=True,
    )

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Done.")


if __name__ == "__main__":
    main()
