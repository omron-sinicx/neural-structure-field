from logging import getLogger

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.pool import nearest
from tqdm import tqdm

logger = getLogger(__name__)


# generate global query points corresponding to the bounding box of the dataset
def generate_global_grid_points(bounding_box, interval):
    pos_min, pos_max = bounding_box

    pos_min = pos_min - interval
    pos_max = pos_max + interval

    grid_points_x = torch.arange(pos_min[0], pos_max[0], step=interval)
    grid_points_y = torch.arange(pos_min[1], pos_max[1], step=interval)
    grid_points_z = torch.arange(pos_min[2], pos_max[2], step=interval)

    grid_points_x, grid_points_y, grid_points_z = torch.meshgrid(
        grid_points_x, grid_points_y, grid_points_z, indexing="ij"
    )
    grid_points = torch.stack(
        [grid_points_x.flatten(), grid_points_y.flatten(), grid_points_z.flatten()],
        dim=1,
    )

    return grid_points


# generate local query points positions corresponding to the detected centers
def generate_local_grid_points(interval, range):
    pos_max = range / 2
    pos_min = -range / 2

    grid_points = torch.arange(pos_min, pos_max, step=interval)

    grid_points_x, grid_points_y, grid_points_z = torch.meshgrid(
        grid_points, grid_points, grid_points, indexing="ij"
    )
    grid_points = torch.stack(
        [grid_points_x.flatten(), grid_points_y.flatten(), grid_points_z.flatten()],
        dim=1,
    )

    return grid_points


# generate repeated query for batch
def get_repeated_query_batch(batch_size, query):
    # query: [points, dim]
    assert query.shape[1] == 3
    batch_query_list = [Data(pos=query) for _ in range(batch_size)]
    batch = Batch.from_data_list(batch_query_list)
    batch = batch.to(query.device)

    # <batch>[pos=(query, dim)]
    return batch


# compute ground truth position field from batch_data for query points
def compute_ground_truth_pos(query, batch_data):
    # query: <batch>[query, dim]
    # batch_data: <batch>[points, dim]
    assert query.num_graphs == batch_data.num_graphs
    assert query.pos.shape[1] == 3
    assert batch_data.pos.shape[1] == 3

    nearest_idx = nearest(
        query.pos, batch_data.pos, query.batch, batch_data.batch
    )  # [query]
    nearest_pos = batch_data.pos[nearest_idx, :]  # [query, dim]
    diff_pos = nearest_pos - query.pos  # [batch*query, dim]

    return diff_pos


# compute ground truth species field from batch_data for query points
def compute_ground_truth_spec(query, batch_data):
    assert query.pos.shape[1] == batch_data.pos.shape[1]
    if query.pos.shape[0] == 0:
        return torch.zeros([0, query.pos.shape[1]]).to(query.pos.device)
    nearest_idx = nearest(
        query.pos, batch_data.pos, query.batch, batch_data.batch
    )  # [query]
    true_y = batch_data.x[nearest_idx, :].view(nearest_idx.shape[0])
    true_y = true_y.detach()

    return true_y


# readout lattice constants from batch_data
def extract_lattice(batch_data):
    y = batch_data.y.reshape(-1, 6)
    y_length = y[:, 0:3].reshape(-1, 3)
    y_angle = y[:, 3:6].reshape(-1, 3)
    y_length = y_length.detach()
    y_angle = y_angle.detach()

    return y_length, y_angle


# generate global query for batch
# add perturbation for training
def generate_global_query(batch_data, query_pos, grid_distribution=None):
    # batch_data: <batch>[pos=(points, dim)]
    # detect_grid_points: query, dim
    # detect_grid_distribution: dim
    assert batch_data.pos.shape[1] == 3
    assert query_pos.shape[1] == 3

    query_pos = query_pos.to(batch_data.pos.device)
    this_batch_size = batch_data.num_graphs

    query = get_repeated_query_batch(this_batch_size, query_pos)

    # add perturbation for training
    if grid_distribution is not None:
        assert grid_distribution.loc.shape[0] == 3
        random_delta_pos = grid_distribution.sample(
            [query.pos.shape[0]]
        )  # [query, dim]
        random_delta_pos = random_delta_pos.to(batch_data.pos.device)
        query.pos = query.pos + random_delta_pos  # [query, dim]

    # <batch>[pos=(query, dim)]
    return query


# generate local query for batch
# add perturbation for training
def generate_local_query(batch_data, query_pos, grid_distribution=None):
    # batch_data: <batch>[pos=(points, dim)]
    # detect_grid_points: query, dim
    # detect_grid_distribution: dim
    assert batch_data.pos.shape[1] == 3
    assert query_pos.shape[1] == 3

    query_pos = query_pos.to(batch_data.pos.device)
    num_query_pos = query_pos.shape[0]
    num_data_pos = batch_data.pos.shape[0]

    data_pos = batch_data.pos.view(num_data_pos, 1, 3).repeat(1, num_query_pos, 1)
    query_pos = query_pos.view(1, num_query_pos, 3).repeat(num_data_pos, 1, 1)
    query_pos = data_pos + query_pos

    # add perturbation for training
    if grid_distribution is not None:
        assert grid_distribution.loc.shape[0] == 3
        random_delta_pos = grid_distribution.sample([num_data_pos, num_query_pos])
        random_delta_pos = random_delta_pos.to(batch_data.pos.device)
        query_pos = query_pos + random_delta_pos

    query_pos = query_pos.view(num_data_pos * num_query_pos, 3)
    query_batch = batch_data.batch.view(num_data_pos, 1).repeat(1, num_query_pos)
    query_batch = query_batch.view(-1)

    query = Batch(pos=query_pos, batch=query_batch)

    # <batch>[pos=(query, dim)]
    return query


# integrate two query batch
def integrate_query_batch(a_query, b_query):
    this_batch_size = max(a_query.num_graphs, b_query.num_graphs)

    query_list = []
    for batch_idx in range(this_batch_size):
        a_batch_indices = a_query.batch == batch_idx
        b_batch_indices = b_query.batch == batch_idx
        a_batch_pos = a_query.pos[a_batch_indices, :]
        b_batch_pos = b_query.pos[b_batch_indices, :]
        batch_pos = torch.cat([a_batch_pos, b_batch_pos], dim=0)
        query_list.append(Data(pos=batch_pos))

    batch = Batch.from_data_list(query_list)
    batch = batch.to(a_query.pos.device)

    return batch


# non-maximum suppression for a single data
def nms_single_data(centers, scores, radius, num_points_threshold):
    radius_sqr = radius**2
    num_centers = centers.shape[0]
    assert centers.shape[0] == num_centers
    assert centers.shape[1] == 3
    assert scores.shape[0] == num_centers

    sort_indices = scores.argsort()
    centers = centers[sort_indices, :]  # [points, dim]
    scores = scores[sort_indices]  # [points]

    detected_centers = []

    while centers.shape[0] > 0:
        target_center = centers[0, :]
        distance_sqr = (centers - target_center).pow(2).sum(dim=1)
        rnn_inner_indices = distance_sqr < radius_sqr

        if rnn_inner_indices.sum() >= num_points_threshold:
            rnn_inner_pos = centers[rnn_inner_indices, :]  # [points, dim]
            rnn_inner_scores = scores[rnn_inner_indices]  # [points]

            weights = F.softmax(-rnn_inner_scores, dim=0)
            mean_center = torch.matmul(weights, rnn_inner_pos)
            detected_centers.append((mean_center, rnn_inner_indices.sum()))

        rnn_outer_indices = torch.logical_not(rnn_inner_indices)
        centers = centers[rnn_outer_indices, :]
        scores = scores[rnn_outer_indices]

    if len(detected_centers) > 0:
        detected_centers, num_points_in_cluster = zip(*detected_centers)
        detected_centers = torch.stack(detected_centers)
        num_points_in_cluster = torch.stack(num_points_in_cluster)
    else:
        detected_centers = torch.empty([0, 3])
        num_points_in_cluster = torch.empty([0])

    # [points, dim]
    return detected_centers, num_points_in_cluster


# apply non-maximum suppression for batch_data
def nms_batch(
    batch_data,
    radius,
    num_points_threshold,
    progress_bar=False,
):
    # batch_data: <batch>(pos=[query, dim], x=[query])

    if len(batch_data.batch) == 0:
        return ([], [])
    this_batch_size = batch_data.num_graphs
    assert batch_data.pos.shape[0] == batch_data.x.shape[0]
    assert batch_data.pos.shape[1] == 3

    detected_centers_list = []
    num_points_in_cluster_list = []

    if progress_bar:
        r = tqdm(range(this_batch_size))
    else:
        r = range(this_batch_size)

    for batch_idx in r:
        batch_indices = batch_data.batch == batch_idx
        centers = batch_data.pos[batch_indices]
        scores = batch_data.x[batch_indices]

        detected_centers, num_points_in_cluster = nms_single_data(
            centers, scores, radius, num_points_threshold
        )
        if detected_centers is not None:
            detected_centers = Data(pos=detected_centers)
            detected_centers_list.append(detected_centers)
            num_points_in_cluster_list.append(num_points_in_cluster)

    # <batch>[points, dim]
    return detected_centers_list, num_points_in_cluster_list


# generate grid points
def generate_training_sampling_points(bounding_box, training_sampling_config):
    training_global_grid_points = generate_global_grid_points(
        bounding_box=bounding_box,
        interval=training_sampling_config["global"]["interval"],
    )
    training_local_grid_points = generate_local_grid_points(
        interval=training_sampling_config["local"]["interval"],
        range=training_sampling_config["local"]["range"],
    )
    training_species_grid_points = generate_local_grid_points(
        interval=training_sampling_config["species"]["interval"],
        range=training_sampling_config["species"]["range"],
    )

    logger.info(f"training_global_grid_points: {training_global_grid_points.shape}")
    logger.info(f"training_local_grid_points: {training_local_grid_points.shape}")
    logger.info(f"training_species_grid_points: {training_species_grid_points.shape}")

    # prepare grid distributions
    global_grid_distribution = torch.distributions.normal.Normal(
        torch.zeros((3,)),
        torch.ones((3,)) * training_sampling_config["global"]["stddev"],
    )
    local_grid_distribution = torch.distributions.normal.Normal(
        torch.zeros((3,)),
        torch.ones((3,)) * training_sampling_config["local"]["stddev"],
    )
    species_grid_distribution = torch.distributions.normal.Normal(
        torch.zeros((3,)),
        torch.ones((3,)) * training_sampling_config["species"]["stddev"],
    )

    return {
        "global_grid_points": training_global_grid_points,
        "global_grid_distribution": global_grid_distribution,
        "local_grid_points": training_local_grid_points,
        "local_grid_distribution": local_grid_distribution,
        "species_grid_points": training_species_grid_points,
        "species_grid_distribution": species_grid_distribution,
    }


def generate_inference_sampling_points(bounding_box, inference_sampling_config):
    inference_global_grid_points = generate_global_grid_points(
        bounding_box=bounding_box,
        interval=inference_sampling_config["global"]["interval"],
    )
    inference_species_grid_points = generate_local_grid_points(
        interval=inference_sampling_config["species"]["interval"],
        range=inference_sampling_config["species"]["range"],
    )

    logger.info(f"inference_global_grid_points: {inference_global_grid_points.shape}")
    logger.info(f"inference_species_grid_points: {inference_species_grid_points.shape}")

    return {
        "global_grid_points": inference_global_grid_points,
        "species_grid_points": inference_species_grid_points,
    }


def compute_nearest_atom_distance_in_dataset(dataset):
    nearest_dist_list = []

    for d in dataset:
        p = d.pos
        num_atom = p.shape[0]
        if num_atom == 1:
            continue
        p_1 = p.view(num_atom, 1, 3).repeat(1, num_atom, 1)
        p_2 = p.view(1, num_atom, 3).repeat(num_atom, 1, 1)
        p = p_1 - p_2
        p = p.pow(2).sum(dim=2).sqrt()
        p = p + (torch.eye(num_atom) * 10000)
        p, _ = p.min(dim=1)
        nearest_dist_list.append(p)

    nearest_dist_list = torch.cat(nearest_dist_list)
    nearest_atom_dist = nearest_dist_list.min().item()
    return nearest_atom_dist


def sampling_interval_check(dataset, interval):
    # check detection sampling interval by using train dataset
    nearest_atom_dist = compute_nearest_atom_distance_in_dataset(dataset)
    logger.info(f"Nearest atom dist: {nearest_atom_dist}")

    ok_flag = interval < nearest_atom_dist / 2
    if not ok_flag:
        logger.warning(
            "Detection grid interval whould be too small for selected dataset."
        )

    return ok_flag
