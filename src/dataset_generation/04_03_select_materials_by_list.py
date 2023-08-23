import itertools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
input_path = data_path / "structures_conventional_cell"


num_entry_limit = None
accept_species_ratio = 0.9
accept_lattice_ratio = 0.2
num_sites_range = 0

dataset_name = "YBCO13"

print(f"dataset_name: {dataset_name}")

output_file_path = data_path / "pre_dataset" / f"dataset_{dataset_name}/dataset.pkl"
output_file_path.parent.mkdir(exist_ok=True, parents=True)

input_list_path = Path(__file__).parent / "retrieved_neighbours_mp-20674_YBCO.csv"
load_data = pd.read_csv(input_list_path, encoding="ms932", sep=",")
target_mpid_list = load_data["mp_id"]
target_mpid_set = set(target_mpid_list)


def process_file(path):
    with open(path, "rb") as f:
        entries = pickle.load(f)
    return entries


input_file_path_list = list(input_path.glob("*.pkl"))

entries = []
for path in tqdm(input_file_path_list):
    with open(path, "rb") as f:
        entries = entries + pickle.load(f)

print(len(entries))

# remove entries not in dataset_mpid_list and pick up entries in target_mpid_list
dataset_mpid_dict = {e[0]["material_id"]: e for e in entries}
target_list = [
    dataset_mpid_dict[mpid]
    for mpid in tqdm(target_mpid_list)
    if mpid in dataset_mpid_dict
]

# extract template entry
print(f"target: {target_list[0][0]['material_id']}")

target = target_list[0][1]
print(f"target_num_atoms: {target.num_sites}")
print(f"target_lattice: {target.lattice.matrix}")
print(
    f"target_lattice_length: {np.sqrt(np.power(target.lattice.matrix, 2).sum(axis=1))}"
)

target_num_sites = target.num_sites
target_lattice = target.lattice.matrix
target_lattice_length = np.sqrt(np.power(target_lattice, 2).sum(axis=1))

min_num_atoms = target_num_sites - num_sites_range
max_num_atoms = target_num_sites + num_sites_range

# filtering by num_atoms compared with template
entries = [
    e
    for e in tqdm(entries)
    if (min_num_atoms <= e[1].num_sites and e[1].num_sites <= max_num_atoms)
]
print(len(entries))

# lattice distortion
extracted_entries = []

for e in entries:
    m = e[1].lattice.matrix
    lattice_diff = m - target_lattice
    lattice_length_diff = np.sqrt(np.power(lattice_diff, 2).sum(axis=1))
    max_lattice_ratio = (lattice_length_diff / target_lattice_length).max()
    if max_lattice_ratio < accept_lattice_ratio:
        extracted_entries.append(e)

entries = extracted_entries
print(len(entries))

# limit number of entries
if num_entry_limit is not None:
    entries = entries[:num_entry_limit]
print(len(entries))

# limit variations of species
species = [list({s.Z for s in e[1].species}) for e in entries]
species = list(itertools.chain.from_iterable(species))

species_count = {s: 0 for s in set(species)}
for s in species:
    species_count[s] += 1

spec_list = [(v, k) for k, v in species_count.items()]
spec_list = list(reversed(sorted(spec_list)))

num_accept_species = int(len(spec_list) * accept_species_ratio)
spec_list = spec_list[:num_accept_species]
allowed_species = [z for _, z in spec_list]
allowed_species = set(allowed_species)

entries = [
    e for e in entries if len({s.Z for s in e[1].species} - allowed_species) == 0
]
print(len(entries))

print(f"selected entries: {len(entries)}")

# generate attributes

species = [{s.Z for s in e[1].species} for e in entries]
species = set.union(*species)
species = sorted(list(species))

sidx2z_list = species
num_atom_idx = len(sidx2z_list)

print(f"num_sidx: {num_atom_idx}")

dataset_attr = {
    "num_species": len(sidx2z_list),
    "sidx_to_z": sidx2z_list,
}

with open(output_file_path.parent / "dataset_attr.json", "w") as f:
    json.dump(dataset_attr, f, indent=4)

print(f"dataset_attr: {dataset_attr}")

with open(output_file_path, "wb") as f:
    pickle.dump(entries, f)

print("done")
