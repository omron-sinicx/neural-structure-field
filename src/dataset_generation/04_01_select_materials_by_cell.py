import itertools
import json
import os
import pickle
from pathlib import Path

from joblib import Parallel, delayed

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
input_path = data_path / "structures_conventional_cell"


accept_species_ratio = 0.9
max_lattice_length = 6
min_lattice_coord = -0.1

dataset_name = f"lim_l{max_lattice_length}"
print(f"dataset_name: {dataset_name}")

output_file_path = data_path / "pre_dataset" / f"dataset_{dataset_name}/dataset.pkl"
output_file_path.parent.mkdir(exist_ok=True, parents=True)


def process_entry(e):
    entry, conventional_cell_structure = e
    if max_lattice_length < conventional_cell_structure.lattice.a:
        return None
    if max_lattice_length < conventional_cell_structure.lattice.b:
        return None
    if max_lattice_length < conventional_cell_structure.lattice.c:
        return None
    if conventional_cell_structure.lattice.matrix.min() < min_lattice_coord:
        return None
    return e


def process_file(path):
    with open(path, "rb") as f:
        entries = pickle.load(f)
    entries = [process_entry(e) for e in entries]
    entries = [e for e in entries if e is not None]
    return entries


input_file_path_list = list(input_path.glob("*.pkl"))

# run in parallel
entries = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_file)(path) for path in input_file_path_list
)

# serialize into single list
entries = list(itertools.chain.from_iterable(entries))

# make species list
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
