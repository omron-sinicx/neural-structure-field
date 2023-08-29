import itertools
import json
import os
import pickle
from pathlib import Path

from joblib import Parallel, delayed
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
input_path = data_path / "structures_conventional_cell"


candidate_formulas = ["ABC3", "ABC2", "AB"]
max_lattice_length = 6
min_lattice_coord = -0.1
max_num_atoms = 40

dataset_name = "ICSG3D"

print(f"dataset_name: {dataset_name}")

output_file_path = data_path / "pre_dataset" / f"dataset_{dataset_name}/dataset.pkl"
output_file_path.parent.mkdir(exist_ok=True, parents=True)


def process_entry(e):
    entry, conventional_cell_structure = e
    if conventional_cell_structure.num_sites > max_num_atoms:
        return None
    if entry["formula_anonymous"] not in candidate_formulas:
        return None
    if SpacegroupAnalyzer(conventional_cell_structure).get_crystal_system() != "cubic":
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
