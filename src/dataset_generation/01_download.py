import datetime
import os
import pickle
from pathlib import Path

from mp_api.client import MPRester

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
download_path = data_path / "download"
download_path.mkdir(parents=True, exist_ok=True)

print(f"download_path: {download_path}")

API_KEY_path = Path(__file__).parent / "API_KEY.txt"

print(f"API_KEY_path: {API_KEY_path}")

if not API_KEY_path.exists():
    print("Please put your API_KEY in the file API_KEY.txt")
    print("See: https://next-gen.materialsproject.org/api")
    exit(1)

with open(API_KEY_path, "r") as f:
    API_KEY = f.read().strip()

print(f"API_KEY: {API_KEY}")
mpr = MPRester(API_KEY, use_document_model=False)

entries = mpr.materials.summary._search(fields=["material_id"], nsites_min=-1)
# entries = mpr.materials.summary._search(fields=["material_id"], nsites_min=-1, nsites_max=1) # for debugging

# for debugging
# entries = entries[0:2000]

mpids = [e["material_id"] for e in entries]

all_properties = [
    "builder_meta",
    "nsites",
    "elements",
    "nelements",
    "composition",
    "composition_reduced",
    "formula_pretty",
    "formula_anonymous",
    "chemsys",
    "volume",
    "density",
    "density_atomic",
    "symmetry",
    "property_name",
    "material_id",
    "deprecated",
    "deprecation_reasons",
    "last_updated",
    "origins",
    "warnings",
    "structure",
    "task_ids",
    "uncorrected_energy_per_atom",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "equilibrium_reaction_energy_per_atom",
    "decomposes_to",
    "xas",
    "grain_boundaries",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "is_gap_direct",
    "is_metal",
    "es_source_calc_id",
    "bandstructure",
    "dos",
    "dos_energy_up",
    "dos_energy_down",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "total_magnetization_normalized_formula_units",
    "num_magnetic_sites",
    "num_unique_magnetic_sites",
    "types_of_magnetic_species",
    "k_voigt",
    "k_reuss",
    "k_vrh",
    "g_voigt",
    "g_reuss",
    "g_vrh",
    "universal_anisotropy",
    "homogeneous_poisson",
    "e_total",
    "e_ionic",
    "e_electronic",
    "n",
    "e_ij_max",
    "weighted_surface_energy_EV_PER_ANG2",
    "weighted_surface_energy",
    "weighted_work_function",
    "surface_anisotropy",
    "shape_factor",
    "has_reconstructed",
    "possible_species",
    "has_props",
    "theoretical",
    "database_IDs",
]

today = datetime.date.today()
today = today.strftime("%Y%m%d")
print(f"dataset date: {today}")

num_chunks = len(mpids) // 10000 + 1

for i in range(num_chunks):
    print(f"chunk {i+1}/{num_chunks}")
    this_mpids = mpids[i * 10000 : (i + 1) * 10000]
    # print(len(this_mpids))
    this_mpids = ",".join(this_mpids)
    # print(this_mpids)
    entries = mpr.materials.summary._search(
        material_ids=this_mpids, all_fields=True, chunk_size=100
    )
    # entries = mpr.materials.summary._search(material_ids=this_mpids, fields=all_properties, chunk_size=100)
    # print(len(entries))
    with open(download_path / f"{today}_{i:04}.pkl", "wb") as f:
        pickle.dump(entries, f)

print("done")
