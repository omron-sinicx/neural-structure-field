import os
import pickle
import shutil
from pathlib import Path

from joblib import Parallel, delayed

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
download_path = data_path / "download"
structure_path = data_path / "structures"

# remove structure_path
if structure_path.exists():
    shutil.rmtree(structure_path)

structure_path.mkdir(parents=True, exist_ok=True)

download_list = list(download_path.glob("*.pkl"))
download_date_list = list(set(p.stem.split("_")[0] for p in download_list))
assert len(download_date_list) > 0

# select latest download
download_date = sorted(download_date_list)[-1]

print(f"selected download_date: {download_date}")

download_file_list = list(download_path.glob(f"{download_date}_*.pkl"))


def process_entry(e):
    if "material_id" not in e:
        print("does not have material_id. Skip.")
        return None
    # if ("final_structure" not in e):
    #    print(e["material_id"], "does not have final_structure. Skip.")
    #    return None
    if "structure" not in e:
        print(e["material_id"], "does not have structure. Skip.")
        return None
    # if ("cif" not in e):
    #    print(e["material_id"], "does not have cif. Skip.")
    #    return None
    # if ("anonymous_formula" not in e):
    #    print(e["material_id"], "does not have anonymous_formula. Skip.")
    #    return None
    if "formula_anonymous" not in e:
        print(e["material_id"], "does not have formula_anonymous. Skip.")
        return None
    # if ("crystal_system" not in e):
    #    print(e["material_id"], "does not have crystal_system. Skip.")
    #    return None
    if "symmetry" not in e:
        print(e["material_id"], "does not have symmetry. Skip.")
        return None
    if "formation_energy_per_atom" not in e:
        print(e["material_id"], "does not have formation_energy_per_atom. Skip.")
        return None
    entry = {
        "material_id": e["material_id"],
        "structure": e["structure"],
        # "cif": e["cif"],
        "formula_anonymous": e["formula_anonymous"],
        "symmetry": e["symmetry"],
        "formation_energy_per_atom": e["formation_energy_per_atom"],
    }
    return entry


def process_file(path, structure_path):
    print(f"processing {path}")
    with open(path, "rb") as f:
        entries = pickle.load(f)
    input_num = len(entries)
    entries = [process_entry(e) for e in entries]
    entries = [e for e in entries if e is not None]
    with open(structure_path / path.name, "wb") as f:
        pickle.dump(entries, f)
    output_num = len(entries)
    return input_num, output_num


# run in parallel
result_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_file)(path, structure_path) for path in download_file_list
)

input_num_sum = sum([input_num for input_num, output_num in result_list])
output_num_sum = sum([output_num for input_num, output_num in result_list])
print(f"#input: {input_num_sum}, #output: {output_num_sum}")

print("done")
