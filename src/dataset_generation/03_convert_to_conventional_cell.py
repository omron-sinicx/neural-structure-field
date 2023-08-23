import os
import pickle
import shutil
from pathlib import Path

from joblib import Parallel, delayed
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

data_path = Path(os.environ.get("APPLICATION_DIRECTORY", "/workspace/")) / "data"
input_path = data_path / "structures"
output_path = data_path / "structures_conventional_cell"

# remove structure_path
if output_path.exists():
    shutil.rmtree(output_path)

output_path.mkdir(parents=True, exist_ok=True)


def process_entry(e):
    # try:
    #    cs = SpacegroupAnalyzer(e["structure"]).get_conventional_standard_structure()
    # except:
    #    return None
    cs = SpacegroupAnalyzer(e["structure"]).get_conventional_standard_structure()
    return (e, cs)


def process_file(path, output_path):
    print(f"processing {path}")
    with open(path, "rb") as f:
        entries = pickle.load(f)
    input_num = len(entries)
    entries = [process_entry(e) for e in entries]
    entries = [e for e in entries if e is not None]
    with open(output_path / path.name, "wb") as f:
        pickle.dump(entries, f)
    output_num = len(entries)
    return input_num, output_num


input_file_path_list = list(input_path.glob("*.pkl"))
# run in parallel
result_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_file)(path, output_path) for path in input_file_path_list
)

input_num_sum = sum([input_num for input_num, output_num in result_list])
output_num_sum = sum([output_num for input_num, output_num in result_list])
print(f"#input: {input_num_sum}, #output: {output_num_sum}")

print("done")
