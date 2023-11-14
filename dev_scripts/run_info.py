import wandb
import json

api = wandb.Api()
#print(api)
runs = api.runs(
    path="nchiba/NeSF_project",
)


topics = [
    "angle_error",
    "length_error",
    "num_atom_correct_ratio",
    "position_error_detected",
    "position_error_exist",
    "species_correct_detected",
    "species_correct_exist",
]

splits = ["train", "validation", "test"]

result = dict()
for split in splits:
    result[split] = dict()
    for topic in topics:
        result[split][topic] = []


for run in runs:
    dataset = run.config["dataset"]
    dataset = dataset.replace("'", '"')
    #print(dataset)
    dataset = json.loads(dataset)
    #print(dataset)
    #print(type(dataset))
    dataset_name = dataset["dataset_name"]
    #print(dataset_name)
    if dataset_name[0:4] != "YBCO":
        continue
    for split in splits:
        for topic in topics:
            #print(run.summary.keys())
            result[split][topic].append(run.summary[f"reconstruct/{split}/{topic}"])

#print(result)

import torch
import pprint

for split in splits:
    for topic in topics:
        x = result[split][topic]
        x = torch.tensor(x)
        x = (x.mean().item(), x.std().item())
        result[split][topic] = x

#print(result)
pprint.pprint(result)
