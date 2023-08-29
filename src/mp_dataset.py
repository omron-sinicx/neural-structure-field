import json
import pickle
import shutil
from logging import getLogger
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

logger = getLogger(__name__)


class MPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.num_species = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset.pkl"]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def download(self):
        src_file_path = Path(self.root) / self.raw_file_names[0]
        dst_file_path = Path(self.raw_dir) / self.raw_file_names[0]
        if not src_file_path.exists():
            raise RuntimeError("Dataset not found.")
        logger.info(f"copy: {src_file_path} to {dst_file_path}")
        shutil.copy(src_file_path, dst_file_path)

    def process(self):
        with open(str(self.root) + "/dataset_attr.json", "r") as f:
            dataset_attr = json.load(f)
        num_species = dataset_attr["num_species"]

        logger.info(f"Loading data: {self.raw_paths[0]}")
        with open(self.raw_paths[0], "rb") as f:
            load_data = pickle.load(f)

        positions, species, abcs, angles, mpids = zip(*load_data)
        positions = [torch.tensor(p, dtype=torch.float) for p in positions]
        species = [torch.tensor(s, dtype=torch.long).unsqueeze(-1) for s in species]
        lattice_params = [
            torch.tensor((*abc, *angle), dtype=torch.float)
            for abc, angle in zip(abcs, angles)
        ]

        # x: species, pos: positions, y: lattice_params, mpid: mp-id
        data_list = [
            Data(x=s, pos=p, y=l, mpid=mpid)
            for p, s, l, mpid in zip(positions, species, lattice_params, mpids)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, num_species), self.processed_paths[0])


class MPDatasetDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_dir,
        validation_dataset_dir,
        test_dataset_dir,
        num_dataloder_workers,
        batch_size,
        dataset_name="mp_dataset",
    ):
        super().__init__()
        self.train_dataset_dir = train_dataset_dir
        self.validation_dataset_dir = validation_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.num_dataloder_workers = num_dataloder_workers
        self.batch_size = batch_size
        self.num_species = -1
        self.dataset_name = dataset_name
        # self.prepare_data()
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MPDataset(self.train_dataset_dir)
        self.validation_dataset = MPDataset(self.validation_dataset_dir)
        self.test_dataset = MPDataset(self.test_dataset_dir)
        assert self.train_dataset.num_species == self.validation_dataset.num_species
        assert self.train_dataset.num_species == self.test_dataset.num_species
        self.num_species = self.train_dataset.num_species
        self.bounding_box = self.compute_bounding_box_for_dataset()

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloder_workers,
        )
        return train_dataloader

    def val_dataloader(self):
        validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloder_workers,
        )
        return validation_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloder_workers,
        )
        return test_dataloader

    # compute bounding box for the dataset (mainly for global query)
    # this bounding box is given by training set
    def compute_bounding_box_for_dataset(self):
        pos_list = [d.pos for d in self.train_dataset]

        pos_max = torch.cat([p.max(dim=0)[0].view(1, 3) for p in pos_list], dim=0)
        pos_max = pos_max.max(dim=0)[0]
        pos_min = torch.cat([p.min(dim=0)[0].view(1, 3) for p in pos_list], dim=0)
        pos_min = pos_min.min(dim=0)[0]

        return pos_min, pos_max


if __name__ == "__main__":
    pass
