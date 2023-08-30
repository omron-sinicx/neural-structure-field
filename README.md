# NeSF: Neural Structure Fields

![stable](https://img.shields.io/badge/stable-v1.0.0-blue)
![python versions](https://img.shields.io/badge/python-3.10-blue)
[![MIT License](https://img.shields.io/github/license/cvpaperchallenge/Ascender?color=green)](LICENSE)

This is a official implementation of [Neural Structure Fields with Application to Crystal Structure Autoencoders](https://arxiv.org/abs/2212.13120) (NeSF).

## Paper

- Title: Neural Structure Fields with Application to Crystal Structure Autoencoders
- Authors: Naoya Chiba, Yuta Suzuki, Tatsunori Taniai, Ryo Igarashi, Yoshitaka Ushiku, Kotaro Saito, Kanta Ono
- URL (Journal version):
- URL (Preprint version): https://arxiv.org/abs/2212.13120
- URL (NeurIPS 2022 Workshop version): https://openreview.net/forum?id=qLKFSAvMka4

### Abstract in Paper

Representing crystal structures of materials to facilitate determining them via neural networks is crucial for enabling machine-learning applications involving crystal structure estimation. Among these applications, the inverse design of materials can contribute to next-generation methods that explore materials with desired properties without relying on luck or serendipity. We propose neural structure fields (NeSF) as an accurate and practical approach for representing crystal structures using neural networks. Inspired by the concepts of vector fields in physics and implicit neural representations in computer vision, the proposed NeSF considers a crystal structure as a continuous field rather than as a discrete set of atoms. Unlike existing grid-based discretized spatial representations, the NeSF overcomes the tradeoff between spatial resolution and computational complexity and can represent any crystal structure. To evaluate the NeSF, we propose an autoencoder of crystal structures that can recover various crystal structures, such as those of perovskite structure materials and cuprate superconductors. Extensive quantitative results demonstrate the superior performance of the NeSF compared with the existing grid-based approach.

### Citation

- Journal version

- Preprint version

```
@misc{chiba2022nesf_arxiv,
  title={Neural Structure Fields with Application to Crystal Structure Autoencoders},
  author={Naoya Chiba and Yuta Suzuki and Tatsunori Taniai and Ryo Igarashi and Yoshitaka Ushiku and Kotaro Saito and Kanta Ono},
  year={2022},
  eprint={2212.13120},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci}
}
```

- NeurIPS 2022 Workshop version

```
@inproceedings{chiba2022nesf_neuripsws,
  title={Neural Structure Fields with Application to Crystal Structure Autoencoders},
  author={Naoya Chiba and Yuta Suzuki and Tatsunori Taniai and Ryo Igarashi and Yoshitaka Ushiku and Kotaro Saito and Kanta Ono},
  booktitle={AI for Accelerated Materials Design NeurIPS 2022 Workshop},
  year={2022},
  url={https://openreview.net/forum?id=qLKFSAvMka4}
}
```

## Installation

### Docker

We recommend to use Docker to run the code. This repository is based on the template [Ascender](https://github.com/cvpaperchallenge/Ascender) and this template author provide step-by-step instructions for installing Docker and Docker Compose. Please refer to the following link for details.

https://github.com/cvpaperchallenge/Ascender#prerequisites

In short, we summarize the installation commands as follows:

```bash
# Set up the repository
sudo apt update

sudo apt install ca-certificates curl gnupg lsb-release

sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker and Docker Compose

sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update

sudo apt install -y nvidia-docker2

sudo systemctl restart docker
```

For start docker container, please run the following command:

```bash
cd environments/gpu
sudo docker compose up -d
```

For stop docker container, please run the following command:

```bash
cd environments/gpu
sudo docker compose stop
```

If you do not want to use Docker, please refer to the followings: https://github.com/cvpaperchallenge/Ascender#use-ascender-without-docker

### Poetry

We use Poetry to manage Python packages. Please refer to the following link for details.

At /workspace in the container, following commands installs dependencies.

```bash
poetry install
```

## Dataset Preparation

Please see: [src/dataset_generation/README.md](src/dataset_generation/README.md)

## Training

Run the following command to train the model.

```bash
poetry run python src/train.py dataset=ICSG3D
poetry run python src/train.py dataset=lim_l6
poetry run python src/train.py dataset=YBCO13
```

Settings are given by `src/configs/config.yaml`. We use [Hydra](https://hydra.cc/) to manage settings.

## Reconstruction

Run the following command to reconstruct the crystal structure.

Before running this script, please write checkpoint path to `trained_checkpoints` in `src/config/config.yaml`.

This script outputs the reconstructed crystal structure at `workspace/reconstruction`.

This script also can reconstruct the crystal structure from the ground truth strcutre field, which is for test how behave the reconstruction algorithm.

```bash
# Reconstruction from the predicted structure field
poetry run python src/reconstruction.py dataset=ICSG3D
poetry run python src/reconstruction.py dataset=lim_l6
poetry run python src/reconstruction.py dataset=YBCO13
# Reconstruction from the ground truth structure field
poetry run python src/reconstruction.py dataset=ICSG3D reconstruction.mode=ground_truth
poetry run python src/reconstruction.py dataset=lim_l6 reconstruction.mode=ground_truth
poetry run python src/reconstruction.py dataset=YBCO13 reconstruction.mode=ground_truth
```

## Evaluation

Run the following command to evaluate the model.

```bash
poetry run python src/evaluate.py dataset=ICSG3D
poetry run python src/evaluate.py dataset=lim_l6
poetry run python src/evaluate.py dataset=YBCO13
```

This script outputs the evaluation results at `workspace/logs`.

### License

[MIT License](LICENSE.md)

### Acknowledgements

- We thank the following projects.

  - [The Materials Project](https://next-gen.materialsproject.org/)
  - [Pymatgen](https://pymatgen.org/)
  - [PyTorch](https://pytorch.org/)
  - [PyTorch Geometric](https://pyg.org/)
  - [PyTorch Lightning](https://lightning.ai/)

- We also thank the authors of the repositories template [Ascender](https://github.com/cvpaperchallenge/Ascender).
