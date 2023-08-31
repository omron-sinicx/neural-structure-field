from logging import getLogger
from pathlib import Path

import hydra
import pytorch_lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from evaluate import evaluate_splits
from model import ModelPTL
from mp_dataset import MPDatasetDataModule
from reconstruction import reconstruction, trained_structure_fields_functor

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config", config_path="./config")
def main(config):
    Path(config["logger"]["save_dir"]).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(**config["logger"])
    wandb_logger.log_hyperparams(config)

    # prepare dataset
    data_module = MPDatasetDataModule(**config["dataset"], **config["data_module"])

    # create model
    modelptl = ModelPTL.create_model(
        config=config,
        bounding_box=data_module.bounding_box,
        num_species=data_module.num_species,
    )
    wandb_logger.watch(modelptl, log="all")

    # create trainer
    checkpoint_params = {**config["checkpoint"]}
    if wandb_logger.version is not None:
        checkpoint_params["dirpath"] = (
            Path(checkpoint_params["dirpath"])
            / wandb_logger.name
            / wandb_logger.version
        )

    print(f"Checkpoint dirpath: {checkpoint_params['dirpath']}")
    checkpoint_callback = ModelCheckpoint(**checkpoint_params)

    trainer = pytorch_lightning.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **config["trainer"],
        # val_percent_check=0,
    )

    # start training
    trainer.fit(modelptl, data_module)
    logger.info("Training finished.")

    # reconstruction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelptl = modelptl.to(device)
    modelptl.eval()

    structure_fields = trained_structure_fields_functor(modelptl)

    reconstructed = reconstruction(
        modelptl.device,
        config,
        data_module,
        structure_fields,
        splits={"train", "validation", "test"},
        progress_bar=True,
    )
    logger.info("Reconstruction finished.")

    results = evaluate_splits(
        data_module,
        reconstructed,
        splits={"train", "validation", "test"},
        log_dir=Path(config["evaluate"]["log_dir"]),
        postfix="trained",
    )
    logger.info("Evaluation finished.")

    metrics_dict = dict()

    for split, result in results.items():
        for k, v in result.items():
            print(f"{split}/{k}: {v}")
            metrics_dict[f"reconstruct/{split}/{k}"] = v

    wandb_logger.log_metrics(metrics_dict)
    logger.info("Evaluation finished.")

    with open(Path(config["logger"]["save_dir"]) / "metrics.csv", mode="a") as f:
        for split, result in results.items():
            for k, v in result.items():
                f.write(f"{data_module.dataset_name},{split},{k},{v}\n")

    # finalize
    wandb_logger.finalize("success")
    logger.info("Done.")


if __name__ == "__main__":
    main()
