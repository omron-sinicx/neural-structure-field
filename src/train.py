from logging import getLogger
from pathlib import Path

import hydra
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import ModelPTL
from mp_dataset import MPDatasetDataModule

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

    # finalize
    wandb_logger.finalize("success")
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
