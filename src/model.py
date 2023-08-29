from logging import getLogger

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import (
    BatchNorm1d,
    CrossEntropyLoss,
    Dropout,
    Identity,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import global_max_pool

from utils import (
    compute_ground_truth_pos,
    compute_ground_truth_spec,
    extract_lattice,
    generate_global_query,
    generate_local_query,
    generate_training_sampling_points,
    integrate_query_batch,
)

logger = getLogger(__name__)


# General MLP
class MLP(Module):
    def __init__(self, channels, use_bn=True, use_dropout=True, drop_prob=0.3):
        super().__init__()
        net = []
        for prev_ch, next_ch in zip(channels[:-2], channels[1:-1]):
            net.append(Linear(prev_ch, next_ch))
            net.append(ReLU())
            if use_bn:
                net.append(BatchNorm1d(next_ch))
            if use_dropout:
                net.append(Dropout(drop_prob))
        net.append(Linear(channels[-2], channels[-1]))
        self.net = Sequential(*net)

    def forward(self, x):
        return self.net(x)


# Encoder of NeSF
class Encoder(Module):
    def __init__(
        self, num_species, vae, latent_size, smlp_pos, smlp_spec, smlp_mix, mlp
    ):
        super().__init__()
        smlp_pos = dict(smlp_pos)
        smlp_spec = dict(smlp_spec)
        smlp_mix = dict(smlp_mix)
        mlp = dict(mlp)

        # tune number of channels to fit
        smlp_pos["channels"] = [
            3,  # input (pos)
            *smlp_pos["channels"],  # layers
        ]
        smlp_spec["channels"] = [
            num_species,  # input (spec)
            *smlp_spec["channels"],  # layers
        ]
        smlp_mix["channels"] = [
            smlp_pos["channels"][-1]
            + smlp_spec["channels"][-1],  # input (smlp_pos + smlp_spec)
            *smlp_mix["channels"],  # layers
        ]
        mlp["channels"] = [
            smlp_mix["channels"][-1] + 6,  # input (smlp_mix + length + angle)
            *mlp["channels"],  # layers
            latent_size,  # output (latent)
        ]
        if vae:
            mlp["channels"][-1] = mlp["channels"][-1] * 2

        # construct network modules
        self.smlp_pos = MLP(**smlp_pos)
        self.smlp_spec = MLP(**smlp_spec)
        self.smlp_mix = MLP(**smlp_mix)
        self.mlp = MLP(**mlp)
        self.latent_size = latent_size
        self.vae = vae

        self.num_species = num_species

    def forward(self, batch_data):
        # data: <batch>[points, dim]
        assert batch_data.pos.shape[1] == 3

        length_x, angle_x = extract_lattice(batch_data)

        pos_x = batch_data.pos
        pos_x = self.smlp_pos(pos_x)

        spec_x = batch_data.x.view(batch_data.x.shape[0])
        spec_x = F.one_hot(spec_x, num_classes=self.num_species)
        spec_x = spec_x.to(torch.float32)
        spec_x = self.smlp_spec(spec_x)

        x = torch.cat([pos_x, spec_x], dim=1)
        x = self.smlp_mix(x)
        x = global_max_pool(x, batch_data.batch)
        x = torch.cat([x, length_x, angle_x], dim=1)
        x = self.mlp(x)

        if self.vae:
            mean = x[:, : self.latent_size]
            var = x[:, self.latent_size :]
            var = F.softplus(var)
            return mean, var
        else:
            return x


# Neural Field-based Decoder
class ImplicitDecoder(Module):
    def __init__(
        self, channels, input_pos, use_bn=True, use_dropout=False, drop_prob=0.3
    ):
        super().__init__()
        assert len(channels) == len(input_pos) + 1

        # construct network modules
        self.layers = []
        for p, n, ipp in zip(channels[:-1], channels[1:], input_pos):
            if ipp:
                self.layers.append(Linear(p + 3, n))
            else:
                self.layers.append(Linear(p, n))
        self.layers = ModuleList(self.layers)
        if use_bn:
            self.bn = [BatchNorm1d(n) for n in channels[1:-1]]
        else:
            self.bn = [Identity() for n in channels[1:-1]]
        self.bn = ModuleList(self.bn)
        self.relu = ReLU()
        if use_dropout:
            self.dropout = Dropout(drop_prob)
        else:
            self.dropout = Identity()

        self.input_pos = input_pos

    def forward(self, latent, query):
        # latent: [batch, channel], query: <batch>[query, dim]
        assert query.pos.shape[1] == 3
        x = latent[query.batch, :]

        for layer, bn, ipp in zip(self.layers[:-1], self.bn, self.input_pos[:-1]):
            if ipp:
                x = torch.cat([x, query.pos], dim=1)
            x = layer(x)
            x = self.relu(x)
            x = bn(x)
            x = self.dropout(x)
        if self.input_pos[-1]:
            x = torch.cat([x, query.pos], dim=1)
        x = self.layers[-1](x)

        # [query, channel]
        return x


# Integrated Model of NeSF
class Model(Module):
    def __init__(
        self,
        num_species,
        latent_size,
        vae,
        encoder,
        decoder,
        sampling,
        loss_weights,
    ):
        super().__init__()

        # tune number of channels to fit
        length_decoder = dict(decoder["length"])
        length_decoder["channels"] = [latent_size, *length_decoder["channels"], 3]
        angle_decoder = dict(decoder["angle"])
        angle_decoder["channels"] = [latent_size, *angle_decoder["channels"], 3]
        pos_decoder = dict(decoder["pos"])
        pos_decoder["channels"] = [latent_size, *pos_decoder["channels"], 3]
        pos_decoder["input_pos"] = [True, *pos_decoder["input_pos"], False]
        spec_decoder = dict(decoder["spec"])
        spec_decoder["channels"] = [latent_size, *spec_decoder["channels"], num_species]
        spec_decoder["input_pos"] = [True, *spec_decoder["input_pos"], False]

        # construct network modules
        self.encoder = Encoder(
            num_species=num_species,
            vae=vae,
            latent_size=latent_size,
            **encoder,
        )
        self.length_decoder = MLP(**length_decoder)
        self.angle_decoder = MLP(**angle_decoder)
        self.pos_decoder = ImplicitDecoder(**pos_decoder)
        self.spec_decoder = ImplicitDecoder(**spec_decoder)

        # loss related parameters
        self.loss_weight = loss_weights
        self.ce_loss = CrossEntropyLoss()

        self.vae = vae
        self.global_grid_points = sampling["global_grid_points"]
        self.global_grid_distribution = sampling["global_grid_distribution"]
        self.local_grid_points = sampling["local_grid_points"]
        self.local_grid_distribution = sampling["local_grid_distribution"]
        self.species_grid_point = sampling["species_grid_points"]
        self.species_grid_distribution = sampling["species_grid_distribution"]

    def forward(self, batch_data):
        # batch_data: <batch>[points, dim],
        # assert(data.pos.shape[1] == 3)
        this_batch_size = batch_data.num_graphs

        # <batch>[points, dim] -> [batch, channels]
        if self.vae:
            mean, var = self.encoder(batch_data)
            latent = self.sampling_latent(mean, var)
        else:
            latent = self.encoder(batch_data)

        # query generation
        # automatically not adding perturbation for inference
        global_query = generate_global_query(
            batch_data, self.global_grid_points, self.global_grid_distribution
        )
        if self.training:
            local_query = generate_local_query(
                batch_data, self.local_grid_points, self.local_grid_distribution
            )
            spec_query = generate_local_query(
                batch_data, self.species_grid_point, self.species_grid_distribution
            )
        else:
            local_query = generate_local_query(batch_data, self.local_grid_points, None)
            spec_query = generate_local_query(batch_data, self.species_grid_point, None)
        integrated_query = integrate_query_batch(global_query, local_query)

        # prediction
        pred_pos = self.pos_decoder(latent, integrated_query)
        pred_spec = self.spec_decoder(latent, spec_query)
        pred_length = self.length_decoder(latent)
        pred_angle = self.angle_decoder(latent)

        # ground truth
        true_pos = compute_ground_truth_pos(integrated_query, batch_data)
        true_pos = true_pos.detach()
        true_spec = compute_ground_truth_spec(spec_query, batch_data)
        true_spec = true_spec.detach()
        true_length, true_angle = extract_lattice(batch_data)
        true_length = true_length.detach()
        true_angle = true_angle.detach()

        # compute loss
        loss_pos = (pred_pos - true_pos).pow(2).mean() * self.loss_weight["pos"]
        loss_spec = self.ce_loss(pred_spec, true_spec) * self.loss_weight["spec"]
        loss_length = (pred_length - true_length).pow(2).mean() * self.loss_weight[
            "length"
        ]
        loss_angle = (pred_angle - true_angle).pow(2).mean() * self.loss_weight["angle"]

        loss = loss_pos + loss_spec + loss_length + loss_angle
        if self.vae:
            loss_KLd = (
                -0.5
                * torch.mean(torch.sum(1 + torch.log(var) - mean.pow(2) - var))
                * self.loss_weight["kld"]
            )
            loss += loss_KLd

        # return losses
        retval = {
            "loss": loss,
            "loss_pos": loss_pos,
            "loss_spec": loss_spec,
            "loss_length": loss_length,
            "loss_angle": loss_angle,
            "seen": this_batch_size,
        }
        if self.vae:
            retval["loss_KLd"] = loss_KLd

        return retval

    # sampling latent variable for VAE
    @staticmethod
    def sampling_latent(mean, var):
        latent = mean + torch.sqrt(var) * torch.randn(mean.shape).to(mean.device)
        return latent


class ModelPTL(LightningModule):
    def __init__(self, training, model):
        super().__init__()

        # learning settings
        self.lr = training["lr"]
        self.scheduler_step_size = training["scheduler_step_size"]
        self.scheduler_gamma = training["scheduler_gamma"]

        # log hyperparameters
        self.save_hyperparameters()

        # construct network
        self.model = Model(**model)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, batch_data):
        return self.model(batch_data)

    def training_step(self, batch_data, batch_nb):
        retval = self.forward(batch_data)
        self.training_step_outputs.append(retval)
        return retval

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        losses = self.losses_aggregation(outputs)
        for k, v in losses.items():
            self.log(k, v, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch_data, batch_nb):
        retval = self.forward(batch_data)
        self.validation_step_outputs.append(retval)
        return retval

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        losses = self.losses_aggregation(outputs)
        for k, v in losses.items():
            self.log("val_" + k, v, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(
            optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def losses_aggregation(losses):
        agg_losses = {k: 0.0 for k in losses[0].keys()}
        for loss in losses:
            for k, v in loss.items():
                if k == "seen":
                    agg_losses[k] += float(v)
                else:
                    agg_losses[k] += float(v) * float(loss["seen"])
        for k, v in agg_losses.items():
            if k != "seen":
                agg_losses[k] = agg_losses[k] / agg_losses["seen"]

        return agg_losses

    @staticmethod
    def create_model(
        config,
        bounding_box,
        num_species,
        instanciate=True,  # if false, return model parameteres
    ):
        # construct network
        training_sampling_params = generate_training_sampling_points(
            bounding_box=bounding_box,
            training_sampling_config=config["sampling"]["training"],
        )

        model_params = {
            "num_species": num_species,
            "sampling": training_sampling_params,
            "loss_weights": config["loss_weights"],
            **config["network"],
        }

        params = {
            "training": config["training"],
            "model": model_params,
        }

        if instanciate:
            modelptl = ModelPTL(**params)
            return modelptl
        else:
            return params


if __name__ == "__main__":
    pass
