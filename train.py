import math

import hydra
import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch.callbacks import TQDMProgressBar
from safetensors.torch import save_model

from src.data import DataModule
from src.modules.ddpm import DiffusionModel
from src.modules.sampler import Sampler


class Model(L.LightningModule):
    def __init__(self, net, sampler, learning_rate=5e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "sampler"])
        self.net = net
        self.sampler = sampler
        self.total_steps = self.sampler.num_steps

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        bs = batch.shape[0]
        device = batch.device

        timesteps = torch.randint(0, self.total_steps, (bs,), device=device)
        noisy_images, noise = self.sampler.add_noise(batch, timesteps)

        pred_noise = self.net(noisy_images, timesteps)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/lr", sch.get_last_lr()[0], prog_bar=True, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        """No need to anything but we need it for on_validation_epoch_end"""
        return

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        batch = next(
            iter(self.trainer.val_dataloaders)
        )  # load batch just tot image dimension
        device = self.sampler.betas.device
        n_imgs = 4

        imgs_over_timesteps = [[] for _ in range(n_imgs)]
        store_freq = self.total_steps // 10

        images = torch.randn(n_imgs, *batch.shape[1:], device=device)
        for t in reversed(range(self.total_steps)):
            ts = torch.full((n_imgs,), t, device=device)
            pred_noise = self.net(images, ts)
            images = self.sampler.remove_noise(images, ts, pred_noise)
            if t % store_freq == 0:
                for i, img in enumerate(images):
                    imgs_over_timesteps[i].append(img)

        imgs_over_timesteps = torch.stack(
            [torch.stack(all_steps) for all_steps in imgs_over_timesteps]
        )
        imgs_over_timesteps = imgs_over_timesteps.reshape(-1, *batch.shape[1:])
        self.logger.experiment.add_image(
            "val/gen_imgs",
            torchvision.utils.make_grid(imgs_over_timesteps, nrow=10),
            self.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.hparams.learning_rate
        )
        optimizer_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(0.05 * optimizer_steps)

        def cosine_with_warmup(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            progress = float(step - warmup_steps) / float(
                max(1.0, optimizer_steps - warmup_steps)
            )
            progress = min(1.0, progress)  # keep to floor at the end of training
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    L.seed_everything(cfg.seed)
    dm = DataModule(**cfg.data)
    net = DiffusionModel(**cfg.net)
    sampler = Sampler(num_steps=cfg.sampler.total_steps)
    model = Model(net, sampler, **cfg.model)
    trainer = L.Trainer(
        check_val_every_n_epoch=None,
        callbacks=[TQDMProgressBar(refresh_rate=50)],
        **cfg.trainer,
    )
    trainer.fit(model, dm)
    save_model(net, "model.safetensors")


if __name__ == "__main__":
    main()
