import torch
from torch import nn

from .unet import UNET


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim=128, scaled_time_embed_dim=256):
        super().__init__()
        half_dim = time_embed_dim // 2
        inv_freqs = 1.0 / 10000 ** (torch.arange(half_dim) / (half_dim - 1))
        self.register_buffer("inv_freqs", inv_freqs[None, :], persistent=False)
        self.time_scaled = nn.Sequential(
            nn.Linear(time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
            nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
        )

    def forward(self, timesteps):
        inv_freqs = timesteps.unsqueeze(1) * self.inv_freqs
        embed = torch.cat([inv_freqs.sin(), inv_freqs.cos()], dim=-1)
        return self.time_scaled(embed)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        start_dim=64,
        dim_mults=(1, 2, 4),
        residual_blocks_per_group=1,
        groupnorm_n_groups=16,
        time_embed_dim=128,
        time_embed_dim_ratio=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.time_embed_dim = time_embed_dim
        self.scaled_time_embed_dim = int(time_embed_dim_ratio * time_embed_dim)

        self.sinusoidal_embed = SinusoidalTimeEmbedding(
            time_embed_dim=time_embed_dim,
            scaled_time_embed_dim=self.scaled_time_embed_dim,
        )

        self.unet = UNET(
            in_channels=in_channels,
            start_dim=start_dim,
            dim_mults=dim_mults,
            residual_blocks_per_group=residual_blocks_per_group,
            groupnorm_n_groups=groupnorm_n_groups,
            time_embed_dim=self.scaled_time_embed_dim,
        )

    def forward(self, noisy_images, timesteps):
        time_embed = self.sinusoidal_embed(timesteps)

        pred_noise = self.unet(noisy_images, time_embed)
        return pred_noise


if __name__ == "__main__":
    st = SinusoidalTimeEmbedding()
    timesteps = torch.tensor([1, 2, 3])
    output = st(timesteps)
    print(output.shape)
    diff = DiffusionModel()
    rand_image = torch.rand(4, 3, 128, 128)
    rand_time_embed = torch.randint(0, 128, (4,))
    output = diff(rand_image, rand_time_embed)
    print(output.shape)
