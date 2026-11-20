import einops
import torch
import torch.nn.functional as F
from torch import nn


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        return self.upsample(x)


class Transformer(nn.TransformerEncoderLayer):
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = super().forward(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groupnorm_n_groups,
        time_embed_dim,
        name="residual",
    ):
        super().__init__()
        self.name = name
        self.time_expand = nn.Linear(time_embed_dim, out_channels)

        self.groupnorm1 = nn.GroupNorm(groupnorm_n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

        self.groupnorm2 = nn.GroupNorm(groupnorm_n_groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embed):
        residual = x

        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # add time embedding
        time_embed = self.time_expand(time_embed)
        x = x + time_embed.view(*time_embed.shape, 1, 1)

        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        # residual connnection
        residual = self.skip_connection(residual) + x
        return residual


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        start_dim=64,
        dim_mults=(1, 2, 4),
        residual_blocks_per_group=1,
        groupnorm_n_groups=16,
        time_embed_dim=128,
    ):
        super().__init__()
        self.input_img_channel = in_channels

        channel_sizes = [start_dim * i for i in dim_mults]
        starting_dim, ending_dim = channel_sizes[0], channel_sizes[-1]

        self.encoder_config = []
        for i, ch in enumerate(channel_sizes):
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((ch, ch), "residual"))

            self.encoder_config.append(((ch, ch), "downsample"))
            self.encoder_config.append((ch, "attention"))

            if i < len(channel_sizes) - 1:
                self.encoder_config.append(((ch, channel_sizes[i + 1]), "upchannel"))

        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_dim, ending_dim), "residual"))

        reversed_encoder_config = self.encoder_config[::-1]

        self.decoder_config = []
        end_dim = ending_dim
        for meta, name in reversed_encoder_config:
            if name == "attention":
                d_model = meta
                self.decoder_config.append((d_model, "attention"))
            elif name == "residual":
                in_ch, out_ch = meta
                self.decoder_config.append(((out_ch + end_dim, in_ch), "residual"))
                end_dim = in_ch
            elif name == "upchannel":
                in_ch, out_ch = meta
                self.decoder_config.append(((out_ch, in_ch), "downchannel"))
                end_dim = in_ch
            elif name == "downsample":
                in_ch, out_ch = meta
                self.decoder_config.append(((in_ch, in_ch), "upsample"))

        # build our model
        self.in_proj = nn.Conv2d(in_channels, start_dim, kernel_size=3, padding="same")

        self.encoder = nn.ModuleList()
        for meta, name in self.encoder_config:
            if name == "attention":
                d_model = meta
                self.encoder.append(
                    Transformer(
                        d_model=d_model,
                        nhead=4,
                        dim_feedforward=int(4 * d_model),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                )
            elif name == "downsample":
                in_ch, out_ch = meta
                self.encoder.append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
                )
            else:  # residual or upchannel
                in_ch, out_ch = meta
                self.encoder.append(
                    ResidualBlock(
                        in_ch, out_ch, groupnorm_n_groups, time_embed_dim, name=name
                    )
                )

        self.bottleneck = nn.ModuleList()
        for meta, name in self.bottleneck_config:
            in_ch, out_ch = meta
            self.bottleneck.append(
                ResidualBlock(in_ch, out_ch, groupnorm_n_groups, time_embed_dim)
            )

        self.decoder = nn.ModuleList()
        for meta, name in self.decoder_config:
            if name == "attention":
                d_model = meta
                self.decoder.append(
                    Transformer(
                        d_model=d_model,
                        nhead=4,
                        dim_feedforward=int(4 * d_model),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                )
            elif name == "upsample":
                in_ch, out_ch = meta
                self.decoder.append(UpSampleBlock(in_ch, out_ch))
            else:  # residual or downchannel
                in_ch, out_ch = meta
                self.decoder.append(
                    ResidualBlock(
                        in_ch, out_ch, groupnorm_n_groups, time_embed_dim, name=name
                    )
                )

        self.out_proj = nn.Conv2d(
            starting_dim, self.input_img_channel, kernel_size=3, padding="same"
        )

    def forward(self, x, time_embed):
        residual_list = []

        x = self.in_proj(x)
        residual_list.append(x)

        # pass through encoder
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, time_embed)
                if module.name == "residual":
                    residual_list.append(x)
            elif isinstance(module, nn.Conv2d):  # downsample
                x = module(x)
            else:  # attention
                x = module(x)

        # pass through bottleneck
        for module in self.bottleneck:
            x = module(x, time_embed)

        # pass through decoder
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                if module.name == "residual":
                    residual = residual_list.pop()
                    x = torch.cat([residual, x], dim=1)
                x = module(x, time_embed)
            elif isinstance(module, UpSampleBlock):
                x = module(x)
            else:  # attention
                x = module(x)

        return self.out_proj(x)


if __name__ == "__main__":
    rb = ResidualBlock(3, 64, 1, 32)
    rand = torch.rand(4, 3, 28, 28)
    time_embed = torch.rand(4, 32)
    output = rb(rand, time_embed)
    print(output.shape)
    usb = UpSampleBlock(64, 64)
    output = usb(output)
    print(output.shape)

    rand_image = torch.randn(4, 3, 64, 64)
    rand_time_embeddings = torch.randn(4, 128)
    unet = UNET()
    output = unet(rand_image, rand_time_embeddings)
    print(output.shape)
