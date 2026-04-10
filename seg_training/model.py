from monai.networks.nets import UNet

from .config import TrainConfig


def build_model(config: TrainConfig) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
    )
