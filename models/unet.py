import torch
from monai.networks.nets import UNet

from .base import BaseModel


class UNetModel(BaseModel):
    """3D U-Net model for FeTS 2022 brain tumor segmentation."""

    def build(self):
        return UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=("INSTANCE", {"affine": True}),
        )

_CACHED_IN_KEYS: set[str] | None = None

def instance_norm_state_keys() -> set[str]:
    """
    Return state-dict keys belonging to InstanceNorm layers.

    These parameters/statistics are kept local in FedIN-EDAR
    and are not aggregated by the server.
    """
    global _CACHED_IN_KEYS
    if _CACHED_IN_KEYS is not None:
        return _CACHED_IN_KEYS
    
    model = UNetModel().build()

    keys: set[str] = set()

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
            prefix = f"{module_name}." if module_name else ""

            for state_name in module.state_dict().keys():
                keys.add(f"{prefix}{state_name}")

    _CACHED_IN_KEYS = keys
    return keys


_CACHED_OUTPUT_KEYS: tuple[str, str | None] | None = None

def output_conv_state_keys() -> tuple[str, str | None]:
    """Return the state_dict keys for the final 4-class output convolution."""

    global _CACHED_OUTPUT_KEYS
    if _CACHED_OUTPUT_KEYS is not None:
        return _CACHED_OUTPUT_KEYS

    model = UNetModel().build()
    weight_key, bias_key = None, None

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv._ConvNd) and getattr(module, "out_channels", None) == 4:
            prefix = f"{module_name}." if module_name else ""
            weight_key = f"{prefix}weight"
            bias_key = f"{prefix}bias" if module.bias is not None else None

    if weight_key is None:
        raise RuntimeError("Could not locate the final 4-channel output convolution in UNet.")

    _CACHED_OUTPUT_KEYS = (weight_key, bias_key)
    return _CACHED_OUTPUT_KEYS