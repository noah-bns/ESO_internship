import torch
import numpy as np
from typing import List
import time

# Import your custom classes
from fours.models.rotation import FieldRotationModel
from applefy.detections.contrast import DataReductionInterface


class RawContrastReductionGPU(DataReductionInterface):

    def __init__(self, device):
        self.device = device

    def get_method_keys(self) -> List[str]:
        return ["raw_contrast"]

    def __call__(
        self,
        stack_with_fake_planet: np.ndarray,
        parang_rad: np.ndarray,
        psf_template: np.ndarray,
        exp_id: str,
    ):

        residual = raw_contrast_gpu(
            images=stack_with_fake_planet, angles=parang_rad, device=self.device
        )

        result_dict = dict()
        result_dict["raw_contrast"] = residual

        return result_dict


def raw_contrast_gpu(device, images: np.ndarray, angles: np.ndarray):

    images = torch.from_numpy(images).to(device)

    # No mean/median subtraction:
    # keep the original frames as-is.
    residual_sequence = images

    # Combine residual stack on GPU
    rotation_model = FieldRotationModel(
        all_angles=angles,
        input_size=images.shape[1],
        subsample=1,
        inverse=False,
        register_grid=True,
    )

    rotation_model = rotation_model.to(device)

    rotated_frames = rotation_model(
        residual_sequence.unsqueeze(1).float(),
        parang_idx=torch.arange(len(residual_sequence), device=device),
    ).squeeze(1)

    residual_image = torch.mean(rotated_frames, axis=0).cpu().numpy()

    return residual_image
