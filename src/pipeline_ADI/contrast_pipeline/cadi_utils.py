from typing import List, Dict, Union
import numpy as np
from pathlib import Path
from datetime import datetime

import multiprocessing
from scipy.ndimage import rotate
from tqdm import tqdm

import torch
from astropy.io import fits

from applefy.detections.contrast import DataReductionInterface
from fours.models.psf_subtraction import FourS
from fours.models.rotation import FieldRotationModel

# adapted script from fours.utils.adi_tools, with the additional option to either subtract the mean or median in cadi
# - combine and substraction options


def cadi_fits_logging(
    log_dir: Union[str, Path],
    noise_frame: Union[np.ndarray, torch.Tensor],
) -> None:
    """Write the cADI temporal noise model as a FITS image."""

    noise_frames_dir = Path(log_dir) / "Noise_frames"
    noise_frames_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(noise_frame, torch.Tensor):
        noise_frame = noise_frame.detach().cpu().numpy()
    else:
        noise_frame = np.asarray(noise_frame)

    if noise_frame.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional cADI noise frame, got {noise_frame.shape}."
        )

    fits.writeto(
        noise_frames_dir / "cADI.fits",
        data=noise_frame,
        overwrite=True,
    )


def cadi_psf_subtraction(
    images: np.ndarray,
    angles: np.ndarray,
    subtraction: str = "mean",
    combine: str = "mean",
    noise_visualise: bool = False,
    work_dir: Union[str, Path, None] = None,
) -> Union[np.ndarray, None]:

    if subtraction not in {"mean", "median"}:
        raise ValueError("subtraction must be 'mean' or 'median'")

    if combine not in {"mean", "median"}:
        raise ValueError("combine must be 'mean' or 'median'")

    if subtraction == "median":
        sub_frame = np.median(images, axis=0)
        print("Subtracting a median.")
    if subtraction == "mean":
        sub_frame = np.mean(images, axis=0)
        print("Subtracting a mean.")

    if noise_visualise:
        if work_dir is None:
            raise ValueError("work_dir must be provided when noise_visualise=True.")
        cadi_fits_logging(log_dir=work_dir, noise_frame=sub_frame)
        return None

    residual_sequence = images - sub_frame

    residual_image = combine_residual_stack(
        residual_stack=residual_sequence, angles=angles, combine=combine
    )

    return residual_image


def cadi_psf_subtraction_gpu(
    device,
    images: np.ndarray,
    angles: np.ndarray,
    subtraction: str = "mean",
    combine: str = "mean",
    noise_visualise: bool = False,
    work_dir: Union[str, Path, None] = None,
) -> Union[np.ndarray, None]:

    if subtraction not in {"mean", "median"}:
        raise ValueError("subtraction must be 'mean' or 'median'")

    if combine not in {"mean", "median"}:
        raise ValueError("combine must be 'mean' or 'median'")

    images = torch.from_numpy(images).to(device)

    if subtraction == "median":
        sub_frame = torch.median(
            images, dim=0
        ).values  # before: torch.median(images, axis=0)[0]
        print("Subtracting a median.")
    if subtraction == "mean":
        sub_frame = torch.mean(images, dim=0)
        print("Subtracting a mean.")

    if noise_visualise:
        if work_dir is None:
            raise ValueError("work_dir must be provided when noise_visualise=True.")
        cadi_fits_logging(log_dir=work_dir, noise_frame=sub_frame)
        return None

    residual_sequence = images - sub_frame

    # combine residual stack on GPU
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

    if combine == "median":
        residual_image = torch.median(rotated_frames, dim=0).values
    else:
        residual_image = torch.mean(
            rotated_frames, dim=0
        )  # before: called axis instead of dim

    return residual_image.cpu().numpy()


def combine_residual_stack(
    residual_stack, angles, combine="mean", subtract_temporal_average=False, num_cpus=4
):

    if combine == "mean":
        temporal_average = np.mean(residual_stack, axis=0)
    else:
        temporal_average = np.median(residual_stack, axis=0)

    if subtract_temporal_average:
        residual_stack = residual_stack - temporal_average

    if num_cpus == 1:
        de_rotated = []

        for i in tqdm(range(residual_stack.shape[0])):
            tmp_frame = rotate(
                residual_stack[i, :, :], -np.rad2deg(angles[i]), (0, 1), False
            )
            de_rotated.append(tmp_frame)

    else:
        arguments = [
            (residual_stack[i, :, :], -np.rad2deg(angles[i]), (0, 1), False)
            for i in range(residual_stack.shape[0])
        ]

        pool = multiprocessing.Pool(num_cpus)
        de_rotated = np.array(pool.starmap(rotate, arguments))
        pool.close()

    if combine == "mean":
        return np.mean(de_rotated, axis=0)
    else:
        return np.median(de_rotated, axis=0)


class CADIDataReduction(DataReductionInterface):

    def __init__(
        self,
        subtraction: str | None = None,
        combine: str | None = None,
        work_dir: Union[str, Path, None] = None,
        noise_visualise: bool = False,
    ):
        self.subtraction = subtraction
        self.combine = combine
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.noise_visualise = noise_visualise

    def get_method_keys(self) -> List[str]:
        return [
            "cADI",
        ]

    def __call__(
        self,
        stack_with_fake_planet: np.ndarray,
        parang_rad: np.ndarray,
        psf_template: np.ndarray,
        exp_id: str,
    ):

        optional_args = {}

        if self.subtraction is not None:
            optional_args["subtraction"] = self.subtraction

        if self.combine is not None:
            optional_args["combine"] = self.combine

        residual = cadi_psf_subtraction(
            images=stack_with_fake_planet,
            angles=parang_rad,
            noise_visualise=self.noise_visualise,
            work_dir=self.work_dir,
            **optional_args,
        )

        if self.noise_visualise:
            return {}

        return {"cADI": residual}


class CADIDataReductionGPU(DataReductionInterface):

    def __init__(
        self,
        device,
        subtraction: str | None = None,
        combine: str | None = None,
        work_dir: Union[str, Path, None] = None,
        noise_visualise: bool = False,
    ):
        self.device = device
        self.subtraction = subtraction
        self.combine = combine
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.noise_visualise = noise_visualise

    def get_method_keys(self) -> List[str]:
        return [
            "cADI",
        ]

    def __call__(
        self,
        stack_with_fake_planet: np.ndarray,
        parang_rad: np.ndarray,
        psf_template: np.ndarray,
        exp_id: str,
    ):

        optional_args = {}

        if self.subtraction is not None:
            optional_args["subtraction"] = self.subtraction

        if self.combine is not None:
            optional_args["combine"] = self.combine

        residual = cadi_psf_subtraction_gpu(
            images=stack_with_fake_planet,
            angles=parang_rad,
            device=self.device,
            noise_visualise=self.noise_visualise,
            work_dir=self.work_dir,
            **optional_args,
        )

        if self.noise_visualise:
            return {}

        return {"cADI": residual}
