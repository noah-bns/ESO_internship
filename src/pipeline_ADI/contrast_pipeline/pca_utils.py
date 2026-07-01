import torch
import numpy as np
import gc
import time
from typing import Tuple, Callable, Optional
from typing import List, Dict, Union
from datetime import datetime
import time

# Import your custom classes
from applefy.utils.field_rotation import FieldRotationModel
from fours.models.rotation import FieldRotationModel
from applefy.detections.contrast import DataReductionInterface, Path
from fours.utils.pca import pca_tensorboard_logging


def _get_pca_fitter(
    pca_method: str,
    n_components: int,
    n_samples: int,
    n_features: int,
    gram_threshold: float = 0.5,
    oversample: int = 5,
    niter: int = 2,
    eps: float | None = None,
    approx_svd_trunc: int | None = None,
) -> Tuple[Callable, str]:
    """
    Select and return the appropriate PCA fitting function based on method and data shape.
    
    Returns a callable that takes (X) and returns (components, singular_values).
    """
    
    # Auto-select method if needed
    if pca_method == "auto":
        if n_components >= 0.8 * min(n_samples, n_features):
            method = "svd"
        elif n_samples <= gram_threshold * n_features:
            method = "gram"
        else:
            method = "lowrank"
    else:
        method = pca_method
    
    if method == "svd":
        def fitter(X: torch.Tensor) -> torch.Tensor:
            """Exact reduced SVD."""
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            components = Vh[:n_components]
            #singular_values = S[:n_components]
            return components #, singular_values
        
        return fitter, method
    
    elif method == "gram":
        def fitter(X: torch.Tensor) -> torch.Tensor:
            """Exact PCA via sample Gram matrix."""
            eps_val = eps if eps is not None else float(torch.finfo(X.dtype).eps)
            
            C = X @ X.T  # (n_samples, n_samples)
            evals, U = torch.linalg.eigh(C)
            
            # Sort by largest eigenvalues
            idx = torch.argsort(evals, descending=True)
            evals = evals[idx]
            U = U[:, idx]
            
            evals = evals[:n_components]
            U = U[:, :n_components]
            
            singular_values = torch.sqrt(torch.clamp(evals, min=0.0))
            
            # Filter out invalid singular values
            valid = singular_values > eps_val
            if not torch.any(valid):
                raise RuntimeError("All singular values are numerically zero.")
            
            U_valid = U[:, valid]
            S_valid = singular_values[valid]
            
            # Compute right singular vectors: V = X.T @ U @ diag(1/S)
            V = X.T @ (U_valid / S_valid)
            
            # Improve numerical orthogonality
            V, _ = torch.linalg.qr(V, mode="reduced")
            
            components = V[:, :n_components].T
            
            # Pad if needed to maintain consistent shapes
            if components.shape[0] < n_components:
                missing = n_components - components.shape[0]
                pad_components = torch.zeros(
                    missing,
                    X.shape[1],
                    device=X.device,
                    dtype=X.dtype,
                )
                components = torch.cat([components, pad_components], dim=0)
                
            #     pad_singular_values = torch.zeros(
            #         missing,
            #         device=X.device,
            #         dtype=X.dtype,
            #     )
            #     singular_values = torch.cat([S_valid, pad_singular_values], dim=0)
            # else:
            #     singular_values = singular_values[:n_components]
            
            return components #, singular_values
        
        return fitter, method
    
    elif method == "lowrank":
        def fitter(X: torch.Tensor) -> torch.Tensor:
            """Randomized approximate low-rank PCA."""
            q = approx_svd_trunc if approx_svd_trunc is not None else min(
                n_components + oversample,
                min(X.shape),
            )
            
            _, S, V = torch.pca_lowrank(
                X,
                q=q,
                center=False,
                niter=niter,
            )
            
            components = V[:, :n_components].T
            #singular_values = S[:n_components]
            return components #, singular_values
        
        return fitter, method
    
    else:
        raise ValueError(f"Invalid PCA method: {pca_method}")
    

# def pca_psf_subtraction_gpu(
#         images: np.ndarray,
#         angles: np.ndarray,
#         pca_numbers: np.ndarray,
#         device,
#         approx_svd: int = -1,
#         subsample_rotation_grid: int = 1,
#         verbose: bool = False,
#         combine: str = "mean"
# ) -> np.ndarray:

def pca_psf_subtraction_gpu(
        images: np.ndarray,
        angles: np.ndarray,
        pca_numbers: np.ndarray,
        device: str = "auto",
        pca_method: str = "auto",
        oversample: int = 5,
        niter: int = 2,
        gram_threshold: float = 0.5,
        random_state: int | None = None,
        eps: float | None = None,
        approx_svd_trunc: int | None = None,
        subsample_rotation_grid: int = 1,
        verbose: bool = False,
        combine: str = "mean",
        dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    PCA-based PSF subtraction using different PCA methods.
    
    Parameters
    ----------
    images : np.ndarray
        Image cube with shape (n_frames, height, width)
    angles : np.ndarray
        Parallactic angles with shape (n_frames,) in radians
    pca_numbers : np.ndarray
        PCA component numbers to evaluate
    device : str
        Device: "auto" (GPU if available), "cuda", or "cpu"
    pca_method : str
        PCA method: "auto", "svd", "gram", or "lowrank"
    oversample : int
        Oversampling for lowrank method
    niter : int
        Power iterations for lowrank method
    gram_threshold : float
        Threshold for auto-switching to gram method
    random_state : int, optional
        Random seed for reproducibility
    eps : float, optional
        Epsilon for numerical stability
    approx_svd_trunc : int, optional
        Truncation rank for lowrank approximation
    subsample_rotation_grid : int
        Subsampling for field rotation model
    verbose : bool
        Print progress information
    combine : str
        Combination method: "mean" or "median"
    dtype : torch.dtype
        Data type for tensors
    
    Returns
    -------
    np.ndarray
        Residual images with shape (len(pca_numbers), height, width)
    """


    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    pca_numbers = np.asarray(pca_numbers, dtype=int)

    if pca_numbers.ndim != 1:
        raise ValueError(f"Expected pca_numbers to be 1D, got {pca_numbers.shape}.")

    if np.any(pca_numbers < 1):
        raise ValueError("All PCA numbers should be >= 1.")
    

    with torch.no_grad():
        t0 = time.perf_counter()

        # 1.) Convert images to torch tensor
        im_shape = images.shape
        #images_torch = torch.from_numpy(images).to(device)
        n_frames, height, width = im_shape
        images_torch = torch.as_tensor(images, device=device, dtype=dtype)

        t1 = time.perf_counter()
        print(f"[Timing] Convert to tensor: {t1 - t0:.6f}s")

        # 2.) remove the mean as needed for PCA
        images_torch = images_torch - images_torch.mean(dim=0)

        # 3.) reshape images to fit for PCA
        #images_flat = images_torch.view(im_shape[0], im_shape[1] * im_shape[2])
        images_flat = images_torch.reshape(n_frames, height * width)
        
        # 4.) Fit PCA using PCATorch
        if verbose:
            print(f"Fit PCA ({pca_method}) ...", end="")

        
        #### 
        # based off https://github.com/markusbonse/near_processing/blob/main/near_processing/utils/pca.py#L11 PCATorch class
        ####

        # pca = PCATorch(
        #     n_components=max_components,
        #     method=pca_method,
        #     center=True,  # Remove mean
        #     device=device,
        #     dtype=images_torch.dtype,
        #     oversample=oversample,
        #     niter=niter,
        #     gram_threshold=gram_threshold,
        #     random_state=random_state,
        #     eps=eps,
        # )
        
        # pca.fit(images_flat)

        # Get the appropriate PCA fitter

        max_components = int(np.max(pca_numbers))
        n_samples, n_features = images_flat.shape
        max_rank = min(n_samples, n_features)

        if max_components > max_rank:
            raise ValueError(
                f"n_components={max_components} is larger than "
                f"min(n_samples, n_features)={max_rank}."
            )

        if random_state is not None:
            torch.manual_seed(random_state)

        pca_fitter, method_used = _get_pca_fitter(
            pca_method=pca_method,
            n_components=max_components,
            n_samples=n_frames,
            n_features=height * width,
            gram_threshold=gram_threshold,
            oversample=oversample,
            niter=niter,
            eps=eps,
            approx_svd_trunc=approx_svd_trunc
        )

        # Fit PCA once
        components = pca_fitter(images_flat)

        if verbose:
            print(f"[DONE] (method: {method_used})")

        t2 = time.perf_counter()
        print(f"[Timing] Compute PCA Basis: {t2 - t1:.6f}s")

        # 5.) Build rotation model
        rotation_model = FieldRotationModel(
            all_angles=angles,
            input_size=im_shape[1],
            subsample=subsample_rotation_grid,
            inverse=False,
            register_grid=True
        ).to(device)

        t3 = time.perf_counter()
        print(f"[Timing] Field rotation: {t3 - t2:.6f}s")

        # 6.) Compute PCA residuals for all given PCA numbers
        pca_residuals = []
        if verbose:
            print("Compute PCA residuals ...", end="")

        for pca_number in pca_numbers:
            pca_number = int(pca_number)


            # Project onto PCA components
            pca_scores = images_flat @ components.T  # shape: (n_frames, pca_number)
            
            # Reconstruct noise model
            noise_estimate = pca_scores @ components  # shape: (n_frames, n_features)
            
            # Compute residuals
            residual = images_flat - noise_estimate
            residual_sequence = residual.view(im_shape[0], im_shape[1], im_shape[2])
            del residual, noise_estimate, pca_scores

            # Subtract temporal median if not using mean combine
            if combine != "mean":
                residual_sequence = residual_sequence - torch.median(residual_sequence, dim=0)[0]

            # Derotate frames
            rotated_frames = rotation_model(
                residual_sequence.unsqueeze(1).float(),
                parang_idx=torch.arange(len(residual_sequence), device=device)
            ).squeeze(1)
            del residual_sequence

            # Combine derotated frames
            if combine == "mean":
                residual_final = torch.mean(rotated_frames, dim=0).cpu().numpy()
            elif combine == "median":
                residual_final = torch.median(rotated_frames, dim=0)[0].cpu().numpy()
            else:
                raise ValueError(f"Invalid combine method: {combine}")
            
            pca_residuals.append(residual_final)
            del rotated_frames, residual_final

            if verbose:
                print("[DONE]")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        t4 = time.perf_counter()
        print(f"[Timing] PCA Residuals: {t4 - t3:.6f}s")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.3f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.3f}GB")

        return np.array(pca_residuals)
    


class PCADataReductionGPU(DataReductionInterface):
    """
    GPU-accelerated PCA data reduction with flexible PCA methods.
    """

    def __init__(
            self,
        pca_numbers: np.ndarray,
        work_dir: Union[str, Path] = None,
        special_name: str = None,
        device: str = "auto",
        pca_method: str = "auto",
        oversample: int = 5,
        niter: int = 2,
        gram_threshold: float = 0.5,
        random_state: int | None = None,
        eps: float | None = None,
        approx_svd_trunc: int | None = None,
        subsample_rotation_grid: int = 1,
        combine: str = "mean",
        verbose: bool = False,
    ):
        """
        Initialize PCA Data Reduction.
        
        Parameters
        ----------
        pca_numbers : np.ndarray
            PCA component numbers to test
        work_dir : Path, optional
            Directory for checkpoints/logging
        special_name : str, optional
            Special name for results
        device : str
            Device: "auto", "cuda", or "cpu"
        pca_method : str
            PCA method: "auto", "svd", "gram", or "lowrank"
        oversample : int
            Oversampling for lowrank
        niter : int
            Power iterations for lowrank
        gram_threshold : float
            Threshold for auto gram selection
        random_state : int, optional
            Random seed
        eps : float, optional
            Epsilon for numerical stability
        approx_svd_trunc : int, optional
            SVD truncation (for backward compatibility)
        subsample_rotation_grid : int
            Subsampling for rotation model
        combine : str
            Combination method: "mean" or "median"
        verbose : bool
            Print progress
        """
        self.pca_numbers = pca_numbers
        self.device = device
        self.pca_method = pca_method
        self.oversample = oversample
        self.niter = niter
        self.gram_threshold = gram_threshold
        self.random_state = random_state
        self.eps = eps
        self.approx_svd_trunc = approx_svd_trunc
        self.subsample_rotation_grid = subsample_rotation_grid
        self.combine = combine
        self.verbose = verbose
        
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.special_name = special_name if special_name is not None else ""


    def get_method_keys(self) -> List[str]:
        """Get result dictionary keys for each PCA number."""

        keys = [self.special_name + "_PCA_" + str(num_pcas).zfill(3) +
                "_components" for num_pcas in self.pca_numbers]

        return keys

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Execute PCA reduction.
        
        Parameters
        ----------
        stack_with_fake_planet : np.ndarray
            Science image cube with injected fake planets
        parang_rad : np.ndarray
            Parallactic angles in radians
        psf_template : np.ndarray
            PSF template (not used in new version)
        exp_id : str
            Experiment ID for logging
        
        Returns
        -------
        dict
            Dictionary with PCA residuals for each component count
        """
        # Call the new PCA function
        pca_residuals = pca_psf_subtraction_gpu(
            images=stack_with_fake_planet,
            angles=parang_rad,
            pca_numbers=self.pca_numbers,
            device=self.device,
            pca_method=self.pca_method,
            oversample=self.oversample,
            niter=self.niter,
            gram_threshold=self.gram_threshold,
            random_state=self.random_state,
            eps=self.eps,
            approx_svd_trunc=self.approx_svd_trunc,
            subsample_rotation_grid=self.subsample_rotation_grid,
            verbose=self.verbose,
            combine=self.combine
        )

        if self.work_dir is not None:
            time_str = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            current_logdir = self.work_dir / \
                Path(exp_id + "_" + self.special_name + "_PCA_" + time_str)
            current_logdir.mkdir(exist_ok=True, parents=True)

            pca_tensorboard_logging(
                log_dir=current_logdir,
                pca_residuals=pca_residuals,
                pca_numbers=self.pca_numbers)

        result_dict = dict()
        for idx, tmp_algo_name in enumerate(self.get_method_keys()):
            result_dict[tmp_algo_name] = pca_residuals[idx]

        return result_dict
