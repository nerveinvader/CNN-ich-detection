"""
File name:		dcm_to_tensor.py
Description:	Loads a dcm file and turn it into a windowed tensor.
Created:		22nd/JULY/2025
Author:			nerveinvader
"""

import cv2
import numpy as np
import pydicom
import pydicom.dataset
from pydicom.pixel_data_handlers.util import _apply_modality_lut
import torch


# Class
class DCMLoader():
    """
    Loads a DCM file to do various utilities on it.
    dcm_to_tensor() -> turn DCM into Torch Tensor.
    """
    ## Window presets from RSNA ICH Challenge
    WINDOW_PRESETS = {
        "brain": {"level": 40, "width": 80},
        "subdural": {"level": 80, "width": 200},
        "bone": {"level": 600, "width": 2800},
    }

    ## DICOM to Tensor
    def dcm_to_tensor(
        self, path: str,
        windows: list[tuple[int, int]] = None,
        out_size: tuple[int, int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Helper: convert one DICOM slice into a torch.Tensor [C x H x W]
        ds			= single CQ500 DICOM slice.
        windows		= CT window to apply. Each tuple becomes one output channel.
                        Default = Brain, Subdural, Bone.
        out_size	= (H, W) to resize the slice.
        dtype		= final tensor percision (float32 recommended).

        Returns		= torch.tensor (shape = C x H x W with normalized values)
        """
        ## Handle None args
        if windows is None:
            windows = [(40, 80), (80, 200), (600, 2800)]
        if out_size is None:
            out_size = (256, 256)

        ds: pydicom.dataset.FileDataset = pydicom.dcmread(str(path))

        ## Raw values > Hounsfield units (HU)
        hu: np.ndarray = _apply_modality_lut(ds.pixel_array, ds).astype(np.int16)
        if out_size is not None and hu.shape != out_size:
            hu = cv2.resize(hu, out_size[::-1], interpolation=cv2.INTER_LINEAR)

        ## Window / Level > 0-1 float per channel
        ## 3 channels to feed into model
        chans: list[np.ndarray] = []
        for level, width in windows:
            level: int
            width: int
            lower: int = level - (width // 2)
            upper: int = level + (width // 2)
            img_clipped: np.ndarray = np.clip(hu, lower, upper)
            img_norm: np.ndarray = (img_clipped - lower) / float(width)  # 0 - 1
            chans.append(img_norm.astype(np.float32))

        ## Stack and convert to tensor
        arr: np.ndarray = np.stack(chans, axis=0)  # C x H x W
        tensor: torch.Tensor = torch.from_numpy(arr).type(dtype)

        return tensor
