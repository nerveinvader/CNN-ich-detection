"""
File name:		data_loader.py
Description:	Data loading pipeline inherited from `torch.utils.data.Dataset`.
Created:		16th/JULY/2025
Author:			nerveinvader
"""

# Imports
import cv2
import numpy as np
from pandas import DataFrame
import torch
import pydicom
import pydicom.dataset
from pydicom.pixel_data_handlers.util import _apply_modality_lut


# Class
class CQ500Dataset(torch.utils.data.Dataset):
	"""
	Connects `cq500ct_manifest.parquet` and `label.csv` and turns into Pytorch `tensors`.
	Returns x (shape), y (torch tensor float32) could be used as tran_ds.
	"""

	def __init__(
		self, manifest_df: DataFrame, labels_df: DataFrame, transform=None
	) -> None:
		"""
		manifest_df is the `parquet` file created from the whole dataset.
		labels_df is the `csv` file created from `reads.csv`.
		"""
		self.mf = manifest_df
		self.lbl = labels_df.set_index("name")  # name as index
		# self.ids become the master ids list, serving the pipeline.
		# find the unique and pixel + labels scans
		self.ids = self.lbl.index.intersection(self.mf["name"].unique())
		self.tf = transform

	def __len__(self) -> int:
		"""Report the number of studies (not slices) as `int`."""
		return len(self.ids)

	def __getitem__(self, idx) -> tuple:
		study = self.ids[idx]  ## Pick a study (not slices)
		df = self.mf[
			self.mf["name"] == study
		]  ## All the rows belonging to this study (chosen index) -> single patient row in manifest
		sid = df["series_uid"].iloc[0]  ## Pick one series #! ???
		slices = df[
			df["series_uid"] == sid
		]  ## All slices (rows) for the selected series within the study
		volume = [to_windowed_tensor(pydicom.dcmread(p)) for p in slices["path"]]
		x = torch.stack(
			volume
		)  ## Stack per-slice tensors into a 4D batch [num_slice, num_chan, h, w]
		y = self.lbl.loc[
			study, "ICH-majority"
		]  ## Target scalar label (soft or majority)
		if self.tf:
			x = self.tf(x)
		return x, torch.tensor(y, dtype=torch.float32)


# Helper
# DICOM to Tensor
def to_windowed_tensor(
	ds: pydicom.dataset.FileDataset,
	windows: list[tuple[int, int]] = [(40, 80), (80, 200), (600, 2800)],
	out_size: tuple[int, int] = (256, 256),
	dtype: torch.dtype = torch.float32,
) -> torch.tensor:
	"""
	Helper: convert one DICOM slice into a torch.Tensor [C x H x W]
	ds			= single CQ500 DICOM slice.
	windows		= CT window to apply. Each tuple becomes one output channel.
			Default = Brain, Subdural, Bone.
	out_size	= (H, W) to resize the slice.
	dtype		= final tensor percision (float32 recommended).

	Returns		= torch.tensor (shape = C x H x W with normalized values)
	"""
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
