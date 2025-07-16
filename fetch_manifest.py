"""
Filename:		fetch_manifest.py
Description:	Write a parquet file to manifest the whole dataset. Faster loading and reference.
Created:		16th/JULY/2025
Author:			nerveinvader
"""
import os, re, glob
import pandas as pd
import pydicom
import pydicom.dataset

## > Get dicom files
### Pattern of the directory:
### /data/qct##/CQ500CT### CQ500CT###/Unknown Study/CT *str*/CT0000##.dcm
### (### = a number from 0 to 999) (*str* = different strings)
DATA_PATH = ""		## Path that contains folders with pattern: qct**
PID = re.compile(r"CQ500CT(\d{1,3})")	# Capture file names with pattern: CQ500CT***
records = []		## Array that appends .dcm file's metadata and info

### Find patient's folder (pattern: CQ500CT### CQ500CT###)
### and pass it as PID_DIR (patient ID directory).
for fp in glob.glob(f"{DATA_PATH}/qct*/*", recursive=True):
	folder = PID.search(fp)		## Folder names matching with PID (compiled above)
	if not folder:
		continue
	pid = int(folder.group(1))	## Patient number as int
	print(f"CQ500CT{pid}")		# Print patient name
	PID_DIR = fp				## Directory to path for the next step >
	for dcm_file in glob.glob(f"{PID_DIR}/*/*/*.dcm", recursive=True):
		ds: pydicom.dataset.FileDataset = pydicom.dcmread(dcm_file, stop_before_pixels=True)	## Metadata only - no pixel data
		# Write into records >
		records.append({
			"name": pid,
			"series_uid": ds.SeriesInstanceUID,
			"instance_num": ds.get("InstanceNumber", -1),
			"path": fp,
			"slice_thick_mm": float(ds.get("SliceThickness", -1)),
			"series_desc": ds.get("SeriesDescription", ""),
		})

### Write records info (dict) into a parquet file.
### Name of the file: cq500ct_manifest.parquet
### File architecture:
### [name] [series_uid] [instance_num] [path] [slice_thick_mm] [series_desc]
manifest = (pd.DataFrame(records).sort_values(["name", "series_uid", "instance_num"]))
manifest.to_parquet("cq500ct_manifest.parquet", index=False)

print("Wrote", len(manifest), "rows to cq500ct_manifest.parquet file")		# Print final results
