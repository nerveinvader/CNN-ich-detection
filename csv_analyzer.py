#%%
# >>> imports
import os, re, glob
from numpy import record
import pandas as pd
import pydicom

#%%
# >>> loading and reading the CSV files
reads = pd.read_csv("csv-files/reads.csv")
compact_reads = pd.read_csv("csv-files/compact_reads.csv")
# reads.head()

# Value check of reads.csv file,
# Brief is below this code
#for col in reads.columns:
#	print("Values per column: ", reads[col].value_counts())

#* to extract info from the dataset:
# b1_count = 0
# b2_count = 0
# for i, r in reads.iterrows():
# 	# print(f"Patient {r['name']}/{r['Category']} = R1: {r['R1:ICH']}/R2: {r['R2:ICH']}/R3: {r['R3:ICH']}")
# 	if r['R1:ICH'] == 1 or r['R2:ICH'] == 1 or r['R3:ICH'] == 1:
# 		if r['Category'] == 'B1':
# 			b1_count += 1
# 		if r['Category'] == 'B2':
# 			b2_count += 1
# print(f"B1 ICH: {b1_count}\nB2 ICH: {b2_count}")

#* the data shows:
# > dtypes are int64
# > R1-2-3: ICH, IPH, IVH, SDH, EDH, SAH
# > ~: BleedLocation-Left / -Right
# > ~: ChronicBleed, Fracture, CalvarialFracture, OtherFracture
# > ~: MassEffect, MisslineShift
# > ~: No missing values
# > the R1-3 are readers and each reader has 14 features
# "name" is the unique id to check patients
# 491 patients
# B1 is real-world appearance of ICH in a 30 days period	=~ 214 (050 - 23%)
# B2 is enriched in ICH (double checked)					=~ 277 (196 - 70%)
# Agreement of the radiologist differ on the scans:
# Only 1 Rad agrees = 41 scans -> we can check with a radiologist
# Only 2 Rads agree = 37 scans
# All  3 Rads agree = 168 scans

# %%
# >>> extracting a compact csv file
# reads['ICH-sum']		= reads[['R1:ICH', 'R2:ICH', 'R3:ICH']].sum(1)
# reads['ICH-majority']	= (reads['ICH-sum'] >= 2).astype(int)
# reads['ICH-soft']		= reads['ICH-sum'] / 3
# compact_reads = reads[['name', 'ICH-sum', 'ICH-majority', 'ICH-soft']]
# compact_reads.to_csv('compact_reads.csv', index=False)
# compact_reads.head()

#%%
# >>> loading data with glob module (file and folders are inconsistent with dataset)
DATA_ROOT = "data/"	# the dataset root (contains qct19 locally)
# CQ500CT9 CQ500CT9 patient folder pattern
# CQ500CT** CQ500CT**/Unknown Study/Plain **/.dcm files
PID = re.compile(r"CQ500CT(\d{1,3})") # 0-999
records = []

for fp in glob.glob(f"{DATA_ROOT}/qct*/*", recursive=True):
	folder = PID.search(fp)		# folder names
	if not folder:
		continue
	pid = int(folder.group(1))	# numbers
	print(f"CQ500CT{pid}")		# not zero padded - main patient folder
	DIR = fp
	for dcm_file in glob.glob(f"{DIR}/*/*/*.dcm", recursive=True):
		print(dcm_file)
		ds: pydicom.FileDataset = pydicom.dcmread(dcm_file, stop_before_pixels=True)	## Metadata only
		records.append({
			"name": pid,
			"series_uid": ds.SeriesInstanceUID,
			"instance_num": ds.get("InstanceNumber", -1),
			"path": fp,
			"slice_thick_mm": float(ds.get("SliceThickness", -1)),
			"series_desc": ds.get("SeriesDescription", ""),
		})
manifest = (pd.DataFrame(records).sort_values(["name", "series_uid", "instance_num"]))
manifest.to_parquet("cq500ct_qct19_manifest.parquet", index=False)
print("Wrote", len(manifest), "rows")

#%%
# >>> checking parquet file
pq = pd.read_parquet("cq500ct_qct19_manifest.parquet")
pq.head()
