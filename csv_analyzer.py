#%%
# >>> imports
from os import read
import glob
import pandas as pd

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

#%%
#
compact_reads.head()

#%%
# loading data with glob module (file and folders are inconsistent with dataset)
# for p in glob.glob('data/**', recursive=True):
# 	print(p)
