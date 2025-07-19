# Design Document:

## TO-DO List:
() analyze one dicom file to get the hang of the whole ds.

## Big picture:
Problem:
* Binary scan-level classification (does the CT slice show intracranial hemorrhage - ICH?)
* Useful in research, radiology & emergency triage
* Current best model:
	* 2-stage image-window CNN + sequence model ~ 0.98 AUROC on CQ500
	* 3D CNNs for volumetric data (e.g., 3D ResNet)
	* 2D CNNs ensambles for slices (e.g., EfficientNet or DenseNet)

Metrics:
* ROC-AUC (AUROC) is used to measure the prediction accuracy
* Sensitivity and Specificity measures for clinical relevance
* F1 score

Frame:
* Supervised
* Train on 2D slices (resource-friendly)
* Patient-level for final diagnosis
* Imbalanced class (~ 5% positive cases)

Output:
* Probability of ICH (per slice, per patient)

## Data analysis:
Qure.ai CQ500 dataset
* Directory: "qct01/CQ500CT### CQ500CT ###"
* Labels:
	* Patient ID matching
* DICOM files:
	* Slice count:	variable (30+)
	* Thickness:	variable (Thin - 5mm)
	* Resolution:	???
	* Metadata
* Class balance
* Image windows
	* Brain, Bone, Subdural windows (stack like RGB)

read.csv:
* Each row corresponds to one patient (study) = one Folder per patient with DICOM
* Normal, ICH (IPH, IVH, SAH, SDH, EDH, Other) per radiologist
* Only 1 Rad agrees = 41 scans -> we can check with a radiologist
* Only 2 Rads agree = 37 scans
* All  3 Rads agree = 168 scans
* 491 patients
* B1 is real-world appearance of ICH in a 30 days period	=~ 214 (050 - 23%)
* B2 is enriched in ICH (double checked)					=~ 277 (196 - 70%)
* No missing readings or values
* "name" is the identifier for each patient file
* Extracted ICH information "compact_reads.csv" file for easier traininig

cq500ct_manifest.parquet:
* Contains *name, *series_uid, *instance_num, path, thickness (mm), series description
* Fast loading for future references

Testing one file before moving to preprocessing:
* data_loader.py (loads data from the dataset folder via parquet + csv)

Kaggle GPU free plan

## Preparation & Preprocessing:

## Models available:

## Fine-tuning:

## Others:
