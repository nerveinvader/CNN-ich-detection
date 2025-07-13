# Design Document:

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
* DICOM files

Kaggle GPU free plan

## Preparation & Preprocessing:

## Models available:

## Fine-tuning:

## Others:
