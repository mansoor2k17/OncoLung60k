OncoLung60K: Pushing the Boundaries of Lung Cancer Diagnosis with AI Driven Fusion Model and Large Scale Dataset

Dataset Link: Mansoor, A. (2025). 
OncoLung60K (Version 1) [Data set]. Zenodo.
https://zenodo.org/records/14995223

  ABSTRACT
  Lung cancer ranks as the primary contributor to cancer-related fatalities. The standard diagnostic method for lung cancer involves pathologists evaluating histopathological images. However, the limited availability of qualified pathologists and benchmark dataset presents a challenge in addressing significant clinical demands. First, the paper aims to create an enhanced dataset named as OncoLung60K of Lung Cancer only having four major types i.e Adenocarcinoma, Squamous Cell Carcinoma, Small Cell Lung Cancer and normal tissues that have 15000 images per class while incorporating all complex images of different scanners at 20x magnification level. The preparation of the data set involved several stages, i.e data slide preparation, H&E staining, tissue subtype classification by two pathologists and the creation of a balanced data set with 60,000 images of 230x patients with a resolution of 512x512. Secondly, we have used artificial Intelligence (AI) i.e comparison and testing of this dataset with state of the art Deep Learning models including new proposed Fusion model of ResNet152, DenseNet121, EfficientNetB7 and Transformer. For the base models ResNet152, DenseNet121, EfficientNetB7 and our own proposed fusion model, default weights of KemiaNet instead of ImageNet were used and obtained an accuracy, Precision, recall, F-1 score and Jaccard loss of 0.994, 0.982, 0.977, 0.980 and 0.960. Fusion model also outperformed on famous datasets like LC25000 & HistoLung 700 dataset with 2% improvement of accuracy in case of Lung25000 and 7% in case of HistoLung700 datasets. This study has large clinical importance specially for third world countries who can not afford costly Whole slide image scanners. The data set and code are made publicly available at https://github.com/mansoor2k17/OncoLung60k.git

![image](https://github.com/user-attachments/assets/4f495397-cab0-4a06-8d36-d2c7920b43d0)

Pre-requisites:

NVIDIA GPU (NVIDIA RTX 6000)
Python (3.7.0), OpenCV (3.4.0), Openslide-python (1.1.1) and Pytorch (1.5.0) 
