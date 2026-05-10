# Distribution-Aware Neural Network: A Novel Patient Representation Learning Algorithm for Traditional Chinese Medicine Diagnosis Prediction 

## 1. Overview

1. This repository accompanies the DANN paper and provides the supporting data, with an experimental framework to be released for subsequent research.

2. If you use the data or code in this repository, please cite our paper published in EAAI:

@article{ZHAO2026114562,
title = {Distribution-aware neural network: A novel patient representation learning algorithm for Traditional Chinese medicine diagnosis prediction},
journal = {Engineering Applications of Artificial Intelligence},
volume = {175},
pages = {114562},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.114562},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626008432},
author = {Zongyao Zhao and Xin Dong and Xinpeng Song and Chenxi Zhao and Weiyu Li and Zuoyuan Luo and Geyan Pan and Sicen Wang and Yong Tang and Dong Luo and Xuezhong Zhou},
keywords = {Traditional Chinese medicine, Diagnosis prediction, Distribution-aware strategy, Latent structure extraction},


## Data Description

1. The construction details and specific description of the TCM-Chronic dataset can be found in our paper.
2. The `data/TCM-Chronic` directory contains two folders. `OriginalData` contains our constructed raw data, publicly available in xlsx format.
3. `10_Random_Combinations_for_Data_Scale_Experiments`:
   1. Contains the datasets used in our experiments with 5-fold cross-validation and 10 random combinations, saved in pkl format. Each file stores an array containing the pre-split 5-fold cross-validation data.
   2. Please use this dataset to reproduce our experiments.
   3. Naming convention: Taking `X_1_repeat0` as an example, `X` indicates that features are stored, `1` indicates a disease scale of 1 (range: 1–15), and `0` indicates the first random combination (range: 0–9).
4. As a sample, this repository currently includes only the data corresponding to **1骨质疏松 (Osteoporosis)**. The complete dataset is available from the corresponding author upon reasonable request.

## Experimental Framework

A comprehensive experimental framework for TCM intelligent diagnosis is currently in preparation. It is intended to provide unified pipelines and built-in reproduction of several state-of-the-art models, including the method described in our paper. The framework will be released in this repository upon completion.

# 2 Contact Us

Email: zhaozong111@gmail.com

