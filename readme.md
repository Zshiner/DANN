# Distribution-Aware Neural Network: A Novel Patient Representation Learning Algorithm for Traditional Chinese Medicine Diagnosis Prediction 

## 1. Overview

1. This repository contains the original data and experiment code for the DANN paper, and provides an experimental framework for subsequent research.

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
abstract = {Traditional Chinese medicine (TCM) diagnosis involves complex and implicit associations between heterogeneous symptoms and diagnostic patterns, and distributional heterogeneity across diseases and patients. Existing intelligent diagnostic models focus primarily on architectural optimization but lack explicit modeling of underlying symptom-diagnosis distributions, resulting in limited robustness, cross-disease generalization, and interpretability. To address these challenges, we propose a distribution-aware neural network (DANN) for diagnostic representation learning. The proposed framework incorporates explicit representations of both global and class-conditional feature distributions and integrates discriminative pruning and latent structure decomposition to capture population-level diagnostic regularities and fine-grained differential variations. In addition, we introduce a cross-disease clinical dataset (TCM-Chronic) covering 15 chronic diseases, 5876 clinical cases, and 97 diagnostic labels to simulate real-world comorbidity scenarios. Experiments on both a public multilabel dataset (TCM-Lung) and a cross-disease dataset demonstrate that the DANN consistently outperforms state-of-the-art machine learning, deep learning, and large language model baselines. With respect to TCM-Lung, the F1-score of the DANN is 0.5112, which is 3.6 percentage points greater than that of the strongest baseline. With respect to TCM-Chronic, the DANN achieves an F1-score of 0.7146, outperforming the random forest by 7.21 percentage points. Ablation and expert evaluations further confirm that distribution-aware modeling contributes to increased diagnostic robustness and better interpretability. These results indicate that explicitly modeling diagnostic feature distributions provides an effective paradigm for intelligent diagnosis, with potential applicability beyond TCM to broader clinical decision-support tasks.}
}

## Data Description

1. The construction details and specific description of the TCM-Chronic dataset can be found in our paper.
2. The `data/TCM-Chronic` directory contains two folders. `OriginalData` contains our constructed raw data, publicly available in xlsx format.
3. `10_Random_Combinations_for_Data_Scale_Experiments`:
   1. Contains the datasets used in our experiments with 5-fold cross-validation and 10 random combinations, saved in pkl format. Each file stores an array containing the pre-split 5-fold cross-validation data.
   2. Please use this dataset to reproduce our experiments.
   3. Naming convention: Taking `X_1_repeat0` as an example, `X` indicates that features are stored, `1` indicates a disease scale of 1 (range: 1–15), and `0` indicates the first random combination (range: 0–9).

## Code Description

1. Although we have released all original code to ensure reproducibility, the code structure is not well-organized due to experiments conducted over a long period of time. Therefore, we recommend using the experimental framework that we will release later.
2. We plan to publicly release a comprehensive TCM intelligent diagnosis experimental framework by June 2026, which will include all pipelines and built-in reproduction of experimental results for several state-of-the-art models, including those in this repository.

# 2 Contact Us

Email: zhaozong111@163.com

