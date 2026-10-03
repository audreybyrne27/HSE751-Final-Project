# Predicting 30-Day Hospital Readmission Among Patients With Diabetes

## Project Overview

This repository contains my final project for **HSE 751: Programming for Health Data Science**.

The project develops a reproducible supervised machine-learning workflow for predicting **30-day hospital readmission among patients with diabetes** using the UCI **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

The dataset contains more than 100,000 hospital encounters and includes demographic, clinical, medication, diagnostic, and healthcare utilization information. The primary prediction task is to identify hospital encounters associated with an increased risk of readmission within 30 days.

Beyond the specific prediction problem, this project is intended to serve as an initial implementation of a reproducible **clinical outcomes analytics pipeline**. The workflow is being organized so that data validation, preprocessing, exploratory analysis, model training, and model evaluation can eventually be adapted to other compatible patient-level clinical datasets.

## Dataset

**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008  
**Source:** UCI Machine Learning Repository  
**UCI Dataset ID:** 296  
**Source URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset contains:

- **101,766 hospital encounters**
- **47 candidate predictor variables**
- Demographic, clinical, diagnostic, medication, and healthcare utilization information

The dataset is retrieved programmatically from the UCI Machine Learning Repository using the `ucimlrepo` Python package. A separate local copy of the dataset is therefore not required to reproduce the analysis.

## Prediction Problem

The project is structured as a **binary classification problem** predicting whether a hospital encounter is followed by readmission within 30 days.

The original UCI outcome variable, `readmitted`, contains three categories:

- `<30` — readmission within 30 days
- `>30` — readmission more than 30 days after discharge
- `NO` — no recorded readmission

For this project, the outcome is transformed into:

- `1` — readmitted within 30 days
- `0` — not readmitted within 30 days

The resulting binary target is named `readmitted_30d`.

In the full dataset:

- **11,357 encounters (11.16%)** experienced a 30-day readmission
- **90,409 encounters (88.84%)** did not experience a 30-day readmission

## Primary Evaluation Metric

The proposed primary model evaluation metric is **F1 score**.

Because only 11.16% of encounters belong to the positive class, overall accuracy could appear high even for a model that performs poorly at identifying patients who experience a 30-day readmission.

F1 score balances **precision** and **recall**, making it more informative for evaluating performance on the minority positive class. Additional classification metrics may be reported to provide a more complete assessment of model performance.

## Current Project Stage

The current notebook contains the **Dataset Selection and Final Project Planning Lab**, which establishes the foundation for the final machine-learning project.

The current workflow includes:

1. Dataset selection and justification
2. Reproducible data import from UCI
3. Dataset structure and variable assessment
4. Binary target construction and validation
5. Predictor-variable characterization
6. Descriptive statistics
7. Exploratory visualization
8. Missing-data and data-quality assessment
9. Identification of anticipated preprocessing requirements
10. Definition of the prediction problem and evaluation strategy

Future stages of the project will extend this workflow through preprocessing, model development, validation, and evaluation.

## Exploratory Findings

Initial exploratory analysis identified several characteristics relevant to subsequent model development:

- The 30-day readmission rate is **11.16%**, demonstrating substantial class imbalance.
- Prior inpatient utilization shows a strong unadjusted relationship with 30-day readmission.
- Several variables contain substantial missing information.
- Some medication predictors are constant or nearly constant.
- Diagnosis variables contain hundreds of distinct codes and will require careful handling.
- The dataset contains a mixture of numerical, categorical, ordinal, and identifier-coded variables.

These findings will inform the preprocessing and modeling strategy used in later stages of the project.

## Anticipated Preprocessing

Planned preprocessing considerations include:

- Evaluating predictors with extensive missingness
- Handling unknown or invalid categorical values
- Removing constant predictors
- Evaluating near-constant predictors
- Encoding categorical variables appropriately
- Treating integer-coded clinical categories as categorical rather than continuous variables
- Managing high-cardinality diagnosis variables
- Addressing class imbalance during model development and evaluation
- Evaluating potential outcome leakage
- Accounting for repeated patient encounters where possible

Preprocessing decisions will be incorporated into a reproducible analytical pipeline to ensure that transformations are applied consistently during model development and evaluation.

## Google Colab Notebook

The current project notebook can be viewed and executed in Google Colab:

[Open the Dataset Selection and Final Project Planning Lab in Google Colab](https://colab.research.google.com/drive/1NMVC0FFs8SD__aJ-3FGumJc5AjkcDqYX?usp=sharing)

## Repository Contents

```text
diabetes-readmission-prediction/
│
├── README.md
└── Dataset_Selection_and_Final_Project_Planning_Lab.ipynb
```

Additional notebooks or project files may be added as the final project progresses.

## Reproducibility

The analysis is designed to execute in **Google Colab using Python 3**.

The dataset is retrieved directly from UCI using:

```python
from ucimlrepo import fetch_ucirepo

diabetes = fetch_ucirepo(id=296)
```

This avoids dependence on a manually stored local dataset and provides a documented data source for reproduction of the analysis.

To reproduce the current analysis:

1. Open the notebook in Google Colab.
2. Run the notebook from the beginning.
3. The required UCI package is installed within the notebook.
4. The dataset is retrieved directly from the UCI Machine Learning Repository.
5. Execute all cells sequentially to reproduce the descriptive and exploratory analyses.

## Software and Libraries

The project currently uses:

- Python 3
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `ucimlrepo`

Additional machine-learning libraries will be documented as they are incorporated into later stages of the project.

## Project Goal

The immediate goal is to develop and evaluate a reproducible model for predicting 30-day readmission in this dataset.

The broader goal is to use this project to establish a structured clinical outcomes analytics workflow in which dataset-specific definitions are separated, where feasible, from reusable components for data validation, preprocessing, model training, and model evaluation.
