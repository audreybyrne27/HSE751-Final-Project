# Predicting 30-Day Hospital Readmission Among Patients With Diabetes

**HSE 751: Programming for Health Data Science**
Dartmouth Geisel School of Medicine · Fall 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/audreybyrne03/HSE751-Final-Project/blob/main/Diabetes_Readmission_Pipeline.ipynb)

## Overview

Thirty-day readmission is a widely used measure of hospital care quality and health care utilization. This project built a reproducible analysis pipeline, in a single Google Colab notebook, that predicts 30-day readmission from the UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset.

The notebook covers the full workflow:

- data loading, validation, and profiling
- cohort definition, preprocessing, and feature engineering
- descriptive statistics and exploratory plots
- inferential statistical testing
- five supervised classification models, evaluated on held-out patients

The pipeline is configurable through a settings cell. It adapts several analytical choices to the data and to the user's stated goal, and it records each automatic decision in a decision log.

## Research Objectives

**Primary objective:** Develop and evaluate supervised classification models that predict whether a hospital encounter for a patient with diabetes is followed by readmission within 30 days, using F1 score as the predefined primary metric.

**Secondary objectives:**

1. Test whether prior inpatient utilization is associated with 30-day readmission, and screen all predictors for association with the outcome, reporting effect sizes and correcting for multiple comparisons.
2. Prevent information leakage between training and evaluation data, including leakage through patients who have repeat encounters.
3. Build the analysis as a reproducible, modular pipeline that validates its input, adapts to dataset characteristics and user-defined goals, and documents its decisions.

## Dataset

| | |
|---|---|
| **Source** | UCI Machine Learning Repository, dataset ID 296: [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **File** | `diabetic_data.csv`, the original file distributed by UCI and stored in this repository |
| **Size** | 101,766 encounters from 71,518 unique patients |
| **Columns** | 50: 47 candidate predictors, the outcome (`readmitted`), and two identifiers (`encounter_id`, `patient_nbr`) |
| **License** | CC BY 4.0 |

**Outcome definition:** The original `readmitted` variable has three levels: `<30`, `>30`, and `NO`. It was recoded as a binary outcome, `readmitted_30d`:

- **1:** readmitted within 30 days (`<30`).
- **0:** otherwise (`>30` or `NO`).

In the full dataset, 11,357 encounters (11.16%) were positive.

**Missing-value coding:**

- In the raw file, `?` denotes a missing value.
- In `max_glu_serum` and `A1Cresult`, the value `None` means the test was not performed. It was kept as its own category. Serum glucose was not measured in 94.75% of encounters, and HbA1c was not measured in 83.28%.

**Cohort exclusions:**

| Step | Encounters removed | Encounters remaining | Unique patients | 30-day readmission (%) |
|---|---|---|---|---|
| Starting dataset | — | 101,766 | 71,518 | 11.16 |
| Discharge to hospice or death (disposition codes 11, 13, 14, 19, 20, 21) | 2,423 | 99,343 | 69,990 | 11.39 |
| Invalid gender value (`Unknown/Invalid`) | 3 | 99,340 | 69,987 | 11.39 |

**Analytic cohort:** 99,340 encounters from 69,987 patients, with 11,314 positive encounters (11.39%).

## Analytical Pipeline

The notebook runs these steps in order. Section numbers refer to the notebook.

1. **User settings** (top of the notebook). The settings cover the data source, the analysis goal (`primary_metric`), the target recall for screening, the class-imbalance cutoff, α, and the random seed. They are validated before anything else runs.
2. **Data import, validation, and profiling** (Section 2):
   - The data are loaded from GitHub, a file upload, or a local file.
   - They are checked against a `DATA_CONTRACT`: required columns, outcome values, allowed category levels, numeric types and plausible ranges, and duplicate encounter IDs.
   - A column-by-column profile is generated (type, missingness, distinct values, most common value, numeric range).
   - A confirmation message is printed when loading, validation, and profiling succeed.
3. **Exploratory analysis of the full dataset** (Sections 3–8). This covers dataset structure, the target definition, descriptive statistics (mean, SD, median, IQR, and frequency tables), readmission rate by age group and by prior inpatient admissions, and a data-quality assessment.
4. **Cohort selection and preprocessing** (Section 11):
   - The cohort rules above were applied.
   - The binary outcome was created, and the original three-level outcome was dropped.
   - Thirteen low-information predictors were removed by rule: `weight` (96.85% missing) and 12 medication variables with a single value in at least 99.9% of encounters.
   - Remaining missing categorical values were coded as an explicit `Missing` category.
5. **Clinical feature engineering** (Section 11.5):
   - ICD-9 codes in `diag_1`–`diag_3` were grouped into clinical categories, following Strack et al. (2014).
   - Age bands were converted to midpoints (`age_years`).
   - A total prior visit count was added (`total_prior_visits`), along with the number of diabetes medications with a dose change (`n_medication_changes`).
   - Medical specialty was limited to its 10 most common values plus `Other` and `Missing`.
   - Admission and discharge codes were treated as categories.
   - The modeling dataset had **36 predictors (11 numeric, 25 categorical)**.
6. **Inferential statistics** (Section 12): the methods are described below.
7. **Supervised classification** (Section 13):
   - Data were split by patient into training, validation, and test sets.
   - A preprocessing pipeline was fitted on the training data only.
   - Two models were tuned on the validation set.
   - Five models were trained, with a decision threshold chosen for each on the validation set.
   - The best model was selected on the validation set.
   - All models were evaluated on the held-out test set, with bootstrap confidence intervals, an overfitting check, ROC and precision–recall curves, and confusion matrices.
   - Permutation feature importance was calculated for the selected model.
8. **Interpretation** (Section 14), **decision log** (Section 15), and **enhancement summary** (Section 16).

## Inferential Statistical Methods

**Primary hypothesis:** A chi-square test of independence compared 30-day readmission across prior inpatient admission groups (0, 1, 2, 3+) in the analytic cohort. Cramér's V was reported as the effect size, and the expected-count assumption (all expected counts ≥ 5) was checked.

**Sensitivity analysis for repeat patients:** The primary test assumes independent observations, but some patients have more than one encounter. The test was therefore repeated using only each patient's first encounter (lowest `encounter_id`).

**Screening of all predictors:** Each of the 36 predictors was tested against the outcome. The test was selected automatically:

| Predictor type | Rule | Test | Effect size |
|---|---|---|---|
| Numeric | \|skewness\| ≤ 1 | Welch's t-test | Cohen's d |
| Numeric | \|skewness\| > 1 | Mann–Whitney U | Rank-biserial r |
| Categorical | Default | Chi-square test of independence | Cramér's V |
| Categorical | 2×2 table with any expected count < 5 | Fisher's exact test | Cramér's V |

In the executed run, these rules selected:

- **Mann–Whitney U:** 8 numeric predictors.
- **Welch's t-test:** 3 numeric predictors (`num_lab_procedures`, `age_years`, `number_diagnoses`).
- **Chi-square:** all 25 categorical predictors.

The Fisher's exact branch is implemented, but no predictor met its condition, so it was not used.

Effect sizes were also put on a common correlation scale so all predictors could be ranked together. Cohen's d was converted with $r = d/\sqrt{d^2+4}$. P-values were adjusted for 36 comparisons with the **Holm** method (α = 0.05). The notebook presents the formulas for the chi-square statistic, Cramér's V, the Mann–Whitney U statistic and rank-biserial correlation, Welch's t and Cohen's d, and the Holm adjustment in LaTeX, and explains how each relates to the analysis.

## Machine Learning Methods

**Models:**

| Model | Configuration |
|---|---|
| Logistic Regression | L2 penalty (C = 1.0), class weights |
| Decision Tree | Tuned: max depth 8, minimum 50 samples per leaf; class weights |
| Random Forest | 300 trees, max depth 12, minimum 20 samples per leaf; class weights |
| XGBoost | Tuned: max depth 5, `min_child_weight` 10, 600 trees; learning rate 0.05; `scale_pos_weight` = 7.80 |
| Neural Network (MLP) | Two hidden layers (64 and 32 units), ReLU activation, Adam optimizer, L2 penalty, early stopping |

**Patient-level split:** The cohort was split 70/15/15 with `GroupShuffleSplit` on `patient_nbr`, so all encounters from a patient fall in one subset. The code checks that no patient appears in more than one subset.

| Subset | Encounters | Patients | 30-day readmission (%) |
|---|---|---|---|
| Training | 69,613 | 48,990 | 11.37 |
| Validation | 14,868 | 10,498 | 11.56 |
| Test | 14,859 | 10,499 | 11.31 |

**Leakage prevention:**

- **Encoding and scaling** were fitted on the training set only, inside a scikit-learn `Pipeline`, and then applied unchanged to the validation and test sets. Numeric predictors received a Yeo-Johnson transform and standardization. Categorical predictors were one-hot encoded, with categories seen in fewer than 20 training encounters grouped together, producing 166 model inputs.
- **The original three-level outcome** was dropped before modeling.
- **Earlier rule-based steps** (cohort rules, low-information predictor removal, and the top-10 specialty grouping) were applied to the full cohort before splitting. None of these steps used the outcome.

**Class imbalance:** Class weights are switched on when the positive class is less than 35% of the training data (setting `imbalance_threshold`). The positive class was 11.4% of training encounters, so weights were applied, with a positive-class weight of 7.80. scikit-learn's MLP does not support class weights and relied on threshold tuning instead.

**Hyperparameter tuning:** The decision tree (8 settings) and XGBoost (12 settings) were tuned on a small grid. Each setting was trained on the training set and scored by validation PR-AUC. The other three models used fixed settings.

**Decision thresholds:** With `primary_metric = "f1"`, each model's threshold was set to the value that maximized F1 on the validation set. Setting `primary_metric = "recall"` instead selects the highest threshold that achieves a target recall (default 0.80). This option is implemented but was not used in the executed run.

**Primary metric and model selection:** F1 was defined as the primary metric during project planning. `select_best_model()` ranks models using **validation-set** F1 only, with ties broken by validation PR-AUC. Test results did not enter the selection rule; they were computed and displayed in earlier cells, but were not used for any modeling decision.

**Held-out evaluation:** Each model was evaluated once on the test set at its validation-chosen threshold. The reported metrics are F1, precision, recall, specificity, accuracy, AUROC, and PR-AUC. 95% confidence intervals for test F1 and AUROC came from a patient-level bootstrap (500 resamples of test patients).

## Results

### Inferential statistics

- **Primary hypothesis:** 30-day readmission rose with the number of prior inpatient admissions: **8.59%** (0), **13.25%** (1), **17.93%** (2), and **26.40%** (3+). The test was χ²(3) = 2411.04, p < 0.001, with Cramér's V = 0.156 (a small effect). The smallest expected cell count was 776.
- **Sensitivity analysis** (first encounter per patient, n = 69,987): the gradient was similar (8.13%, 12.88%, 18.52%, 26.45%), with χ²(3) = 664.62, p < 0.001, and V = 0.097.
- **All-predictor screening:** after Holm correction, 25 of 36 predictors were statistically significant. However, only three reached a small effect size, and all others were negligible:
  - prior inpatient admissions (rank-biserial r = 0.215)
  - total prior visits (r = 0.208)
  - discharge disposition (Cramér's V = 0.115)
- **Statistical vs. clinical significance:** With roughly 99,000 encounters, statistical significance largely reflected sample size, not clinically meaningful differences. For example, readmitted encounters had a mean of 7.69 vs. 7.36 diagnoses (Cohen's d = 0.17).
- **Approximate p-values:** nine categorical predictors had expected counts below 5 in some cells, so their chi-square p-values are approximate.

### Model performance

**Validation-set ranking used for selection (F1):**

| Rank | Model | Validation F1 |
|---|---|---|
| 1 | **XGBoost** | **0.289** |
| 2 | Random Forest | 0.284 |
| 3 | Logistic Regression | 0.280 |
| 4 | Neural Network (MLP) | 0.276 |
| 5 | Decision Tree | 0.270 |

**Held-out test set (n = 14,859 encounters):**

| Model | Test F1 (95% CI) | Test AUROC (95% CI) | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.290 (0.268–0.312) | 0.681 (0.664–0.697) | 0.219 | 0.427 | 0.243 |
| Random Forest | 0.287 (0.263–0.313) | 0.679 (0.661–0.695) | 0.238 | 0.363 | 0.218 |
| Decision Tree | 0.278 (0.256–0.301) | 0.653 (0.637–0.671) | 0.206 | 0.424 | 0.206 |
| Logistic Regression | 0.276 (0.259–0.293) | 0.665 (0.649–0.679) | 0.183 | 0.558 | 0.194 |
| Neural Network (MLP) | 0.272 (0.255–0.289) | 0.663 (0.648–0.678) | 0.188 | 0.495 | 0.210 |
| *Reference: predict all readmitted* | *0.203* | *0.500* | *0.113* | *1.000* | *0.113* |

**Selected model:**

- **Why XGBoost:** it was selected because it had the highest validation F1 (0.289). Its test F1 was 0.290.
- **Recall and precision:** at its validation-chosen threshold (0.547), it identified 42.7% of test-set readmissions, with a precision of 21.9%. That precision is about twice the test-set base rate of 11.3%.
- **Comparison with the runner-up:** the random forest's test F1 (0.287) lies within XGBoost's 95% CI, and the F1 intervals overlap for all five models. XGBoost should therefore not be described as clearly superior.
- **Ranking differences:** the decision tree ranked last on validation F1 but third on test F1.

**Other observations:**

- **Threshold tuning:** the MLP's test F1 was 0.014 at the default 0.5 threshold and 0.272 at its validation-chosen threshold (0.143). For the other models, tuning changed test F1 by 0.012 or less.
- **Overfitting:** the training-to-test AUROC gap was largest for XGBoost (0.124) and the random forest (0.068) and smallest for logistic regression (0.001). Validation and test AUROC were within 0.01 of each other for every model.

### Most informative predictors

Permutation importance for XGBoost was computed on an 8,000-encounter sample of the validation set, as the mean drop in PR-AUC over 5 shuffles. The top five predictors were:

1. discharge disposition (0.067)
2. prior inpatient admissions (0.041)
3. primary diagnosis group (0.012)
4. payer (0.010)
5. total prior visits (0.009)

These agree with the inferential screening, where prior utilization and discharge disposition were the only predictors with more than negligible effects.

### Interpretation

**Discrimination was modest** (test AUROC 0.65–0.68). At the selected operating point, the best model missed more than half of readmissions, and most encounters it flagged were not readmitted.

**Possible use:** the results support using such a model, at most, to help prioritize transitional-care resources across a population. It is not suitable for individual clinical decisions.

**Practical message:** the findings point to recent inpatient use and discharge destination as the most informative signals in this dataset. These are associations; they do not show causal effects.

## Reusable Pipeline and Conditional Logic

**Scope of compatibility:** The pipeline is built for datasets with the **same structure as the UCI Diabetes 130-US Hospitals file**. This means all 47 predictor columns and the `readmitted` outcome (`<30`, `>30`, `NO`), with the same column names and category coding. It is **not** a general tool for arbitrary clinical datasets.

**Data input:** `USER_SETTINGS["data_source"]` selects where the data come from:

- **`"github"`** (default): loads the project file from this repository.
- **`"upload"`**: opens a file picker in Google Colab (`google.colab.files.upload`). It accepts one CSV or Excel file.
- **`"local"`**: reads `diabetic_data.csv` from the notebook's working folder.

**Validation (`DATA_CONTRACT`):**

- **Critical problems stop the run** with a specific message. These include missing required columns, unexpected outcome values, a missing or single-valued outcome, non-numeric count variables, unexpected category levels in key variables, and duplicate `encounter_id` values.
- **Non-critical issues are reported as warnings.** These include out-of-range numeric values, extra columns, absent identifiers, repeat patients, and fewer than 1,000 rows.
- **Invalid settings** are rejected by `validate_settings()`.

**Automated profiling:** `profile_data()` summarizes every column of whatever dataset is loaded.

**Conditional logic:** The table below lists where the pipeline adapts. The last column shows what happened in the executed run.

| Step | Rule | Executed run |
|---|---|---|
| Identifiers | If `encounter_id` is absent, one is created. If `patient_nbr` is absent, splitting and bootstrapping are done by encounter, and the repeat-patient sensitivity analysis is skipped. | Both present; split by patient; sensitivity analysis run |
| Extra columns | Dropped with a warning | None present |
| Numeric tests | Welch's t-test if \|skewness\| ≤ 1, otherwise Mann–Whitney U | 3 Welch, 8 Mann–Whitney |
| Categorical tests | Fisher's exact test for sparse 2×2 tables, otherwise chi-square | 25 chi-square, 0 Fisher |
| Class imbalance | Class weights on if positive class < `imbalance_threshold` | On (11.4% < 35%) |
| Thresholds | Maximize F1, or meet `target_recall` when `primary_metric = "recall"` | F1-maximizing |
| Model selection | Highest validation value of the metric matching `primary_metric` (F1, precision at target recall, PR-AUC, or AUROC) | XGBoost by validation F1 |

**Decision log:** `log_decision()` records each automatic decision and its reason. Section 15 displays the full log. In the executed run, the log contains eight entries.

The branches for missing identifiers, extra columns, Fisher's exact test, the recall goal, and switching class weights off are implemented in code but were **not triggered** in the executed run, because it used the complete project dataset with default settings. The written interpretations in the notebook describe the project dataset. Tables, plots, and the decision log are regenerated for any dataset that passes validation.

## Repository Structure

```
HSE751-Final-Project/
├── Diabetes_Readmission_Pipeline.ipynb   # Final executed notebook
├── diabetic_data.csv                     # Project dataset (UCI ID 296)
├── requirements.txt                      # Python dependencies
└── README.md
```

## Reproducibility Instructions

### Option A: Google Colab with the project dataset

1. Click the **Open in Colab** badge at the top of this page.
2. Leave `"data_source": "github"` in the **User Settings** cell. The notebook reads `diabetic_data.csv` directly from `https://raw.githubusercontent.com/audreybyrne03/HSE751-Final-Project/main/diabetic_data.csv`, so no upload is needed.
3. Select **Runtime → Run all**.
4. Check that Section 2 prints `Data loaded successfully`, a validation report ending in `PASSED`, and `Data upload, validation, and profiling completed successfully.`

The executed notebook ran without installing any packages, using Colab's preinstalled libraries.

### Option B: Uploading a compatible dataset

1. Open the notebook in Colab.
2. In **User Settings**, set `"data_source": "upload"`. You can optionally change `primary_metric` to `"f1"`, `"recall"`, `"pr_auc"`, or `"roc_auc"`.
3. Select **Runtime → Run all**. Execution pauses in Section 2 until you choose a file in the upload widget. Upload exactly one CSV, `.xlsx`, or `.xls` file.
4. The file must have:
   - All 47 predictor columns listed in `PREDICTOR_COLUMNS` (in the `DATA_CONTRACT` cell).
   - The outcome column `readmitted`, coded `<30`, `>30`, or `NO`.
   - Missing values coded as `?` or blank.
   - `encounter_id` and `patient_nbr` are optional.
5. If the file fails validation, the notebook prints a report listing each problem and stops with a `ValueError`. If it passes, all later sections run on the uploaded data.

### Option C: Running locally

```bash
git clone https://github.com/audreybyrne03/HSE751-Final-Project.git
cd HSE751-Final-Project
pip install -r requirements.txt
jupyter notebook Diabetes_Readmission_Pipeline.ipynb
```

Run all cells. The default setting (`"github"`) needs an internet connection. To use the cloned copy of the data instead, set `"data_source": "local"`. The `"upload"` option works only in Google Colab.

**Reproducibility notes:**

- **Fixed seed:** all random steps use `random_state = 42`, so results should be identical when the notebook is rerun in the same software environment.
- **Different library versions:** small numerical differences can occur with other versions of scikit-learn or XGBoost, so exact values from the executed notebook may not be reproduced elsewhere.

## Dependencies

The notebook requires Python 3 and the following packages, which are listed in `requirements.txt`:

- **Data handling:** pandas, NumPy.
- **Plots:** Matplotlib, seaborn.
- **Statistical tests and Holm correction:** SciPy, statsmodels.
- **Preprocessing, models, metrics, and permutation importance:** scikit-learn.
- **XGBoost model:** xgboost.
- **Excel uploads only:** openpyxl and xlrd.
- **Running locally:** Jupyter.

The executed notebook was run in Google Colab with Python 3.13.

## Limitations

- **Historical data:** the encounters are from 1999–2008 and use ICD-9 coding. Treatment patterns and coding have changed since.
- **Incomplete capture of readmissions:** readmissions to hospitals outside the 130 participating hospitals may not be recorded, so some positive cases may be labeled negative.
- **Missing clinical and social variables:** the data lack social support, functional status, detailed laboratory values, and post-discharge care, all of which plausibly affect readmission.
- **Administrative data:** diagnoses and utilization come from administrative records. These are subject to coding variation and do not capture disease severity well.
- **Repeat encounters:** the patient-level split keeps patients from appearing in more than one subset. However, repeat encounters *within* the training set were treated as independent, and only the primary hypothesis test had a one-encounter-per-patient sensitivity analysis.
- **Limited generalizability and no external validation:** the models were evaluated on held-out patients from the same dataset only. Performance in other hospitals or time periods is unknown.
- **Modest predictive performance:** test AUROC was 0.65–0.68, and F1 was at most 0.29. Model calibration was not assessed.
- **Tuning limits:** tuning used small grids. For XGBoost, the selected values (max depth 5, 600 trees) were at the upper edge of the grid, so better settings may exist outside the range searched.
- **Threshold choice:** thresholds maximized F1, which weights precision and recall equally. A clinical application might prefer a different trade-off.

## References

1. Clore J, Cios K, DeShazo J, Strack B. *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository; 2014. https://doi.org/10.24432/C5230J
2. Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International.* 2014;2014:781670. https://doi.org/10.1155/2014/781670
