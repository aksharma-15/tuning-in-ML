# tuning-in-ML
Fundamentals of tuning hyperparameters in machine learning models using Optuna, XGBoost and scikit-learn early stopping.

# Robust Cross-Validation & Hyperparameter Optimization Framework

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-11557c?style=flat-square&logo=matplotlib&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-2.1.3-013243?style=flat-square&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-2.2.3-150458?style=flat-square&logo=pandas&logoColor=white) ![Optuna](https://img.shields.io/badge/optuna-4.9.0-2E2A69?style=flat-square) ![XGBoost](https://img.shields.io/badge/xgboost-3.4.1-1B8C1E?style=flat-square) ![SciPy](https://img.shields.io/badge/scipy-1.16.3-8CAAEE?style=flat-square&logo=scipy&logoColor=white)

As a machine learning engineer, I've seen too many models achieve stellar accuracy in notebooks only to collapse upon deployment. The root cause is almost universally flawed validation strategies and data leakage. A 2023 review in *Patterns* journal by Kapoor and Narayanan analyzed ML papers across 17 scientific disciplines, uncovering 329 studies compromised by data leakage that inflated reported accuracies by 29-62%. This issue was starkly visible during the COVID-19 pandemic, where hundreds of AI diagnostic tools for CT scans failed clinical utility because models learned to recognize specific patients across mixed training and test sets, rather than generalized disease markers. 

This repository provides a production-grade blueprint for designing unbiased validation pipelines, selecting appropriate cross-validation strategies, and executing intelligent hyperparameter searches.

## Core Engineering Objectives

*   **Strategic Cross-Validation:** Learn to match the cross-validation strategy to the specific data scenario, utilizing `KFold`, `StratifiedKFold`, `GroupKFold`, or `TimeSeriesSplit`.
*   **Leakage Prevention:** Understand why `GroupKFold` (preventing familiarity bias) and `TimeSeriesSplit` (respecting temporal causality) often yield lower, but vastly more honest, R-squared estimates compared to standard random splits.
*   **Intelligent Optimization:** Replace brute-force grid searches with Optuna studies, leveraging custom objective functions and Tree-structured Parzen Estimator (TPE) samplers for efficient hyperparameter discovery. For more on Optuna [click here](https://optuna.org/#getting-started)
*   **Compute Efficiency:** Implement early stopping in gradient boosting models to terminate poor-performing configurations early, saving valuable compute resources.
*   **Visual Diagnostics:** Interpret Optuna visualization plots and map the expanding training windows unique to `TimeSeriesSplit`.

## Data Architecture

To concretely demonstrate how naive validation fails, this codebase utilizes an extended synthetic real estate dataset consisting of 200 records. The data is specifically engineered with complex dependencies to simulate real-world modeling challenges:
*   **Spatial Groupings:** A `neighborhood` feature categorizes properties by location, enabling robust demonstrations of `GroupKFold` behavior.
*   **Temporal Dynamics:** A `sale_month` attribute tracks chronological ordering over a 36-month span, serving as the foundation for `TimeSeriesSplit` testing.
*   **Engineered Complexity:** The target `price` variable mathematically integrates a neighborhood-based location multiplier and an upward time trend, penalizing models that fail to isolate these factors during validation.

## Dependencies

This pipeline requires a strict environment to ensure reproducibility. Ensure the following library versions are installed:
*   `scikit-learn==1.6.1`
*   `matplotlib==3.10.0`
*   `numpy==2.1.3`
*   `pandas==2.2.3`
*   `optuna==4.9.0`
*   `xgboost==3.4.1`
*   `scipy==1.16.3`

or directly install packages from the provided requirements file: ```pip install -r requirements.txt```


---

## 🤝 Contributing

Contributions to improve the examples, add new functions or methods, or fix typos are always welcome. Please feel free to open an issue or submit a pull request!

---

## Connect with me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abhay-kumar-sharma-a22a94171)

