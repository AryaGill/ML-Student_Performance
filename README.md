# Predicting Student Performance:

## Improving Accuracy with XGBoost

## Table of Contents

- [Predicting Student Performance:](#predicting-student-performance)
  - [Improving Accuracy with XGBoost](#improving-accuracy-with-xgboost)
  - [Table of Contents](#table-of-contents)
    - [Overview:](#overview)
    - [Problem Statement](#problem-statement)
    - [Original Approach](#original-approach)
    - [My Approach: XGBoost Implementation](#my-approach-xgboost-implementation)
      - [Hyperparamter Optimization with Hyperopt](#hyperparamter-optimization-with-hyperopt)
      - [Improvements Achieved](#improvements-achieved)
    - [Dataset](#dataset)
    - [Preprocessing Steps](#preprocessing-steps)
    - [Conclusion](#conclusion)
    - [References](#references)

### Overview:

This project aims to recreate and improve the performance of best model from a [research paper](https://www.semanticscholar.org/paper/61d468d5254730bbecf822c6b60d7d6595d9889c) [1] on predicting student performance. The original study utilized several machine learning techniques such as Decision Trees, Random Forests, Neural Networks, and Support Vector Machines for Binary Classifications, Five-level Classifications, and Regression. In this implementation, I focused on the Regression model utilizing the Portuguese dataset and enhanced it with the use of XGBoost, achieving better accuracy with a lower RMSE score.

### Problem Statement

The goal of this original study was to predict students' academic performance in two classes, mathematics and portuguese, based on a dataset of various factors, including socio-economic, academic, and demographic features. The key challenges were to accurately predict:

- Final Grade (G3)

### Original Approach

The research paper explored various machine learning techniques:

- Decision Trees
- Random Forest
- Neural Networks
- Support Vector Machines

Each method was then applied to Binary Classification, Five-level Classifications, and Regression.

The **Random Forest** model was reported to achieve the highest performance with an RMSE of 1.32 in the Portuguese Dataset with Regression. While, Random Forests are powerful, the are sometimes outperformed by gradient boosting methods.

### My Approach: XGBoost Implementation

To improve this model's predictive capabilities, I implemented teh XGBoost algorithm. XGBoost stands for "Extreme Gradient Boosting," which is a "scalable, distributed gradient-boosted decision tree (GBDT) machine learning library" [2]. This often delivers superior performance on data due since it develops one tree at a time, and corrects faults created by previous trees. This vastly outperforms simple Random Forest algorithms as they creates trees independently of each another.

#### Hyperparamter Optimization with Hyperopt

In order to fine-tune the performance of the XGBRegressor model, I employed Bayesian optimization using the `Hyperopt` library. Bayesian Optimization is an effective method for hyperparmater training, as it uses past evaluations to estimate the optimal configuration more effectively than traditional grid or random grid search.

Using Hyperopt, I optimized key hyperparameters such as:

- n_estimators
- learning_rate
- max_depth
- colsample_bytree

#### Improvements Achieved

By using XGBoost, I drastically improved the RMSE by **69%** as demonstrated below:

- XGBoost (My Approach): 0.78
- Random Forest (Original Approach): 1.32

These improvements highlight the advantage of using gradient boosting over basic methods such as Random Forests.

### [Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

The dataset [3] used in this project is the same as in the original paper. Although the original paper contains both Mathematics and Portuguese dataset, only the Portuguese dataset is utilized in my application.

### Preprocessing Steps

- The data provided is already cleaned with no missing values or wrong type.
- Data is one-hot encoded using `pd.get_dummies()` method.
- To keep the dataset the same as the original paper, all dataset features were utilized for training the model.
- Cross validation set of 0.1 is created to test the model.

### Conclusion

This project demonstrates the effectiveness of XGBoost in improving the accuracy and reducing the error of student performance prediction models. While the original Random Forest performed well, XGBoost's advanced boosting techniques combined with Hyperparameter Optimization helped push the model performance even further.

### References

[1] "Using data mining to predict secondary school student performance." Available: https://www.semanticscholar.org/paper/61d468d5254730bbecf822c6b60d7d6595d9889c
[2] NVIDIA, "XGBoost," Available: https://www.nvidia.com/en-us/glossary/xgboost/
[3] UCI Machine Learning Repository, "Student Performance," Available: https://archive.ics.uci.edu/dataset/320/student+performance

