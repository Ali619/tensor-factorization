# Main Approaches and Their Accuracy

In this document, I will explain how the system functions and evaluate its accuracy.

## Original system, Consist of 3 models

his approach involves creating three models (Linear Regression, Naive Bayes, and XGBoost) for each customer to make predictions. The accuracy of these models is presented in the table below, evaluated using the **F1** metric for `XGBoost` and `Naive Bayes`, and the **Mean Squared Error (MSE)** for `Linear Regression`:

Linear Regression | Naive Bayes | XGBoost
---                 | ---       | --- 
38.124              | 0.191     |	0.470

You can see the results are far from minimum and is not acceptable for bith classifier and regression models.
We tried `RandomForesstClassifier` model too but the accuracy still remains below 50% in overall:

Linear Regression | Naive Bayes | XGBoost | Random Foresst
---                 | ---       | ---      | ---
38.124              | 0.191     |	0.470  | 0.444

In our view, this approach is not worthing to spend time on it

## History-Search approach

This approach will check each day of week and day of month of previuse bought and returns true if at the sam day of week or month a buyer bought an item. The main idea behind this approach is just this simple, and here is the **F1** accuracy result:

History Search |
--- |
0.206

You can see even this simple method, has better accuracy than `NB` and `LR` models.

<div class="page"/>

## Recommendation approach

In this approach, we aim to recommend a number of items to users based on their previous purchasing history using **tensor factorization**. The number of recommended items is denoted by `K`, which represents how many items we want to recommend. Our experiments are based on `K=1` to ensure fair evaluation with other methods.

The accuracy of this method is shown below, measured using the **F1** score:

| Recommendation System |
| ---                   |
| 0.0019                |

> **NOTE:** This number is calculated by comparing **flattened** train data and test data.

Also, the average of F1 for each combination of `(user, time)` is: **0.07655**.

### Here is the all results since the begining:

Linear Regression (MSE)| Naive Bayes | XGBoost  | Random Foresst | History Search | Recommendation System | Recommendation System (for each user) |
---                    | ---         | ---      | ---            | ---            | ---                   | ---                                   |
38.124                 | 0.191       |	0.470   | 0.444          | 0.206          | 0.0019                | 0.07655                               |
