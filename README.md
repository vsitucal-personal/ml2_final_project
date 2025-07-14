# Machine Learning 2 Final Project

![](images/title.png)

- [View Notebook](ml_final_project_final-regularization.ipynb) 

## Intoduction

This project aims to use ML models as an alternative to pricing options aside from the more popular and traditionally used Black-Scholes-Merton equation. Finance data was obtained from Yahoo! Finance via webscraping and Linear and Tree Models were used as prospective models for Machine Learning.

![](images/bsm_equation.png)

## Model Results

### Without regularization
Train-Test Model performance metrics:
![](images/wo_reg_summary1.png)
Train-Test Model performance metrics % difference:
![](images/wo_reg_summary2.png)

### With regularization
Train-Test Model performance metrics:
![](images/w_reg_summary1.png)
 Train-Test Model performance metrics % difference:
![](images/w_reg_summary2.png)

## Comparison vs. Black-Scholes-Merton
Baselines vs. Best Model performance metrics regularized
![img.png](images/bsm_compa.png)

## Top feature predictors in terms of mean SHAP value

![](images/shap1.png)

## Feature contribution to option price for a certain instance per model

![](images/shap2.png)
