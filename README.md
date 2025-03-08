## WHAT-IF SCENARIO GENERATION TOOL

>The implementation of the TBWISA framework from [Decision intelligence system for generating scenario outcomes using machine learning methods]()

## 1. INTRODUCTION

What-if analysis: data-intensive simulation whose goal is to inspect the behavior of a complex system (i.e., the enterprise business or a part of it) under some given hypotheses (called scenarios) [[Designing what-if analysis: Towards a methodology](https://dl.acm.org/doi/10.1145/1183512.1183523)]

[[Types of scenario planning and their effectiveness: A review of reviews](https://www.sciencedirect.com/science/article/pii/S0016328723000575)]

Time series forecasting

Steps in defining scenarios: identify the key drivers, define assumptions, develop, evaluate the scenarios, select scenarios

Known-in-advance factors, external factors

Scenario analysis applications: forecasting financial performance, payroll planning, investment analysis, inventory planning, employee management, epidemics
challenges + [[Scenario analysis: a review of methods and applications for engineering and environmental systems](https://link.springer.com/article/10.1007/s10669-013-9437-6)]

The main contribution of this work is defining a decision intelligence framework based on transformers model for generating what-if scenario outcomes in e-commerce applications.
The rest of the report is structured as follows. Section 2 includes the related research and the explanation of models in the domain of scenario analysis. Section 3 presents the proposed model and the architecture for generating scenarios outcomes. Section 4 has the experiments performed in use case of e-commerce demand prediction. The last section contains the conclusions and future work for improving the decision intelligence tool.

## 2. RELATED WORK

Generating what-if scenarios is applied in different industries and is researched in businesses and academia. This section reviews the related work in the area of scenario analysis.

Generating scenarios can be effectively accomplished using **traditional methods** such as time-series decomposition, Monte Carlo simulations, sensitivity analysis, regression analysis, and other statistical techniques.
- [[Generating What-If Scenarios for Time Series Data](https://dl.acm.org/doi/10.1145/3085504.3085507)] -> *TODO*
- [[Predicting the Ecological Quality of Rivers: A Machine Learning Approach and a What‑if Scenarios Tool](https://link.springer.com/article/10.1007/s10666-024-09980-y)] -> generating scenarios by applying percentage changes to input variables

**Machine learning methods**
- Fuzzy cognitive maps: [[FCM Expert: Software Tool for Scenario Analysis and Pattern Classification Based on Fuzzy Cognitive Maps](https://www.researchgate.net/publication/328218204_FCM_Expert_Software_Tool_for_Scenario_Analysis_and_Pattern_Classification_Based_on_Fuzzy_Cognitive_Maps)] -> *TODO*
- XGBoost: [[Predicting the Ecological Quality of Rivers: A Machine Learning Approach and a What‑if Scenarios Tool](https://link.springer.com/article/10.1007/s10666-024-09980-y)] -> data cleaning, applying simple transformations to the input datasets and using machine learning forecasting models (XGBoost) to generate the scenarios, model explainability using Shapley Additive exPlanations (SHAP)

## 3. PROPOSED MODEL

![TBWISA framework diagram](/images/tbwisa_framework.png)

### 3.1. Dataset collection and preparation

Steps: data collection, data augmentation.

Domains: ecommerce, ...

Environmental Variables

Variables used

### 3.2. Scenario generation

- defining dependent variables
- defining assumptions
- determining the correlation between variables
- applying transformations to time series

### 3.3. Outcome simulation

- forecasting using transformer [[Attention Is All You Need](https://arxiv.org/abs/1706.03762)]
- determining the optimal window
- determining the outcome

### 3.4 Model explainability

TODO: determine model explainability using Shapley Additive exPlanations (SHAP) if applicable


## 4. EXPERIEMENTS

### Configure environment

- Create environment
```setup
conda create -n whatif-env python=3.10.9 anaconda
```
- Activate environment
```setup
source activate whatif-env
```
- Install requirements
```setup
pip install -r requirements.txt
```
- Run *scenario-generation* notebook

### Dataset

- [Amazon related time series](https://github.com/aws-samples/amazon-forecast-samples/blob/main/library/content/RelatedTimeSeries.md): historic data
- [Amazon target time series](https://github.com/aws-samples/amazon-forecast-samples/blob/main/library/content/TargetTimeSeries.md): extra data

### Evaluation

The framework achieves the following performance:

[image]()

| Model name         | Amazon CE Dataset
| ------------------ | ------------------ |
| TBWISA             | x%                 |

### Results

    - price & demand for product
    - correlation between them
    - demand time series
    - revenue time series
    - optimal window & total revenue

- Discussions

## CONCLUSIONS

## REFERENCES

- [Amazon what-if analysis](https://github.com/aws-samples/amazon-forecast-samples/blob/main/notebooks/advanced/WhatIf_Analysis/WhatIf_Analysis.ipynb)
- [Stock prediction models](https://github.com/huseinzol05/Stock-Prediction-Models/tree/master)

## Contributing

>📋 License: MIT licenses

>📋 Contribution: ...