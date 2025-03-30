## WHAT-IF SCENARIO GENERATION

>The implementation of the TBWISA framework from [Decision intelligence system for generating scenario outcomes using machine learning methods]()

*TODO: decide if the article should be generalized for generating what-if scenarios or specialised for promotion forecasting*

## 1. INTRODUCTION

What-if analysis is a technique for creating simulations that are used by business executives to evaluate the behavior of complex systems under some assumptions. It plays an important role for decision-making in business strategy, financial planning, logistics, and environmental management. Simulating alternative futures helps decision-makers to determine potential risks, forecast outcomes, and make informed choices under uncertainty.
[[Designing what-if analysis: Towards a methodology](https://dl.acm.org/doi/10.1145/1183512.1183523)]

There are different types of scenarios planning, normative vs exploratory approaches and intuitive versus formalized methods for scenarion planning. Scenario planning is most effective when it facilitates strategic thinking, improves organizational learning, and supports decision-making under uncertainty. There are some limitations, such as lack of empirical validation and inconsistent application across studies, stating the need for more rigorous frameworks and outcome-focused evaluation in scenario planning practices [[Types of scenario planning and their effectiveness: A review of reviews](https://www.sciencedirect.com/science/article/pii/S0016328723000575)].

Scenario analysis has applications in forecasting financial performance, payroll planning, investment analysis, inventory planning, employee management and epidemics [[Scenario analysis: a review of methods and applications for engineering and environmental systems](https://link.springer.com/article/10.1007/s10669-013-9437-6)].

The main contribution of this work is defining a framework based on transformers model for generating what-if scenario outcomes in e-commerce applications. The rest of the report is structured as follows. Section 2 includes the related research in the domain of scenario analysis. Section 3 presents the proposed model and the architecture for generating scenarios outcomes. Section 4 has the experiments performed in use case of e-commerce demand prediction when using different scenarios. The last section contains the conclusions and future work for improving the framework.

## 2. RELATED WORK

Generating what-if scenarios is applied in different industries and is researched in businesses and academia. This section reviews the related work in the area of scenario analysis. Traditional methods have been used by different researchers are time-series decomposition, sensitivity analysis, and regression models:
- Giorgini et al. used the foundation of what-if analysis as a simulation-based approach for exploring hypothetical situations, proposing a structured methodology to support scenario-based decision-making processes [[Designing What-If Analysis: Towards a Methodology](https://dl.acm.org/doi/10.1145/1183512.1183523)];
- Bolis and Castelletti conducted a comprehensive review of scenario analysis in engineering and environmental systems, highlighting the techniques used and the challenges in managing uncertainty in such applications [[Scenario analysis: a review of methods and applications for engineering and environmental systems](https://link.springer.com/article/10.1007/s10669-013-9437-6)];
- Wright et al. extended this view with a meta-review, categorizing various scenario planning approaches and evaluating their strategic impact [[Types of scenario planning and their effectiveness: A review of reviews](https://www.sciencedirect.com/science/article/pii/S0016328723000575)].

Machine learning methods increase forecasting accuracy and have been used in scenario generation frameworks:
- Politikos et al. (2024) used XGBoost models to develop a what-if scenario tool for predicting the ecological status of rivers under altered environmental inputs. The tool combines data transformation, machine learning forecasting, and model interpretability using SHAP values [[Predicting the Ecological Quality of Rivers: A Machine Learning Approach and a What‑if Scenarios Tool](https://link.springer.com/article/10.1007/s10666-024-09980-y)];
- Junike et al. (2023) focused on validating machine learning-based scenario generators, especially in economic modeling, where robustness and generalizability have a significant impact [[Validation of Machine Learning Based Scenario Generators](https://link.springer.com/article/10.1007/s10203-023-00412-1)];
- Nápoles et al. introduced FCM Expert, a software tool that facilitates scenario analysis and pattern classification using expert-designed FCMs optimized via machine learning algorithms. Fuzzy cognitive maps (FCMs) are as a flexible modeling technique used for scenario simulation [[FCM Expert: Software Tool for Scenario Analysis and Pattern Classification Based on Fuzzy Cognitive Maps](https://www.researchgate.net/publication/328218204)];
- Liebig and Käfer (2017) proposed a framework for generating what-if scenarios directly from time series data. The system allows users to define alternative futures interactively by modifying descriptive features of the time series, enabling quick feedback and exploration of potential developments [[Generating What-If Scenarios for Time Series Data. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining](https://dl.acm.org/doi/10.1145/3085504.3085507)].

What-if scenarios can be used in promotion forecasting in the retail domain:
- Henzel and Sikora (2022) applied gradient boosting to forecast performance indicators that measure promotion efficiency in FMCG retail, providing insights for optimizing marketing strategies [[Gradient Boosting Application in Forecasting of Performance Indicators Values for Measuring the Efficiency of Promotions in FMCG Retail](https://link.springer.com/article/10.1007/s10100-022-00817-7)];
- Shen et al. (2021) proposed a scalable ML-driven framework to personalize promotional incentives based on customers’ predicted responses, aligning business objectives with user-specific engagement strategies [[A Framework for Massive Scale Personalized Promotion](https://arxiv.org/abs/2101.05491)].

*TODO: research promotion forecasting articles*

## 3. PROPOSED MODEL

A *transformers-based what-if scenario analysis* (TBWISA) framework is proposed. The workflow of the proposed framework contains multiple steps: data collection for gathering the required data from multiple sources and cleaning it, data augmentation for generating synthetic data for incomplete datasets, scenario generation which selects the scenarios that are explored, forecasting for generating future values in the current time series and outcome generation for determining the outcomes for the created scenarios.

![TBWISA framework diagram](/images/tbwisa_framework.png)

### 3.1. Data Collection and Preparation
Historical and contextual e-commerce data, including price, promotions, and seasonal effects, are collected and augmented to enhance model training and forecasting accuracy. The following steps are performed:

- **Data Collection:** Gathering relevant historical data from e-commerce operations and external variables impacting demand, such as pricing, promotions, or seasonal effects;
- ***Data Augmentation:** Applying data enhancement techniques to improve forecasting accuracy;*
    
    *TODO: add data augmentation or remove if not necessary!?*

### 3.2. Transformer Forecasting Model

Transformers were introduced for natural language processing tasks and offered improved results in modeling complex dependencies within sequential data [[Attention Is All You Need](https://arxiv.org/abs/1706.03762)]. The self-attention mechanism allows the dynamic modeling of interactions among elements in a sequence, making it particularly suitable for time series forecasting, where capturing temporal relationships is important.

Unlike traditional sequence-to-sequence Transformers, which contain both encoder and decoder layers, the model used for time series forecasting uses an **encoder-only architecture**. This model captures internal temporal dependencies directly from historical data, without the complexity introduced by decoder layers. It is composed by the following components:

- Embedding Layer;
- Multi-Head Self-Attention;
- Residual Connections and Layer Normalization;
- Feed-Forward Network (Dense layers);
- Output Projection Layer.

The Transformer uses multi-head self-attention to dynamically compute attention scores between all elements within a sequence. Given an embedded sequence $`\ H = [h_1, h_2, ..., h_T]`$, the attention mechanism computes: 

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where $`\ Q, K, V `$ are the Query, Key, and Value matrices obtained by linear transformations of $`\ H `$ and $`\ d_k `$ is the dimension of keys and queries.

Multi-head attention enhances this by combining parallel attention computations:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

The encoder-only transformer offers benefits over traditional forecasting methods, such as ARIMA, LSTM and GRU. It processes all the elements simultaneously, reducing computational overhead compared to recurrent models. The self-attention mechanisms efficiently captures dependencies over long time horizons.
Adaptive Representations: Dynamically adjusts attention to capture relevant temporal contexts effectively.

### 3.3. Scenario Generation
Scenarios are systematically developed by defining dependent variables, initial assumptions, and analyzing variable correlations. Time series transformations ensure that simulations accurately reflect realistic market scenarios and conditions.

The scenario generation starts with historical data analysis to estimate the price elasticity of demand. This elasticity estimation feeds into a non-linear adjustment component to simulate consumer the behavior changes in response to significant price variations. A structural causal model is integrated to determine the cause-effect relationships between variables. The last part is the stochastic adjustment module, which introduces controlled randomness to replicate market uncertainties, increasing scenario realism in uncertain market conditions.

*TODO:*
- *handle **cannibalisation** in promotion planning or generating scenarios*
- *using causation of variables, not correlation*

The components of the architecture are shown below:

#### 3.3.1 Structural Causal Model

The causal impact of price on demand is determined by applying the *structural causal model* [[The book of why](https://dl.acm.org/doi/10.5555/3238230)]. The causal relationships are defined as follows:

- *Price ($P$)* affects *Demand ($D$)*;
- *Demand ($D$)* influences *Sales ($S$)* (revenue);
- Unobserved external factors (e.g., seasonality, brand perception) are captured in $U_D$ and $U_S$.

The structural equations are:

$$
D = \alpha + \beta P + U_D \\
S = \gamma + \delta D + \eta P + U_S
$$

The counterfactual analysis is performed using the *do* operator in the following steps:

1. **Abduction:** estimating the unobserved factors $U_D$ using observed data;
2. **Action:** setting the price to a new value, $P = P_{cf}$.
3. **Prediction:** calculating the counterfactual demand:

$$
D_{P_{cf}} = \alpha + \beta P_{cf} + U_D
$$

The method allows the simulation of realistic scenarios where price interventions are treated as deliberate actions rather than passive observations.

To generate scenarios reflecting deliberate price interventions, the counterfactual demand $D_{cf}$ is calculated by modifying the original structural elasticity equation with adjusted prices and residuals:

$$
D_{cf} = \exp\left(\alpha + E_{adjusted}\cdot\log(P_{cf}) + U_D\right)
$$

where:
- $P_{cf}$ is the counterfactual (adjusted) price after the intervention;
- $E_{adjusted}$ is the elasticity adjusted for non-linear consumer reactions;
- $U_D$ are the residuals from the initial estimation.

#### 3.3.3. Non-linear Elasticity Adjustment

This module adjusts the elasticity to reflect stronger consumer responses to large price changes 
[[Basic Econometrics](https://www.scirp.org/reference/referencespapers?referenceid=1568730)]:

$$
E_{adjusted} = E \cdot \left(1 + 0.5 \cdot (\Delta P)^2\right)
$$

where $\Delta P$ represents the relative price change.

#### 3.3.4. Stochastic Demand Adjustment

Stochastic variations are introduced as:

$$
\text{D}_{stochastic} = \text{D}_{forecasted} \cdot (1 + \text{Randomness})
$$

where randomness is sampled from a normal distribution with defined bounds [[Forecasting: methods and applications](https://robjhyndman.com/forecasting/)].

### 3.4. Optimal Profit Window Model
An optimization approach to identify the optimal profit window from the generated scenarios is used. The goal is to select a continuos time window that maximizes profit while penalizing long windows (from revenue), considering external factors (demand).

The steps for each scenario are as follows:
1. Evaluate candidate windows;
2. Compute generalized penalized scores;
3. Select the highest scoring window;
4. Identify overall best scenario-window combination.

$R = \{r_1, r_2, \dots, r_T\}$ represent a forecasted revenue time series, and $D = \{d_1, d_2, \dots, d_T\}$ represent an external factor time series, such as demand or risk. A window $W(i, \ell)$ is defined as:

$$
W(i, \ell) = \{ r_i, r_{i+1}, \dots, r_{i+\ell-1} \}
$$

A generalized penalized score function $S(W(i, \ell))$ is introduced, incorporating the revenue and external factors. The function penalizes longer windows and considers external metrics to guide optimization:

$$
S(W(i, \ell)) = \frac{1}{\ell} \sum_{j=i}^{i+\ell-1} r_j - \lambda \ell + \gamma \cdot f\left(\frac{1}{\ell} \sum_{j=i}^{i+\ell-1}\frac{d_j}{\max(D)}\right)
$$

where:
- $\frac{1}{\ell} \sum_{j=i}^{i+\ell-1} r_j$: average revenue.
- $\lambda$: window length penalty factor.
- $\gamma$: external factor penalization factor.
- $f$: a function defining how the external factor influences the optimization, typically an inverse or negative correlation.

The optimization problem is formalized as:

$$
(i^*, \ell^*) = \underset{i, \ell}{\mathrm{argmax}} \; S(W(i, \ell))
$$

where $(i^*, \ell^*)$ represent the optimal starting index and length of the profit window. This search uses techniques from dynamic programming and penalized optimization [[Dynamic Programming and Optimal Control](http://athenasc.com/dpbook.html)][[Using penalized contrasts for change-point](https://www.sciencedirect.com/science/article/abs/pii/S0165168405000381)].

## 4. EXPERIEMENTS

### 4.1. Configure environment

- Create environment
```setup
conda create -n whatif-env python=3.10
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

### 4.2. Dataset

Experiments were conducted using publicly available datasets from Amazon and the M5 forecasting accuracy benchmark.

- **[Amazon Related Time Series](https://github.com/aws-samples/amazon-forecast-samples/blob/main/library/content/RelatedTimeSeries.md)**: Dataset with the historical price of items;
- **[Amazon Target Time Series](https://github.com/aws-samples/amazon-forecast-samples/blob/main/library/content/TargetTimeSeries.md)**: Dataset with the historical demand of items;
- **[M5 Forecasting Accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy)**: Dataset with the historical price and demand of items and external events.

### 4.3. Methodology
Price scenarios, ranging from -15% to +15%, were generated to simulate varying market conditions. Demand was recalculated using elasticity and stochastic adjustments. Evaluation metrics included RMSE, MAE, Revenue Uplift, Inventory Turßnover, Demand Fulfillment, and ROI.

Optimal window hyperparameters:
- Revenue Penalty Factor ($\lambda$): 0.1
- External Factor Penalization ($\gamma$): 1.5
- Window Sizes: $\ell \in \{2, 3, 4, 5\}$

### 4.4. Results
The effectiveness of scenarios is validated using multiple metrics across economic, accuracy, inventory, and business dimensions ([Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)):

- price & demand for product
- correlation between them
- demand time series
- revenue time series
- total revenue
- optimal profit window

Optimal Profit window:

![Optimal profit window image](images/optimal_window.png)

Revenue Metrics:

- Total Revenue ($R$):

$$
R = \sum_{t=i}^{i+l-1} P_t \cdot D_t
$$

- Average Revenue ($R_{avg}$):

$$
R_{avg} = \frac{R}{l}
$$

- Revenue Uplift ($RU$):

$$
RU = \frac{R_{scenario}}{R_{baseline}} - 1
$$

Forecast Accuracy Metrics:

- Mean Absolute Error ($MAE$):

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|D_t^{forecast} - D_t^{actual}|
$$

- Root Mean Squared Error ($RMSE$):

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(D_t^{forecast} - D_t^{actual})^2}
$$

Inventory Management Metrics:

- Inventory Turnover ($IT$):

$$
IT = \frac{\text{Units Sold}}{\text{Average Inventory}}
$$

- Demand Fulfillment Rate ($DFR$):

$$
DFR = \frac{\text{Units Sold}}{\text{Units Forecasted}}
$$

Business Metrics:

- Return on Investment ($ROI$):

$$
ROI = \frac{R - \text{Costs}}{\text{Costs}}
$$

- Window Compactness ($WC$):

$$
WC = \frac{l}{T}
$$

| Scenario           | Total Revenue | Avg Revenue per Period | Revenue Uplift (%) | MAE (Demand) | RMSE (Demand) | Inventory Turnover | Demand Fulfillment (%) | ROI (%) | Window Compactness (%) |
|--------------------|---------------|------------------------|--------------------|--------------|---------------|--------------------|------------------------|---------|------------------------|
| Price Change -15%  | 234782.70     | 33540.39               | 18.68              | 39.63        | 45.49         | 27.09              | 100.0                  | 46856.54| 100.0                  |
| Price Change -12%  | 231150.91     | 33021.56               | 16.84              | 34.97        | 40.11         | 26.71              | 100.0                  | 46130.18| 100.0                  |
| Price Change -10%  | 228415.36     | 32630.77               | 15.46              | 31.42        | 36.27         | 26.43              | 100.0                  | 45583.07| 100.0                  |
| Price Change -7%   | 224290.94     | 32041.56               | 13.37              | 25.98        | 31.08         | 25.99              | 100.0                  | 44758.19| 100.0                  |
| Price Change -5%   | 221415.45     | 31630.78               | 11.92              | 22.19        | 28.24         | 25.68              | 100.0                  | 44183.09| 100.0                  |
| Price Change +5%   | 212290.36     | 30327.19               | 7.31               | 17.09        | 26.79         | 24.32              | 100.0                  | 42358.07| 100.0                  |
| Price Change +7%   | 211631.01     | 30233.00               | 6.97               | 17.23        | 28.92         | 24.04              | 100.0                  | 42226.20| 100.0                  |
| Baseline (0%)      | 197835.96     | 28262.28               | 0.00               | 0.00         | 0.00          | 25.12              | 100.0                  | 39467.19| 100.0                  |

### 4.5. Discussions

## CONCLUSIONS

## RESOURCES

- [Amazon what-if analysis](https://github.com/aws-samples/amazon-forecast-samples/blob/main/notebooks/advanced/WhatIf_Analysis/WhatIf_Analysis.ipynb)
- [Stock prediction models](https://github.com/huseinzol05/Stock-Prediction-Models/tree/master)

## REFERENCES


## Contributing

>📋 License: MIT licenses

>📋 Contribution: ...