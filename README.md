# Data Science Coding Resources

- Online Notebooks
  - [Google Colab](#colab-link) (recommended)
  - [Databricks](#databricks-link)
  - [Kaggle](#kaggle-link)
  - [Deepnote](#deepnote-link)
- Local Notebooks
  - [Jupyter Notebook](#jupyter-notebook-website)
  - [Jupyter Lab](#jupyter-lab-website)
  - [VS Code](#vs-code-download) (recommended)
- Cloud Instances
  - [Google Cloud: Vertex AI](#google-cloud-vertex-ai)
  - [AWS: SageMaker](#aws-sagemaker)
- Linear Regression
  - [Import the libraries](#1-import-the-libraries)
  - [Read the data](#2-import-the-libraries)
  - [Linear regression](#3-linear-regression)
  - [Linear regression with categorical data](#4-linear-regression-with-categorical-data)
  - [Instrumental variable and 2-stage least squares](#5-instrumental-variable-and-2-stage-least-squares)
- Panel Data
  - [Install the library](#1-install-the-library)
  - [Import the libraries](#2-read-the-data-and-store-it-in-a-data-frame-df)
  - [Read the data](#3-read-the-data-and-store-it-in-a-data-frame-df)
  - [Pooled model](#4-pooled-model)
  - [Fixed-effects models](#5-fixed-effects-models)
    - [Adding entity-fixed effects](#a-adding-entity-fixed-effects)
    - [Within transformation](#b-within-transformation-within-estimator)
    - [First-difference model](#c-first-difference-model)
    - [Adding time-fixed effects](#d-adding-time-fixed-effects)
  - [Random effects model](#6-random-effects-model)
    - [Hausman test](#hausman-test)
  - [Between model](#7-between-model-between-estimator)
- Statistical Time Series Forecasting
  - [Import the libraries](#1-import-the-libraries)
  - [Autoregressive model](#2-autoregressive-model)
  - [Moving average model](#3-moving-average-model)
  - [Autoregressive moving average model](#4-autoregressive-moving-average-model)
  - [Autoregressive integrated moving average model](#5-autoregressive-integrated-moving-average-model)
  
---

### Colab ([Link](https://colab.google))
<img draggable="false" src="/image/colab_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>

- Notebooks: `Python`, `R`
- Accelerators: CPU, GPU (A100, V100, T4, P100), TPU
- Plans:
  - Free: 16GB memory and GPUs (runtimes limited to 12 hours).
  - Pro ($9.99/M): More memory (32GB) + faster GPUs.
  - Pro+ ($49.99/M): More memory (52GB) + background execution.
- AI: code generation
  - Only available to paid accounts (Pro+: available, Pro: rolling out).

### Databricks ([Link](https://community.cloud.databricks.com/))
<img draggable="false" src="/image/db_icon.png" alt="" style="width: 5%; padding-bottom:5px; pointer-events:none"/>

- Sign-up: [Link](https://www.databricks.com/try-databricks#account)
  1. Enter name, company, email, and title, and click Continue.
  2. On the `Choose a cloud provider` page, click the `Get started with Community Edition` link at the bottom.
- Notebooks: `Python`, `R`, `SQL`, `Scala`.
- AI: N/A.

### Kaggle ([Link](https://www.kaggle.com/code))
<img draggable="false" src="/image/kaggle_icon.png" alt="" style="width: 3%; padding-bottom:5px; pointer-events:none"/>

- Notebooks: `Python`, `R`
- Accelerators: CPU, GPU (T4, P100), TPU
- 14GB memory.
- Runtimes limited to 9 hours.
- AI: N/A.

### Deepnote ([Link](https://deepnote.com/workspace))
<img draggable="false" src="/image/dn_icon.png" alt="" style="width: 5%; padding-bottom:5px; pointer-events:none"/>

- Real-time collaboration with multiple users.
- Notebooks: `Python`, `R`
- Accelerators: K80 (upgrade required)
- Plans:
  - Free: 2 CPUs, 5GB memory, and 5GB storage (3 collaborators).
  - Plus (1.02/h): 4 CPUs, 16GB memory.
  - Performance (3.97/h): 16 CPUs, 64GB memory.
  - High memory (5/h): 16 CPUs, 128GB memory.
  - GPU ($7.56/h): 4 CPUs, 60GB memory.
- AI: N/A.

---
### Jupyter Notebook ([Website](https://jupyter.org/))  
<img draggable="false" src="/image/jn_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>
Jupyter Notebook is a web application focused on creating and managing individual notebook documents.<br><br>

1. Install Anaconda ([Download](https://www.anaconda.com/download#downloads))
2. Open `Terminal` (Mac) or `Anaconda Prompt` (Windows)
3. Type the command `jupyter notebook`
 
Once the browser window opens with the notebook, click New to start a new notebook.

**Install Jupyter Notebook extensions**

Go to `Terminal` (Mac) or `Anaconda Prompt` (Windows):

```sh
conda install -c conda-forge jupyter_contrib_nbextensions
```

### Jupyter Lab ([Website](https://jupyterlab.readthedocs.io/en/latest/))
<img draggable="false" src="/image/jl_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>
JupyterLab offers a more extensible and modular environment integrating notebooks, code, and data in a unified workspace.<br>

1. After installing `Anaconda`, open `Terminal` (Mac) or `Anaconda Prompt` (Windows)
2. Execute command
   
   ```sh
   conda install -c conda-forge jupyterlab
   ```
3. Type the command `jupyter lab`

### VS Code ([Download](https://code.visualstudio.com/))
<img draggable="false" src="/image/vscode_icon.png" alt="" style="width: 5%; padding-bottom:5px; pointer-events:none"/>

1. Install `Anaconda`.
2. Locate the `Extensions` tab on the left vertical panel.
3. Search and install extensions `Python` and `Jupyter`.
   - It will automatically install the necessary dependent extensions.
4. `File` → `New File...` → `Jupyter Notebook`
5. Click `Select Kernel` on the top right corner → `Python Environment` → `Anaconda3 (Python)`
   
- AI: `GitHub Copilot` (extension)
  - Provides autocomplete-style suggestions from an AI pair programmer as you code. 

---

### Google Cloud ([Vertex AI](https://console.cloud.google.com/vertex-ai/workbench/user-managed?hl=en))
<img draggable="false" src="/image/gc_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>

- Setup:
  1. Create a `project`.
  2. `Notebooks` -> `Workbench` -> `User-managed Notebooks` -> `Create New`
     - Default: 4 CPUs, 15GB memory, 100GB storage.
- New instance environments: `Python 3` (recommended), `Tensorflow`, `PyTorch`
  - `Tensorflow` and `PyTorch` instances: Available to add a `T4 GPU`.
- Pricing: $0.15 hourly
  - [Pricing](https://cloud.google.com/vertex-ai/pricing) for other instance configurations.

### AWS ([SageMaker](https://us-west-1.console.aws.amazon.com/sagemaker/home?region=us-west-1#/notebook-instances))
<img draggable="false" src="/image/aws_icon.png" alt="" style="width: 8%; padding-bottom:5px; pointer-events:none"/>

- Setup:
  1. `Create notebook instance`.
  2. Notebook instance type: `ml.t3.medium`.
     - 2 CPUs, 4GB memory, 125GB storage.
  3. [Pricing](https://aws.amazon.com/sagemaker/pricing/) for other instance configurations.

---

### Regression

These are the first *two columns* of the data:

|     Y    |     X1    |     X2    | Category | Variable with Space | Instrument |
|:--------:|:---------:|:---------:|:--------:|:-------------------:|------------|
| 0.129082 |  0.496714 | -0.349898 |     B    |      -0.828995      | -1.415371  |
| 1.857186 | -0.138264 |  0.350462 |     B    |      -0.560181      | -0.420645  |

The `Category` variable includes three categories: `A`, `B`, and `C`. The variable named `Variable with Space` has spaces in its name. `Y` is the dependent/outcome variable.

#### 1. Import the libraries

  ```py
  import pandas as pd
  import statsmodels.api as sm
  import statsmodels.formula.api as smf
  from statsmodels.sandbox.regression.gmm import IV2SLS
  ```

#### 2. Read the [data](https://github.com/haochehsu/Data-Science-Coding-Resources/blob/main/data.csv) and store it in a data frame `df`
  
  ```py
  df = pd.read_csv('data.csv')
  ```
   
#### 3. Linear regression
   
   $y_i= \beta_0 + \beta_1x_{1i} + \beta_2x_{2i} + \beta_3x_{\text{variable with space}, i} + \epsilon_i$
   
  - Passing in variables
  
    ```py
    y = df['Y']
    X = df[['X1', 'X2', 'Variable with Space']]
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    print(model.summary())

  - Passing in formula
     
    ```py
    model = smf.ols(formula='Y ~ X1 + X2 + Q("Variable with Space")', data=df).fit()
    print(model.summary())
    ```
   
#### 4. Linear regression with categorical data

$y_i= \beta_0 + \beta_1x_{i} + \beta_2 \cdot \mathbb{1}\lbrace \text{Category} = A\rbrace_i + \beta_3 \cdot \mathbb{1}\lbrace \text{Category} = B\rbrace_i + \beta_4 \cdot \mathbb{1}\lbrace \text{Category} = C\rbrace_i + \epsilon_i$
  
  - Passing in variables
  
    ```sh
    dummies = pd.get_dummies(df['Category'], drop_first=True, sparse=True, prefix="Category")
    X = pd.concat([df[['X1']], dummies], axis=1)
    
    X = sm.add_constant(X)
    
    model = sm.OLS(df['Y'], X).fit()
    print(model.summary())
    ```
    
  - Passing in formula
  
    ```sh
    model = smf.ols(formula='Y ~ X1 + C(Category)', data=df).fit()
    print(model.summary())
    ```

#### 5. Instrumental variable and 2-stage least squares

Suppose $x_{1i}$ is correlated with $\epsilon_i$ (endogenous $x_{1i}$). `Instrument` is correlated with $x_{1i}$ but uncorrelated with $\epsilon_i$:

$x_{1i} = \alpha_0 + \alpha_{1i} \cdot \text{Instrument}_i + u_i$

$y_{i} = \beta_0 + \beta_{1i} \cdot \widehat{x}_{1i} + \epsilon_i$

  - `IV2SLS` (with corrected standard errors)
  
    ```py
    endog = df['Y']
    exog = sm.add_constant(df[['X1']])
    
    instrument = sm.add_constant(df[['Instrument']])
    
    model = IV2SLS(endog, exog, instrument=instrument).fit()
    print(model.summary())
    ```
    
  - `sm.OLS` (passing in variable): manual 2-stage least squares
  
    ```py
    # First Stage
    X_first_stage = sm.add_constant(df[['Instrument']])
    model_first_stage = sm.OLS(df['X1'], X_first_stage).fit()
    df['X1_hat'] = model_first_stage.predict(X_first_stage)
    
    # Second Stage
    X_second_stage = sm.add_constant(df['X1_hat'])
    model_second_stage = sm.OLS(df['Y'], X_second_stage).fit()
    
    print(model_second_stage.summary())
    ```

#### 6. Other options

   - Clustered standard errors
     - Cluster based on the `Category` variable: pass the arguments to the `.fit()` method.

       ```py
       .fit(cov_type='cluster', cov_kwds={'groups': df['Category']})
       ```

   - Robust standard error
     - Debias the covariance estimator by using a degree of freedom adjustment to obtain the same robust standard error as in STATA.
       
       1. In `IV2SLS`

          ```py
          .fit(debiased=True)
          ```
          
       2. In `sm.OLS` (passing in variable) and `smf.ols` (passing in formula)
          
          ```py
          .fit(cov_type='HC1')
          ```
          
  - Regression without an intercept
    
    - In `IV2SLS`: don't `sm.add_constant()`.
          
    - In `smf.ols` (passing in formula): add `-1` into the formula
   
      ```
      model = smf.ols(formula='Y ~ X1 + X2 - 1', data=df).fit()
      print(model.summary())
      ```
      
---

### Panel Data

#### 1. Install the library

  ```python
  pip install linearmodels
  ```

#### 2. Import the libraries

  ```py
  import pandas as pd
  from linearmodels.panel import PooledOLS
  from linearmodels.panel import RandomEffects
  
  from statsmodels.api import OLS
  from statsmodels.tools.tools import add_constant
  ```

#### 3. Read the [data](https://github.com/haochehsu/Data-Science-Coding-Resources/blob/main/panel_data.csv) and store it in a data frame `df`
  
  ```py
  df = pd.read_csv('panel_data.csv')
  ```

> [!NOTE]
> Panel data combines features of both **cross-sectional** data (observations at a single point in time) and **time-series** data (observations of a single entity across multiple time points), this data tracks groups of entities, like people or firms, over several time periods.
> 
> It can be used to analyze individual heterogeneity and provide control for unobserved characteristics, potentially addressing endogenous concerns by accounting for factors that remain consistent over time.

Each entity $i$ is observed over $t$ periods.

#### 4. Pooled model

  $Y_{it} = \beta_0 + \beta_1 X_{it} + \epsilon_{it}$

  Pools the data and runs a regression.

  ```python
  pooled = PooledOLS(df.Y, df[['X']]).fit()
  print(pooled)
  ```

#### 5. Fixed-effects models

  $Y_{it} = \alpha_i + \beta_1 X_{it} + \epsilon_{it}$ where $\alpha_i$ captures the unobserved, time-invariant individual effects.

  > [!IMPORTANT]
  > In panel data, heterogeneity across entities (e.g. individuals, firms) can lead to omitted variable bias. Time-invariant firm characteristics, such as *company culture*, are often unobservable to econometricians. These uncontrolled characteristics are absorbed into the error term and can be **correlated** with other observed firm characteristics. To account for potential endogeneity, we add/control fixed effects into the model.

The fixed-effects models use the "within variation" (variation of an entity over time):

##### A. Adding entity-fixed effects

  ```python
  FE = PanelOLS(df.Y, df[['X']], entity_effects=True).fit()
  print(FE)
  ```

  This is equivalent to running a **dummy variable regression**:

  ```python
  # generate dummy variables and add them to the dataframe
  dummies = pd.get_dummies(df.reset_index()['id'], drop_first=True).set_index(df.index)
  df_with_dummies = pd.concat([df, dummies], axis=1)

  # add an intercept
  X = add_constant(pd.concat([df_with_dummies.X, df_with_dummies.iloc[:, 3:]], axis=1))

  dummy_regression = OLS(df_with_dummies.Y, X.astype(float)).fit()
  print(dummy_regression.summary())
  ```

##### B. Within transformation (within estimator)

  Subtract the average (taken across time) of each variable from the variable itself to eliminate fixed effects.

  ```python
  grouped_means = df.groupby('id').mean()
  df_with_means = df.merge(grouped_means, on='id', suffixes=('', '_mean'))
  df_with_means['Y_within'] = df_with_means['Y'] - df_with_means['Y_mean']
  df_with_means['X_within'] = df_with_means['X'] - df_with_means['X_mean']
  
  X = add_constant(df_merged['X_within'])
  model = sm.OLS(df_merged['Y_within'], X).fit()
  print(model.summary())
  ```

##### C. First-difference model

  $\Delta Y_{it} = \beta_1 \Delta X_{it} + \Delta\epsilon_{it}$ where $\Delta A_{it} \equiv A_{it}-A_{i, t-1}$

  If we have a **balanced panel** (all entities have observations for all periods), first differencing eliminates the fixed effect. This allows us to focus on only the **changes** within each entity over time, effectively removing any time-invariant characteristics.

  ```python
  df_diff = df.groupby('id').diff().dropna()
  first_diff = PanelOLS(df_diff.Y, df_diff[['X']], entity_effects=False).fit()
  print(first_diff)
  ```

##### D. Adding time-fixed effects

$Y_{it} = \alpha_i + \lambda_i + \beta_1 X_{it} + \epsilon_{it}$ where $\lambda_i$ captures the unobserved, entity-invariant effects specific to each time period $t$.

  ```python
  FE_with_time_effects = PanelOLS(df.Y, df[['X']], entity_effects=True, time_effects=True).fit()
  print(FE_with_time_effects)
  ```

#### 6. Random effects model

  $Y_{it} = \beta_0 + \beta_1 X_{it} + (\alpha_i + \epsilon_{it})$ where $\alpha_i\sim N(0, \sigma_{\alpha}^2)$ and $\sigma_{\alpha}^2$ is the variance of the entity specific effect.

  > [!IMPORTANT]
  > The random effect model assumes that the time-invariant entity-specific effect (e.g. company culture) is a random variable drawn from a normal distribution and is **uncorrelated** with the other observed characteristics which allows the random effects model to leverage both "within" and "between" variations.

  Based on the assumed distribution of the $\alpha_i$'s, heteroscedasticity violates the OLS assumptions. Therefore, we use generalized least squares (GLS) to estimate the model.
  
  ```python
  RE = RandomEffects(df.Y, df[['X']]).fit()
  print(RE)
  ```

##### Hausman test

The test is used to determine whether to use **fixed effects** or a **random effects** model in panel data analysis.

- $H_0$: Errors are uncorrelated with the regressors. The random effects model is consistent and there are no differences in the coefficients of the fixed effects and random effects model.
- Hausman test statistics:
  $H = \left(\beta_{FE} - \beta_{RE}\right)'\times \left[\text{Var}(\beta_{FE})-\text{Var}(\beta_{RE})\right]^{-1}\times \left(\beta_{FE} - \beta_{RE}\right)$ where $\beta_{FE}$ and $\beta_{RE}$ are vectors of coefficients obtained from the two models.

```python
from scipy.stats import chi2

FE = PanelOLS(df.Y, df[['X']], entity_effects=True).fit()
RE = RandomEffects(df.Y, df[['X']]).fit()

diff_params = FE.params - RE.params
diff_cov = FE.cov - RE.cov

hausman_statistics = diff_params.T.dot(np.linalg.inv(diff_cov)).dot(diff_params)

# DF: #Xs being tested
degree_freedom = len(diff_params)

p_value = chi2.sf(hausman_statistics, degree_freedom)
print("Hausman Test Statistic:", hausman_statistics)
print("p-value:", p_value)
```

#### 7. Between model (between estimator)

  ![between](https://latex.codecogs.com/svg.image?\inline&space;\bg{white}\overline{Y}_{i}=\beta_0&plus;\beta_1\overline{X}_i&plus;\overline{\epsilon}_{i})

The model examines the "between variation" (variation across entities/cross-sectional at a particular point in time) in the data by averaging each variable over time for entity $i$.

  ```python
  df_mean = df.groupby('id').mean().reset_index()
  X_mean = df_mean[['X']]
  y_mean = df_mean['Y']
  between_model = OLS(y_mean, X_mean).fit()
  print(between_model.summary())
  ```

---

### Statistical Time Series Forecasting

#### 1. Import the libraries

  ```python
  from statsmodels.tsa.arima_process import ArmaProcess
  from statsmodels.tsa.ar_model import AutoReg
  from statsmodels.tsa.arima.model import ARIMA
  ```

#### 2. Autoregressive model

We use an `AR(2)` model without a constant as an example. In a second-order autoregressive model, the value at time 
$t$ is determined by its two previous values and is represented as follows:

$(1 - 0.8L^2 - 0.3L^2)y_t = \epsilon_t \Longrightarrow y_t = 0.8y_{t-1} + 0.3y_{t-2} + \epsilon_t$

and can be generated by the following code:

```python
ar = np.array([1, 0.8, 0.3])
ma = np.array([1]) # MA(0): no lagged error terms
ar_process = ArmaProcess(ar, ma)
ar_data = ar_process.generate_sample(nsample=10000)
```

Estimating an `AR(2)` model without constant:

```python
model_ar = AutoReg(ar_data, lags=2, trend='n') # without constant
result_ar = model_ar.fit()
print(f"AR parameters: {result_ar.params}")
```

#### 3. Moving average model

In an `MA(2)` model, the current value of a time series is defined by its current and two previous shocks:

$y_t = (1 + 0.3L^1 + 0.9L^2)ε_t \Longrightarrow y_t = ε_t + 0.3ε_{t-1} + 0.9ε_{t-2}$

and can be generated by the following code:

```python
ar = np.array([1]) # AR(0): no autoregressive term
ma = np.array([1, 0.3, 0.9])
ma_process = ArmaProcess(ar, ma)
ma_data = ma_process.generate_sample(nsample=10000)
```

Estimating an `MA(2)` model without constant:

```python
model_ma = ARIMA(ma_data, order=(0,0,2), trend='n')
result_ma = model_ma.fit()
print(f"MA parameters: {result_ma.params}")
```

#### 4. Autoregressive moving average model

The `ARMA(2,1)` model combines the components of `AR(2)` and `MA(1)`:

$(1 - 0.85L^1 - 0.3L^2)y_t = (1 + 0.6L^1)\epsilon_t \Longrightarrow y_t = 0.85y_{t-1} + 0.3y_{t-2} + ε_t + 0.6\epsilon_{t-1}$

```python
ar = np.array([1, 0.85, 0.3]) 
ma = np.array([1, 0.6])
arma_process = ArmaProcess(ar, ma)
arma_data = arma_process.generate_sample(nsample=10000)
```

Estimating an `ARMA(2,0,1)` model without constant:

```python
model_arma = ARIMA(arma_data, order=(2,0,1), trend='n')
result_arma = model_arma.fit()
print(f"ARMA parameters: {result_arma.params}")
```

#### 5. Autoregressive integrated moving average model

The `ARIMA(2,1,1)` model combines the components of `AR(2)` and `MA(1)`, and also accounts for any non-stationarity through the (first-order) `differencing` term:

$(1 - 0.85L^1 - 0.3L^2)(1-L^1)y_t = (1 + 0.6L^1)\epsilon_t \Longrightarrow y_t = 1.85y_{t-1} - 0.55y_{t-2} - 0.3y_{t-3} + \epsilon_t + 0.6\epsilon_{t-1}$

Building on the previously generated `ARMA(2,1)` data, we integrate the series once to obtain `ARIMA(2,1,1)` data:

```python
arima_data = np.cumsum(arma_data)
```

Estimating an `ARIMA(2,1,1)` model without constant:

```python
model_arima = ARIMA(arima_data, order=(2,1,1), trend='n')
result_arima = model_arima.fit()
print(f"ARIMA parameters: {result_arima.params}")
```
