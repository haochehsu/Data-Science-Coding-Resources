# Data Science Coding Resources

- Online notebooks
  - [Google Colab](#colab-link) (recommended)
  - [Databricks](#databricks-link)
  - [Kaggle](#kaggle-link)
  - [Deepnote](#deepnote-link)
- Local notebooks
  - [Jupyter Notebook](#jupyter-notebook-website)
  - [Jupyter Lab](#jupyter-lab-website)
  - [VS Code](#vs-code-download) (recommended)
- Cloud instances
  - [Google Cloud: Vertex AI](#google-cloud-vertex-ai)
  - [AWS: SageMaker](#aws-sagemaker)
- Regression
  - [Import the libraries](#1-import-the-libraries)
  - [Read the data](#2-read-the-data-and-store-it-in-a-data-frame-df)
  - [Linear regression](#3-linear-regression)
  - [Linear regression with categorical data](#4-linear-regression-with-categorical-data)
  - [Instrumental variable and 2-stage least squares](#5-instrumental-variable-and-2-stage-least-squares)

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

#### 1. Import the libraries

  ```py
  import pandas as pd
  import statsmodels.api as sm
  import statsmodels.formula.api as smf
  from statsmodels.sandbox.regression.gmm import IV2SLS
  ```

#### 2. Read the data and store it in a data frame `df`
  
  ```py
  df = pd.read_csv('data.csv')
  ```
   
#### 3. Linear regression
   
   $y_i= \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \beta_3x_{i\text{Variable with Space}} + \epsilon_i$
   
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

  - `IV2SLS` (with corrected standard errors)
  
    ```py
    endog = df['Y']
    exog = sm.add_constant(df[['X1']])
    
    instrument = sm.add_constant(df[['Instrument']])
    
    model = IV2SLS(endog, exog, instrument=instrument).fit()
    print(model.summary())
    ```
    
  - 2-stage least squares
  
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
          
       2. In `sm.ols`
          
          ```py
          .fit(cov_type='HC1')
          ```
