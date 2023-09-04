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

```
conda install -c conda-forge jupyter_contrib_nbextensions
```

### Jupyter Lab ([Website](https://jupyterlab.readthedocs.io/en/latest/))
<img draggable="false" src="/image/jl_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>
JupyterLab offers a more extensible and modular environment integrating notebooks, code, and data in a unified workspace.<br>

1. After installing `Anaconda`, open `Terminal` (Mac) or `Anaconda Prompt` (Windows)
2. Execute command
   
   ```
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

