# Data Science Coding Resources

- Online notebooks
  - [Google Colab](#colab-httpscolabgoogle) (recommended)
  - [Kaggle](#kaggle-httpswwwkagglecomcode)
- Cloud instances
  - [Google Cloud: Vertex AI](#google-cloud-vertex-ai)
  - [AWS: SageMaker](#SageMaker)

---

### Colab (https://colab.google)
<img draggable="false" src="/image/colab_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>

- Notebooks: `Python`, `R`
- Accelerators: CPU, GPU (A100, V100, T4, P100), TPU
- Plans:
  - Free: 16GB memory and GPUs (runtimes limited to 12 hours).
  - Pro ($9.99/M): More memory (32GB) + faster GPUs.
  - Pro+ ($49.99/M): More memory (52GB) + background execution.
- AI: code generation
  - Only available to paid accounts (Pro+: available, Pro: rolling out).

### Kaggle (https://www.kaggle.com/code)
<img draggable="false" src="/image/kaggle_icon.png" alt="" style="width: 3%; padding-bottom:5px; pointer-events:none"/>

- Notebooks: `Python`, `R`
- Accelerators: CPU, GPU (T4, P100), TPU
- 14GB memory.
- Runtimes limited to 9 hours.
- AI: N/A.

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

