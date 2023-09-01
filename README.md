# Data Science Coding Resources

### Colab (https://colab.google), recommended
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

### Google Cloud ([Vertex AI](https://console.cloud.google.com/vertex-ai/workbench/user-managed?hl=en))
<img draggable="false" src="/image/gc_icon.png" alt="" style="width: 6%; padding-bottom:5px; pointer-events:none"/>

- Setup:
  1. Create a `project`.
  2. `Notebooks` -> `Workbench` -> `User-managed Notebooks` -> `Create New`
     - Default: 4 CPUs, 100GB storage, 15GB memory
- New instance environments: `Python 3` (recommended), `Tensorflow`, `PyTorch`
  - `Tensorflow` and `PyTorch` instances: Available to add a `T4 GPU`.
- Pricing: $0.15 hourly
  - [Pricing](https://cloud.google.com/vertex-ai/pricing) for other instance configurations.
