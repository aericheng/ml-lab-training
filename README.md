# ml-lab-training

A collection of machine learning coursework exercises: PyTorch notebooks/scripts for image classification, diffusion-model image generation, and unsupervised anomaly detection.

## Contents

| Path | Topic | Notes |
|------|-------|-------|
| `ML_hw3_CNN.py` | Image classification (Food-11, 11 classes) — solution | ResNet18 trained from scratch with mixup augmentation and K-fold cross-validation (`sklearn.model_selection.KFold`) |
| `ML_hw3_CNN_example.py` | Image classification — provided starter code | Baseline CNN + dataset download steps for the same Food-11 task |
| `ML/HW6/hw6-diffusion-model.ipynb` | Diffusion model (DDPM) image generation | Builds a Gaussian diffusion model on top of the [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) implementation; `submission_DDPM/`, `submission_vision1/`, `results_vision1/`, and the `.gif`/`.jpg` files are generated sample outputs from training runs |
| `ML/HW8/` | Anomaly detection (one-class learning on face images) | Has its own `README.md`/`REPORT.md`; multiple autoencoder-family baselines (`memae`, `deepsvdd`, `cutpaste`, `patchsvdd`, `resnet`, ...) plus ensembling scripts |

`ML_hw3_*.py` and `ML/HW6/hw6-diffusion-model.ipynb` are exported from Kaggle/Colab notebooks — they contain shell-magic lines (`!pip install ...`, `!nvidia-smi`, `!gdown ...`) and are meant to be run cell-by-cell in that kind of environment, not as a plain standalone Python script.

## How to run

- **HW3** (`ML_hw3_CNN.py` / `ML_hw3_CNN_example.py`): open in a notebook environment with a GPU (Kaggle/Colab), install `torch`, `torchvision`, `pandas`, `numpy`, `tqdm`, `scikit-learn`, `Pillow`, run the download cell to fetch the Food-11 dataset, then run the training cells.
- **HW6** (`ML/HW6/hw6-diffusion-model.ipynb`): open in Jupyter/Colab with a GPU; the notebook installs `denoising-diffusion-pytorch` itself in an early cell.
- **HW8** (`ML/HW8/`): see `ML/HW8/README.md` for its own setup and run instructions.

## License

MIT — see `LICENSE`.
