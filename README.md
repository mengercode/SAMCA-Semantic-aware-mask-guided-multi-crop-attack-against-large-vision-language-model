# SAMCA-Semantic-aware-mask-guided-multi-crop-attack-against-large-vision-language-model
# SAMCA: Semantic-Aware Mask-Guided Multi-Crop Attack against Large Vision-Language Models

## Overview

This repository contains the official implementation of **SAMCA (Semantic-Aware Mask-Guided Multi-Crop Attack)**, a black-box transferable adversarial attack framework for Large Vision-Language Models (LVLMs).

SAMCA addresses a key limitation of previous crop-based attacks such as M-Attack: randomly sampled crop regions often fall into background areas rather than semantically important foreground objects, resulting in inefficient use of the perturbation budget.

To overcome this issue, SAMCA introduces semantic priors generated from DINOv2 attention maps and incorporates them throughout the attack pipeline, including crop sampling, loss weighting, and gradient modulation. The resulting perturbations are concentrated on semantically important regions, improving transferability against commercial LVLMs.

## Main Contributions

1. Semantic mask generation using DINOv2 Attention Rollout.
2. Mask-guided importance sampling for crop selection.
3. Spatially adaptive mask-modulated PGD optimization.
4. Improved black-box transferability against commercial LVLMs.

## Repository Structure

SAMCA/
│
├── generate_mask_npy.py
├── attack_generate_adversarial_images.py
├── requirements.txt
└── README.md

### File Description

| File                                    | Description                                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------ |
| `generate_mask_npy.py`                  | Stage 1: Generate semantic masks from DINOv2 attention maps and save them as `.npy` files. |
| `attack_generate_adversarial_images.py` | Stage 2: Generate adversarial examples using mask-guided multi-crop optimization.          |


## Dataset

Experiments are conducted on the NIPS 2017 Adversarial Learning Development Dataset.

Dataset Source:

https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attacks/data

The dataset is not redistributed in this repository. Users should download it from the official source and comply with the original license terms.

## Requirements

Python 3.8+

Recommended environment:

1. CUDA-enabled GPU
2. PyTorch >= 2.0

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
torchvision
transformers
safetensors
numpy
Pillow
opencv-python
tqdm
```

---

## Preparing DINOv2

Download the DINOv2 ViT-L/14 model from Hugging Face:

https://huggingface.co/facebook/dinov2-large

Example:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-large', local_dir='./dinov2-large')"
```

---

## Usage

### Step 1: Prepare Dataset

Organize the dataset as follows:

```text
data/
├── images/
├── targets/
└── masks/
```

---

### Step 2: Generate Semantic Masks

```bash
python generate_mask_npy.py \
    --image_dir data/images \
    --hf_dir ./dinov2-large \
    --out_dir data/masks \
    --sparsity 0.5 \
    --beta_floor 0.05 \
    --take_last_k 6 \
    --out_res 224
```

Generated masks will be stored as `.npy` files.

---

### Step 3: Generate Adversarial Examples

```bash
python attack_generate_adversarial_images.py \
    --src_dir data/images \
    --tgt_dir data/targets \
    --mask_dir data/masks \
    --out_dir data/adversarial \
    --resolution 336 \
    --steps 300 \
    --k_crops 4
```

Generated adversarial images will be saved to:

```text
data/adversarial/
```

---

## Methodology

### Stage 1: Semantic Mask Generation

1. Extract multi-layer self-attention maps from DINOv2.
2. Perform Attention Rollout to aggregate global attention.
3. Obtain CLS-token attention distribution.
4. Generate a spatial importance map.
5. Apply adaptive thresholding and morphological dilation.
6. Produce the final soft semantic mask.

### Stage 2: Mask-Guided Attack

1. Load source images and target images.
2. Sample crop regions according to mask importance.
3. Compute crop-weighted feature alignment loss.
4. Optimize perturbations using MI-FGSM and PGD.
5. Save adversarial examples.

---

## Evaluation

The evaluation protocol follows the framework introduced in M-Attack.

Relevant components include:

* Black-box caption generation
* Keyword Matching Rate (KMR)
* Attack Success Rate (ASR)

The evaluation scripts are adapted from the publicly available M-Attack framework and are not included in this repository.

M-Attack Project:

https://github.com/VILA-Lab/M-Attack

---

## Reproducibility Notes

Commercial LVLM evaluations reported in the paper (e.g., GPT-4o, Gemini, and Claude) require access to the corresponding proprietary APIs and are therefore not included in this repository.

This repository provides the adversarial example generation pipeline required to reproduce the attack methodology.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{gao2026samca,
  title={SAMCA: Semantic-Aware Mask-Guided Multi-Crop Attack against Large Vision-Language Models},
  author={Gao, Xinyan and Zhang, Baiwen and Xu, Meng and Wu, Di and Liu, Feiran and Liu, Tong},
  journal={Under Review},
  year={2026}
}
```

---

## Acknowledgements

This work uses:

* DINOv2 for semantic attention extraction.
* CLIP visual encoders for surrogate optimization.
* NIPS-2017 Adversarial Learning Dataset.
* Components of the evaluation protocol adapted from the M-Attack framework.

---

## License

This repository is released for academic research and reproducibility purposes only.

Users are responsible for complying with the licenses of all third-party datasets and models used in this project.
