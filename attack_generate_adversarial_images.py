# attack.py

import os
import math
import rando
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel, CLIPImageProcessor

# ===================== 1. Configuration (CLI-based, NOT Kaggle-specific) =====================

class Config:
    def __init__(self, args):

        self.src_dir = args.src_dir
        self.tgt_dir = args.tgt_dir
        self.mask_dir = args.mask_dir
        self.out_dir = args.out_dir

        # ensemble models (same as M-Attack design)
        self.ensemble_models = args.models

        # attack hyperparameters
        self.epsilon = 8 / 255
        self.alpha = 1.0 / 255 * 16
        self.steps = 300
        self.momentum = 0.9

        # unified input resolution (LVLM / ensemble consistency)
        self.input_res = args.resolution

        # M-Attack sampling params
        self.k_crops = 1
        self.crop_scale = (0.5, 0.9)
        self.crop_aspect = (0.75, 1.33)

        # semantic mask control
        self.mask_gamma = 2.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 2. Ensemble CLIP Extractor =====================

class EnsembleCLIPExtractor(nn.Module):
    def __init__(self, model_paths, device):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()
        self.preprocessors = []
        self.expected_sizes = []

        print(f"[INFO] Loading {len(model_paths)} CLIP models...")

        for path in model_paths:
            model = CLIPModel.from_pretrained(path).to(device).eval()
            proc = CLIPImageProcessor.from_pretrained(path)

            self.models.append(model)
            self.expected_sizes.append(model.config.vision_config.image_size)

            mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device)
            std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device)

            self.preprocessors.append((mean, std))

    def forward(self, x):

        feats = []

        for i, model in enumerate(self.models):

            target = self.expected_sizes[i]

            # resize to each CLIP backbone size
            x_resized = F.interpolate(
                x, size=(target, target),
                mode="bicubic",
                align_corners=False
            )

            mean, std = self.preprocessors[i]
            x_norm = (x_resized - mean) / std

            feat = model.get_image_features(pixel_values=x_norm)
            feat = F.normalize(feat, dim=-1)

            feats.append(feat)

        return torch.stack(feats).mean(dim=0)

# ===================== 3. Crop sampling (M-Attack + semantic extension) =====================

def sample_crop_box(H, W, mask):

    s = random.uniform(0.5, 0.9)
    ar = random.uniform(0.75, 1.33)

    crop_h = min(H, max(1, int(H * s / math.sqrt(ar))))
    crop_w = min(W, max(1, int(W * s * math.sqrt(ar))))

    # semantic-guided sampling (SAMCA extension)
    prob = (mask.flatten() ** 1.5)
    prob = prob / (prob.sum() + 1e-12)

    idx = torch.multinomial(prob, 1).item()
    cy, cx = divmod(idx, W)

    top = min(max(0, cy - crop_h // 2), H - crop_h)
    left = min(max(0, cx - crop_w // 2), W - crop_w)

    return top, left, crop_h, crop_w


def crop_and_resize(x, box, size):
    if x.dim() == 3:
        x = x.unsqueeze(0)

    t, l, h, w = box
    x = x[..., t:t+h, l:l+w]

    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

# ===================== 4. Attack =====================

def attack_sample(src_path, tgt_path, mask_path, extractor, cfg):

    tf = transforms.Compose([
        transforms.Resize((cfg.input_res, cfg.input_res)),
        transforms.ToTensor()
    ])

    src = tf(Image.open(src_path).convert("RGB")).to(cfg.device)
    tgt = tf(Image.open(tgt_path).convert("RGB")).to(cfg.device)

    # semantic mask (224 → 336 aligned externally)
    mask = np.load(mask_path).astype(np.float32)
    mask = torch.tensor(mask).to(cfg.device)

    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(cfg.input_res, cfg.input_res),
        mode="bilinear"
    )[0, 0]

    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    x_adv = src.clone()
    velocity = torch.zeros_like(x_adv)

    H = W = cfg.input_res

    for _ in range(1):  # FGSM single-step

        x_adv.requires_grad_(True)

        losses = []
        weights = []

        for _ in range(cfg.k_crops):

            box = sample_crop_box(H, W, mask)

            x_c = crop_and_resize(x_adv, box, (cfg.input_res, cfg.input_res))
            t_c = crop_and_resize(tgt, box, (cfg.input_res, cfg.input_res))

            adv_feat = extractor(x_c)
            tgt_feat = extractor(t_c)

            loss = 1 - F.cosine_similarity(adv_feat, tgt_feat).mean()

            losses.append(loss)

            t, l, h, w = box
            wgt = mask[t:t+h, l:l+w].mean().item()
            weights.append(wgt ** cfg.mask_gamma)

        w = torch.tensor(weights, device=cfg.device)
        w = w / (w.sum() + 1e-12)

        loss = torch.stack(losses).dot(w)
        loss.backward()

        with torch.no_grad():

            grad = x_adv.grad
            grad = grad / (grad.abs().mean() + 1e-8)

            velocity = cfg.momentum * velocity + grad

            step_mask = mask ** cfg.mask_gamma

            x_adv = x_adv - cfg.alpha * velocity.sign() * step_mask

            delta = torch.clamp(x_adv - src, -cfg.epsilon, cfg.epsilon)
            x_adv = torch.clamp(src + delta, 0, 1)

        x_adv.grad = None

    return x_adv

# ===================== 5. Main =====================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--tgt_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--resolution", type=int, default=336)

    parser.add_argument("--models", nargs="+", default=[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    ])

    args = parser.parse_args()
    cfg = Config(args)

    os.makedirs(cfg.out_dir, exist_ok=True)

    extractor = EnsembleCLIPExtractor(cfg.ensemble_models, cfg.device)

    src_files = sorted(os.listdir(cfg.src_dir))
    tgt_files = sorted(os.listdir(cfg.tgt_dir))
    mask_files = sorted(os.listdir(cfg.mask_dir))

    n = min(len(src_files), len(tgt_files), len(mask_files))

    print(f"[INFO] Running SAMCA on {n} samples...")

    for i in tqdm(range(n)):

        try:
            adv = attack_sample(
                os.path.join(cfg.src_dir, src_files[i]),
                os.path.join(cfg.tgt_dir, tgt_files[i]),
                os.path.join(cfg.mask_dir, mask_files[i]),
                extractor,
                cfg
            )

            save_path = os.path.join(cfg.out_dir, f"adv_{i:04d}.png")
            transforms.ToPILImage()(adv.cpu()).save(save_path)

        except Exception as e:
            print(f"[Skip] {i}: {e}")


if __name__ == "__main__":
    main()v
