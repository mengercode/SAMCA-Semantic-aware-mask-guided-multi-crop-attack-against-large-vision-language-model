# generate_mask_npy.py

import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

# ===================== 1. Load DINOv2 =====================
def load_local_model(hf_dir, device):
    from transformers import Dinov2Model, AutoImageProcessor, AutoConfig

    processor = AutoImageProcessor.from_pretrained(hf_dir, local_files_only=True)

    try:
        model = Dinov2Model.from_pretrained(
            hf_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
    except OSError:
        cfg = AutoConfig.from_pretrained(hf_dir, local_files_only=True)
        model = Dinov2Model(cfg)

        if os.path.exists(os.path.join(hf_dir, "model.safetensors")):
            from safetensors.torch import load_file
            sd = load_file(os.path.join(hf_dir, "model.safetensors"))
        else:
            sd = torch.load(os.path.join(hf_dir, "pytorch_model.bin"), map_location="cpu")

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        model.load_state_dict(sd, strict=False)

    return processor, model.to(device).eval()


# ===================== 2. Attention Extraction =====================
@torch.no_grad()
def get_attn_stack(model, processor, pil_img, take_last_k, device):
    """
    Extract multi-layer attention maps from DINOv2.
    """
    inputs = processor(images=pil_img, return_tensors="pt")
    px = inputs["pixel_values"].to(device)

    out = model(pixel_values=px, output_attentions=True)
    A_list = out.attentions[-take_last_k:]

    return torch.stack(A_list, dim=0)


# ===================== 3. Attention Rollout =====================
def attention_rollout(A, up=(224, 224)):
    """
    Standard Attention Rollout (Abnar & Zuidema, 2020).

    Converts multi-layer attention into spatial importance map.
    """

    L, B, H, N, _ = A.shape

    # average over heads
    A_avg = A.mean(dim=2)  # (L, B, N, N)

    eye = torch.eye(N, device=A.device).unsqueeze(0)
    R = eye.clone()

    for l in range(L):
        a = A_avg[l]          # (B, N, N)
        a = a + eye           # residual connection
        a = a / a.sum(dim=-1, keepdim=True)  # normalization
        R = a @ R

    # CLS token attention to patches
    cls_attn = R[0, 0, 1:]
    n = int(cls_attn.shape[0] ** 0.5)

    importance = cls_attn.reshape(1, 1, n, n)

    # upscale to image space
    importance = F.interpolate(
        importance,
        size=up,
        mode="bilinear",
        align_corners=False
    )

    importance = importance.squeeze().cpu().numpy()

    # normalize to [0,1]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    return importance 


# ===================== 4. Mask Construction =====================
def make_soft_mask(importance, sparsity, beta_floor=0.05):
    """
    Construct soft semantic mask.

    Steps:
    1. Adaptive threshold via quantile (robust across images)
    2. Morphological dilation for spatial continuity
    3. Soft background retention (beta_floor)
    """

    # adaptive threshold
    theta_s = np.quantile(importance, sparsity)
    M_bin = (importance >= theta_s).astype(np.float32)

    # morphological smoothing
    kernel = np.ones((3, 3), np.uint8)
    M_bin = cv2.dilate(M_bin, kernel, iterations=1)

    # soft mask
    M = M_bin + (1 - M_bin) * beta_floor

    return M.astype(np.float32)


# ===================== 5. Main Pipeline =====================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--sparsity", type=float, default=0.5,
                    help="Foreground ratio (quantile threshold)")

    ap.add_argument("--beta_floor", type=float, default=0.05,
                    help="Background retention factor")

    ap.add_argument("--take_last_k", type=int, default=6,
                    help="Number of transformer layers used")

    ap.add_argument("--out_res", type=int, default=224,
                    help="Mask resolution")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading model from: {args.hf_dir}")
    processor, model = load_local_model(args.hf_dir, args.device)

    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Mask resolution: {args.out_res}×{args.out_res}")
    print(f"[INFO] Sparsity: {args.sparsity}")

    # collect images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files += glob.glob(os.path.join(args.image_dir, ext))
        image_files += glob.glob(os.path.join(args.image_dir, ext.upper()))

    image_files = sorted(set(image_files))

    print(f"[INFO] Total images: {len(image_files)}")

    # ===================== processing =====================
    for i, image_path in enumerate(image_files, 1):

        try:
            name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"[{i}/{len(image_files)}] Processing {name}")

            pil = Image.open(image_path).convert("RGB")

            # attention extraction
            A = get_attn_stack(model, processor, pil, args.take_last_k, args.device)

            # rollout → importance map
            importance = attention_rollout(A, up=(args.out_res, args.out_res))

            # semantic mask
            mask = make_soft_mask(importance, args.sparsity, args.beta_floor)

            # save
            out_path = os.path.join(args.out_dir, f"{name}.npy")
            np.save(out_path, mask)

            print(f"  Saved: {out_path} | shape={mask.shape}")

        except Exception as e:
            import traceback
            print(f"[ERROR] {image_path}: {e}")
            traceback.print_exc()

    print("\n[DONE] All masks generated successfully.")

if __name__ == "__main__":
    main()
