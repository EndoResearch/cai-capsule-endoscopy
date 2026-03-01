#!/usr/bin/env python
# coding: utf-8

# CUDA_VISIBLE_DEVICES=1 nohup python features.py > features.log 2>&1 &

import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2


# --------------------------------------------------
# 1. Argument Parsing
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CNN features using a pretrained ResNet50 backbone.")
    parser.add_argument("--weights_path",  type=str, required=True,
                        help="Path to the pretrained model weights (.pth file).")
    parser.add_argument("--output_dir",    type=str, required=True,
                        help="Base directory where extracted features will be saved.")
    parser.add_argument("--dataframe",     type=str, required=True,
                        help="Path to the CSV file with patient metadata.")
    parser.add_argument("--data_path",     type=str, required=True,
                        help="Root directory containing the image data.")
    parser.add_argument("--batch_size",    type=int, default=64,
                        help="Batch size for the DataLoader (default: 64).")
    parser.add_argument("--num_workers",   type=int, default=8,
                        help="Number of DataLoader worker processes (default: 8).")
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_PATH   = args.weights_path
OUTPUT_BASE_DIR = args.output_dir
DATAFRAME_PATIENTS = args.dataframe
DATA_PATH      = args.data_path
BATCH_SIZE     = args.batch_size
NUM_WORKERS    = args.num_workers


# --------------------------------------------------
# 2. Robust Weight Loading Function
# --------------------------------------------------

def load_backbone_weights(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}")

    print(f"Loading weights from: {os.path.basename(path)}")

    # weights_only=False is required to allow loading numpy scalars
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    found_key = None

    # Priority order for nested checkpoint structures (e.g., DINO)
    if "student" in checkpoint:
        state_dict = checkpoint["student"]
        found_key = "student"
    elif "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
        found_key = "teacher"
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        found_key = "state_dict"
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        found_key = "model"
    else:
        state_dict = checkpoint
        found_key = "flat structure (direct)"

    print(f"Detected structure: weights loaded from key '{found_key}'")

    # Remove common key prefixes
    clean_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        clean_state_dict[k] = v

    # strict=False is necessary to allow partial loading (e.g., missing FC layer)
    msg = model.load_state_dict(clean_state_dict, strict=False)

    print(f"Load report: {len(msg.missing_keys)} missing keys (expected if only FC is absent).")

    if 'conv1.weight' in msg.missing_keys:
        raise RuntimeError("CRITICAL ERROR: 'conv1' layer was not loaded. Check checkpoint prefixes.")

    print("Weights loaded successfully.")


# --------------------------------------------------
# 3. Feature Extraction Function
# --------------------------------------------------

@torch.no_grad()
def extract_cnn_features(model, x):
    """Executes forward pass up to Average Pooling, bypassing the FC layer."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)  # Result: [Batch, 2048]
    return x


# --------------------------------------------------
# 4. Dataset Definition
# --------------------------------------------------

class ImageFeatureDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_name"]
        pt_path  = self.df.loc[idx, "pt_name"]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, pt_path


# --------------------------------------------------
# 5. Transforms
# --------------------------------------------------

transform = T.Compose([
    T.Resize((224, 224), interpolation=Image.LANCZOS),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --------------------------------------------------
# 6. Model Initialization
# --------------------------------------------------

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

df = pd.read_csv(DATAFRAME_PATIENTS, index_col=0)
df.reset_index(inplace=True, drop=True)

resnet = resnet50()
resnet.fc = nn.Identity()
load_backbone_weights(resnet, WEIGHTS_PATH)
resnet.to(device)
resnet.eval()

for p in resnet.parameters():
    p.requires_grad = False


# --------------------------------------------------
# 7. Main Extraction Loop
# --------------------------------------------------

print(f"Starting feature extraction for {len(df['patient'].unique())} patients...")

for num_patient in tqdm(df["patient"].unique(), desc="Patients"):
    df_patient = df[df["patient"] == num_patient].copy()

    path_save_feat = os.path.join(OUTPUT_BASE_DIR, str(num_patient))
    os.makedirs(path_save_feat, exist_ok=True)

    df_patient['image_name'] = df_patient['frame'].apply(
        lambda x: os.path.join(DATA_PATH, num_patient, f"image-{int(x):05d}.png")
    )
    df_patient['pt_name'] = df_patient['frame'].apply(
        lambda x: os.path.join(path_save_feat, f"image-{int(x):05d}.pt")
    )

    # Quick validation: check only the first image to avoid overhead
    first_img = df_patient['image_name'].iloc[0]
    if not os.path.exists(first_img):
        # Fallback: try lowercase extension
        df_patient['image_name'] = df_patient['image_name'].str.replace(".PNG", ".png")
        if not os.path.exists(df_patient['image_name'].iloc[0]):
            print(f"Critical error: images not found at {os.path.dirname(first_img)}")
            print(f"  Searched: {first_img}")
            continue

    dataset = ImageFeatureDataset(df_patient, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    with torch.no_grad():
        for images, save_paths in tqdm(loader, desc=f"P-{num_patient}", leave=False):
            images = images.to(device)

            with torch.cuda.amp.autocast():
                features = extract_cnn_features(resnet, images)  # (B, 2048)

            features = features.cpu()

            for i, path in enumerate(save_paths):
                torch.save(features[i].clone(), path)

print("\nExtraction completed successfully.")