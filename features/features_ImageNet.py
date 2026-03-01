#!/usr/bin/env python
# coding: utf-8

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
    parser = argparse.ArgumentParser(description="Extract CNN features using an ImageNet-pretrained ResNet50 backbone.")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Base directory where extracted features will be saved.")
    parser.add_argument("--dataframe",   type=str, required=True,
                        help="Path to the Excel (.xlsx) file with patient metadata.")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Root directory containing the image data.")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Directory containing per-patient label CSV files.")
    parser.add_argument("--batch_size",  type=int, default=128,
                        help="Batch size for the DataLoader (default: 128).")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of DataLoader worker processes (default: 8).")
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_BASE_DIR    = args.output_dir
DATAFRAME_PATIENTS = args.dataframe
DATA_PATH          = args.data_path
LABELS_PATH        = args.labels_path
BATCH_SIZE         = args.batch_size
NUM_WORKERS        = args.num_workers


# --------------------------------------------------
# 2. Feature Extraction Function
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
# 3. Dataset Definition
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
# 4. Transforms
# --------------------------------------------------

transform = T.Compose([
    T.Resize((224, 224), interpolation=Image.LANCZOS),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --------------------------------------------------
# 5. Model Initialization
# --------------------------------------------------

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

df = pd.read_excel(DATAFRAME_PATIENTS, index_col=0)
df.reset_index(inplace=True, drop=True)

resnet = resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet.to(device)
resnet.eval()

for p in resnet.parameters():
    p.requires_grad = False


# --------------------------------------------------
# 6. Image Existence Validation
# --------------------------------------------------

print("Validating image availability across all patients...")
for num_patient in tqdm(df["Patient"].unique()):
    df_patient = pd.read_csv(os.path.join(LABELS_PATH, f"{num_patient}.csv"))
    for im in df_patient.index:
        image_path = os.path.join(DATA_PATH, str(num_patient), f"frame_{int(df_patient.loc[im, 'frame']):06d}.PNG")
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")


# --------------------------------------------------
# 7. Main Extraction Loop
# --------------------------------------------------

print(f"Starting feature extraction for {len(df['Patient'].unique())} patients...")

for num_patient in tqdm(df["Patient"].unique(), desc="Patients"):
    num_patient = str(num_patient)

    csv_path = os.path.join(LABELS_PATH, f"{num_patient}.csv")
    if not os.path.exists(csv_path):
        print(f"Label CSV not found for patient {num_patient}, skipping...")
        continue

    df_patient = pd.read_csv(csv_path)

    path_save_feat = os.path.join(OUTPUT_BASE_DIR, num_patient)
    os.makedirs(path_save_feat, exist_ok=True)

    df_patient['image_name'] = df_patient['frame'].apply(
        lambda x: os.path.join(DATA_PATH, num_patient, f"frame_{int(x):06d}.PNG")
    )
    df_patient['pt_name'] = df_patient['frame'].apply(
        lambda x: os.path.join(path_save_feat, f"frame_{int(x):06d}.pt")
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
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    with torch.no_grad():
        for images, save_paths in tqdm(loader, desc=f"P-{num_patient}", leave=False):
            images = images.to(device)

            if device == "cuda":
                with torch.cuda.amp.autocast():
                    features = extract_cnn_features(resnet, images)  # (B, 2048)
            else:
                features = extract_cnn_features(resnet, images)

            features = features.cpu()

            for i, path in enumerate(save_paths):
                torch.save(features[i].clone(), path)

print("\nExtraction completed successfully.")