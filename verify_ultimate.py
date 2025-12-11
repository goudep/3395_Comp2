
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# MOCK DATA if files don't exist, otherwise use real
DATA_DIR = Path('data')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

SEED = 42
seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data():
    # Attempt to load real data
    try:
        with open(DATA_DIR / 'train_data.pkl', 'rb') as f:
            train = pickle.load(f)
        with open(DATA_DIR / 'test_data.pkl', 'rb') as f:
            test = pickle.load(f)
        return train, test
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Creating MOCK data for testing")
        # Mock
        train = {
            'images': np.random.randint(0, 255, (100, 28, 28, 3), dtype=np.uint8),
            'labels': np.random.randint(0, 5, (100, 1), dtype=np.int32)
        }
        test = {
            'images': np.random.randint(0, 255, (20, 28, 28, 3), dtype=np.uint8)
        }
        return train, test

train_data, test_data = load_data()

X_train_raw = train_data['images']
y_train = train_data['labels'].flatten()
X_test_raw = test_data['images']

if X_train_raw.max() <= 1.0:
    X_train_raw = (X_train_raw * 255).astype(np.uint8)
    X_test_raw = (X_test_raw * 255).astype(np.uint8)
else:
    X_train_raw = X_train_raw.astype(np.uint8)
    X_test_raw = X_test_raw.astype(np.uint8)

# FIX: Context labels to long
y_train = y_train.astype(np.int64)

class IFT3395Dataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_arr = self.images[idx]
        img = Image.fromarray(img_arr)
        
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            return img, self.labels[idx]
        return img

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def conv_bn(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CustomResNet9(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.prep = conv_bn(3, 64)
        self.layer1_conv = conv_bn(64, 128, pool=True)
        self.layer1_res = nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))
        self.layer2_conv = conv_bn(128, 256, pool=True)
        self.layer3_conv = conv_bn(256, 512, pool=True)
        self.layer3_res = nn.Sequential(conv_bn(512, 512), conv_bn(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        out = self.prep(x)
        out = self.layer1_conv(out)
        out = out + self.layer1_res(out)
        out = self.layer2_conv(out)
        out = self.layer3_conv(out)
        out = out + self.layer3_res(out)
        out = self.classifier(out)
        return out

def get_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    # FIX: Explicit double tensor
    return WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights))

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return 0.0, 0.0

print("Starting Loop Check...")
N_FOLDS = 2 # Reduced for check
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    print(f"Fold {fold}")
    X_tr, y_tr = X_train_raw[train_idx], y_train[train_idx]
    
    sampler = get_sampler(y_tr)
    train_ds = IFT3395Dataset(X_tr, y_tr, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=0)
    
    model = CustomResNet9(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    print("Fold done")
    break # Only one fold needed to verify

print("Verification Complete")
