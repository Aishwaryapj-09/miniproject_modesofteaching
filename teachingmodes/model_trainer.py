"""
model_trainer.py
=================
Reads videos directly from your 3 folders.
Extracts frames, trains MobileNetV2, saves best_model.pth.

Requirements:
  pip install torch torchvision opencv-python scikit-learn tqdm pandas

CHANGE ONLY THE 4 PATHS MARKED WITH STAR BELOW, THEN RUN:
  python model_trainer.py
"""

import os, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
import cv2, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
#  STAR  CHANGE THESE 4 PATHS ONLY
# ══════════════════════════════════════════════════════════════
BOARD_ONLY_FOLDER = r"C:\Users\LENOVO\Desktop\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\datasets\boardonly"
PPT_ONLY_FOLDER   = r"C:\Users\LENOVO\Desktop\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\datasets\pptonly"
BOTH_FOLDER       = r"C:\Users\LENOVO\Desktop\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\datasets\boardandppt"
MODEL_DIR         = r"C:\Users\LENOVO\Desktop\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\teachingmodes"
# ══════════════════════════════════════════════════════════════

FRAMES_PER_SEC = 3
IMG_SIZE       = 224
BATCH_SIZE     = 8
EPOCHS         = 40
LR             = 1e-4
NUM_CLASSES    = 3
CLASS_NAMES    = ["boardonly", "pptonly", "boardandppt"]
EARLY_STOP     = 10
MAX_PER_CLASS  = 500
HASH_THRESH    = 8
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTS     = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# ── Extract frames from videos in a folder ────────────────────────────────────

def extract(folder_path, label, cls_name, tmp_dir):
    folder = Path(folder_path)
    out    = Path(tmp_dir) / cls_name
    out.mkdir(parents=True, exist_ok=True)
    rows   = []

    if not folder.exists():
        print(f"  [NOT FOUND] {folder_path}")
        return rows

    videos = [f for ext in VIDEO_EXTS
               for f in list(folder.rglob(f"*{ext}")) +
                        list(folder.rglob(f"*{ext.upper()}"))]

    if not videos:
        print(f"  [NO VIDEOS] {folder_path}")
        return rows

    print(f"  {cls_name}: {len(videos)} videos")

    for vid in videos:
        cap     = cv2.VideoCapture(str(vid))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step    = max(1, int(src_fps / FRAMES_PER_SEC))
        n = saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if n % step == 0:
                dest = out / f"{cls_name}__{vid.stem}__f{n:06d}.jpg"
                cv2.imwrite(str(dest), frame)
                rows.append({"path": str(dest), "label": label,
                             "class_name": cls_name})
                saved += 1
            n += 1
        cap.release()
        print(f"    {vid.name}: {saved} frames")

    return rows


# ── Perceptual hash deduplication ─────────────────────────────────────────────

def phash(img, size=16):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    bits = (r > r.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming(a, b):
    x, d = a ^ b, 0
    while x:
        d += x & 1
        x >>= 1
    return d

def dedup(df_cls, thresh):
    paths = df_cls["path"].tolist()
    print(f"    Hashing {len(paths)} frames...", end=" ", flush=True)
    hashes = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        hashes.append(phash(img) if img is not None else None)
    print("done")
    kept, kept_h = [], []
    for i, h in enumerate(hashes):
        if h is None: continue
        if not any(hamming(h, kh) <= thresh for kh in kept_h):
            kept.append(i); kept_h.append(h)
    print(f"    {len(paths)} -> {len(kept)} (removed {len(paths)-len(kept)} duplicates)")
    return df_cls.iloc[kept].reset_index(drop=True)

def diverse_sample(df_cls, n):
    if len(df_cls) <= n: return df_cls
    idx = np.linspace(0, len(df_cls)-1, n, dtype=int)
    print(f"    Sampled {n} from {len(df_cls)}")
    return df_cls.iloc[idx].reset_index(drop=True)


# ── Transforms ────────────────────────────────────────────────────────────────

train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02,0.1)),
])
val_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class TeachingDataset(Dataset):
    def __init__(self, df, tfm=None):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = cv2.imdecode(np.fromfile(row["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        label = int(row["label"])
        if img is None: img = np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
        else:           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (self.tfm(img) if self.tfm else transforms.ToTensor()(img)), label


# ── Model ─────────────────────────────────────────────────────────────────────

class TeachingClassifier(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        base          = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1280,512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512,128),  nn.ReLU(),
            nn.Linear(128, n),
        )
    def forward(self, x):
        return self.head(self.pool(self.features(x)).view(x.size(0),-1))


# ── Epoch runner ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, opt, crit, training):
    model.train() if training else model.eval()
    ls = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc="  train" if training else "  val  ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if training: opt.zero_grad()
            out  = model(imgs)
            loss = crit(out, labels)
            if training: loss.backward(); opt.step()
            ls      += loss.item() * imgs.size(0)
            correct += (out.argmax(1)==labels).sum().item()
            total   += imgs.size(0)
    return ls/max(total,1), correct/max(total,1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_dir = os.path.join(MODEL_DIR, "tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"\nDevice : {DEVICE}")
    print(f"\n── Step 1: Extracting frames from videos ──────────────────")

    all_rows = []
    for folder, label, cls in [
        (BOARD_ONLY_FOLDER, 0, "boardonly"),
        (PPT_ONLY_FOLDER,   1, "pptonly"),
        (BOTH_FOLDER,       2, "boardandppt"),
    ]:
        rows = extract(folder, label, cls, tmp_dir)
        all_rows.extend(rows)

    if not all_rows:
        print("\nERROR: No frames extracted.")
        print("Check your 4 folder paths at the top of this file.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\n  Total frames: {len(df)}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {n}: {(df['label']==i).sum()}")

    print(f"\n── Step 2: Deduplication ──────────────────────────────────")
    clean = []
    for label in [0,1,2]:
        part = df[df["label"]==label].copy()
        if len(part) == 0:
            print(f"  WARNING: 0 frames for {CLASS_NAMES[label]}")
            print(f"  Check the folder path and make sure videos are inside.")
            continue
        print(f"\n  {CLASS_NAMES[label]} ({len(part)} frames):")
        part = dedup(part, HASH_THRESH)
        part = diverse_sample(part, MAX_PER_CLASS)
        clean.append(part)

    if not clean:
        print("ERROR: No frames after dedup.")
        return

    df = pd.concat(clean).reset_index(drop=True)
    print(f"\n  After dedup:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {n}: {(df['label']==i).sum()}")

    if len(df) < 30:
        print("\nERROR: Too few frames. Add more videos and rerun.")
        return

    print(f"\n── Step 3: Train / val split (80/20) ──────────────────────")
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"  Train: {len(train_df)}   Val: {len(val_df)}")

    train_labels   = train_df["label"].tolist()
    class_counts   = Counter(train_labels)
    sample_weights = [1.0/class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(train_labels),
                                    replacement=True)

    train_ld = DataLoader(TeachingDataset(train_df, train_tfm),
                          BATCH_SIZE, sampler=sampler, num_workers=0)
    val_ld   = DataLoader(TeachingDataset(val_df, val_tfm),
                          BATCH_SIZE, shuffle=False, num_workers=0)

    weights = torch.tensor(
        [len(train_df)/(NUM_CLASSES*class_counts.get(i,1)) for i in range(NUM_CLASSES)],
        dtype=torch.float).to(DEVICE)
    print(f"  Class weights: {weights.cpu().numpy().round(3)}")

    print(f"\n── Step 4: Training ───────────────────────────────────────")
    model = TeachingClassifier(NUM_CLASSES).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=weights)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best = no_improv = 0
    print(f"  Up to {EPOCHS} epochs | early stop after {EARLY_STOP} no-improve\n")

    for epoch in range(1, EPOCHS+1):
        tl, ta = run_epoch(model, train_ld, opt, crit, True)
        vl, va = run_epoch(model, val_ld,   opt, crit, False)
        sched.step()
        line = (f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train acc={ta:.3f} loss={tl:.4f}  "
                f"val acc={va:.3f} loss={vl:.4f}")
        if va > best:
            best = va; no_improv = 0
            torch.save({
                "epoch":epoch, "model_state":model.state_dict(),
                "val_acc":va, "class_names":CLASS_NAMES,
                "img_size":IMG_SIZE, "num_classes":NUM_CLASSES,
            }, os.path.join(MODEL_DIR, "best_model.pth"))
            line += "  <- best saved"
        else:
            no_improv += 1
            line += f"  (no improve {no_improv}/{EARLY_STOP})"
        print(line)
        if no_improv >= EARLY_STOP:
            print(f"\n  Early stop at epoch {epoch}.")
            break

    torch.save({
        "epoch":epoch, "model_state":model.state_dict(),
        "class_names":CLASS_NAMES, "img_size":IMG_SIZE, "num_classes":NUM_CLASSES,
    }, os.path.join(MODEL_DIR, "final_model.pth"))

    print(f"\n{'='*55}")
    print(f"  Best val accuracy : {best:.3f}")
    print(f"  Model saved to    : {MODEL_DIR}")
    print(f"{'='*55}\n")

    if best >= 0.85:   print("  Excellent. Model is ready.\n")
    elif best >= 0.70: print("  Good. Model is usable. Add more videos to improve.\n")
    else:              print("  Low accuracy. Add more diverse videos and retrain.\n")


if __name__ == "__main__":
    main()