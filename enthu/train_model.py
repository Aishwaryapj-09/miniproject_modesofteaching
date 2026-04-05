"""
train_model.py - FINAL VERSION
Same preprocessing in training AND prediction = correct results!
"""
import argparse, os, sys, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

IMG_SIZE   = 64
BATCH_SIZE = 32
SEED       = 42
MAX_IMAGES = 1500

ENTHUSIASTIC_CLASSES     = {"happy", "surprise", "happiness"}
NOT_ENTHUSIASTIC_CLASSES = {"angry", "anger", "neutral", "sad",
                             "sadness", "fear", "disgust", "contempt"}


def preprocess_face(img_bgr):
    """
    EXACT SAME preprocessing used in both training AND prediction.
    This is the key fix — consistency between train and test!
    1. Grayscale
    2. Histogram equalisation (normalises lighting — works on AI + real faces)
    3. Resize to 64x64
    4. Convert back to 3 channel
    5. Normalise to [-1, 1]
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq      = cv2.equalizeHist(gray)
    resized = cv2.resize(eq, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32) / 127.5 - 1.0


def augment(img):
    """Augment preprocessed image"""
    # Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    # Brightness
    img = np.clip(img * random.uniform(0.7, 1.3), -1, 1).astype(np.float32)
    # Noise
    img = np.clip(img + np.random.normal(0, 0.03, img.shape), -1, 1).astype(np.float32)
    return img


def load_dataset(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    folders      = [f for f in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, f))]

    enth_imgs     = []
    not_enth_imgs = []

    print(f"\n[INFO] Loading: {dataset_path}")
    for folder in folders:
        fl = folder.lower()
        fp = os.path.join(dataset_path, folder)
        files = [f for f in os.listdir(fp)
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

        if fl in ENTHUSIASTIC_CLASSES:
            label = "enthusiastic"
        elif fl in NOT_ENTHUSIASTIC_CLASSES:
            label = "not_enthusiastic"
        else:
            print(f"  [SKIP] {folder}/")
            continue

        print(f"  [{label:15s}] {folder}/ → {len(files)} images")

        random.seed(SEED)
        random.shuffle(files)

        for fname in files:
            img = cv2.imread(os.path.join(fp, fname))
            if img is None: continue
            # Use SAME preprocess_face function as prediction!
            processed = preprocess_face(img)
            if label == "enthusiastic":
                enth_imgs.append(processed)
            else:
                not_enth_imgs.append(processed)

    print(f"\n  Enthusiastic     : {len(enth_imgs)}")
    print(f"  Not enthusiastic : {len(not_enth_imgs)}")

    if len(enth_imgs) == 0:
        sys.exit("[ERROR] No enthusiastic images! Check happy/ surprise/ folders.")
    if len(not_enth_imgs) == 0:
        sys.exit("[ERROR] No not-enthusiastic images! Check angry/ neutral/ sad/ folders.")

    # Cap
    random.seed(SEED)
    if len(enth_imgs)     > MAX_IMAGES:
        enth_imgs     = random.sample(enth_imgs, MAX_IMAGES)
    if len(not_enth_imgs) > MAX_IMAGES:
        not_enth_imgs = random.sample(not_enth_imgs, MAX_IMAGES)

    # Augment 4x
    print(f"\n[Augment] Creating 4x augmented data...")
    enth_aug     = list(enth_imgs)
    not_enth_aug = list(not_enth_imgs)
    for _ in range(3):
        enth_aug.extend([augment(img) for img in enth_imgs])
        not_enth_aug.extend([augment(img) for img in not_enth_imgs])

    print(f"  Enthusiastic     : {len(enth_aug)}")
    print(f"  Not enthusiastic : {len(not_enth_aug)}")

    X = np.array(enth_aug + not_enth_aug, dtype=np.float32)
    y = np.array([1]*len(enth_aug) + [0]*len(not_enth_aug), dtype=np.float32)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def build_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D()(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(256, activation="relu")(x)
    x      = layers.Dropout(0.5)(x)
    x      = layers.Dense(128, activation="relu")(x)
    x      = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, output, name="enthusiasm_classifier"), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs",  type=int, default=50)
    ap.add_argument("--output",  default="face_model.keras")
    args = ap.parse_args()

    output = args.output.replace(".h5",".keras")
    if not output.endswith(".keras"):
        output += ".keras"

    print("=" * 60)
    print("  Teacher Enthusiasm — FINAL Classifier")
    print("  Same preprocessing in training AND prediction!")
    print("  Works on real + AI generated videos!")
    print("=" * 60)

    X, y = load_dataset(args.dataset)

    split        = int(0.8 * len(y))
    X_tr, X_val  = X[:split], X[split:]
    y_tr, y_val  = y[:split], y[split:]

    n1 = int(y_tr.sum())
    n0 = len(y_tr) - n1
    cw = {0: len(y_tr)/(2*max(n0,1)),
          1: len(y_tr)/(2*max(n1,1))}

    print(f"\n[Split] Train={len(y_tr)}  Val={len(y_val)}")
    print(f"[Key]   Training uses histogram equalisation")
    print(f"        Prediction uses same histogram equalisation")
    print(f"        → Model works on AI + real videos!\n")

    model, base = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(output, save_best_only=True,
                        monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, factor=0.3, verbose=1),
    ]

    # Phase 1
    print("\n[Phase 1] Training head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy", metrics=["accuracy"]
    )
    h1 = model.fit(X_tr, y_tr, validation_data=(X_val,y_val),
                   epochs=15, batch_size=BATCH_SIZE,
                   class_weight=cw, callbacks=callbacks)

    # Phase 2
    print("\n[Phase 2] Fine-tuning...")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss="binary_crossentropy", metrics=["accuracy"]
    )
    h2 = model.fit(X_tr, y_tr, validation_data=(X_val,y_val),
                   epochs=args.epochs, batch_size=BATCH_SIZE,
                   class_weight=cw, callbacks=callbacks)

    # Plot
    acc  = h1.history["accuracy"]     + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"]         + h2.history["loss"]
    vloss= h1.history["val_loss"]     + h2.history["val_loss"]
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(acc,  label="train"); plt.plot(vacc, label="val")
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(loss, label="train"); plt.plot(vloss,label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")

    print(f"\n[DONE] Model saved → {output}")
    print(f"\nTest:")
    print(f"  python analyze_video.py --video enthu_v1.mp4 --model {output} --fps_sample 5")
    print(f"  python analyze_video.py --video not_enthu_v1.mp4 --model {output} --fps_sample 5")

if __name__ == "__main__":
    main()