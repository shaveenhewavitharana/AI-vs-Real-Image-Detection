import os
import shutil
import random
from pathlib import Path

random.seed(42)

AI_SOURCE = "Ai_generated_dataset"
REAL_SOURCE = "real_dataset"
BASE_DIR = "data"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def clear_data():
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

def create_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["ai", "real"]:
            os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

def get_images(folder):
    return [p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

def split(files):
    random.shuffle(files)
    n = len(files)
    train = files[:int(0.7 * n)]
    val = files[int(0.7 * n):int(0.85 * n)]
    test = files[int(0.85 * n):]
    return train, val, test

def copy(files, split_name, cls):
    dest = os.path.join(BASE_DIR, split_name, cls)
    for i, f in enumerate(files):
        new_name = f"{cls}_{split_name}_{i:05d}{f.suffix.lower()}"
        shutil.copy2(f, os.path.join(dest, new_name))

if __name__ == "__main__":
    clear_data()
    create_dirs()

    ai_files = get_images(AI_SOURCE)
    real_files = get_images(REAL_SOURCE)

    min_size = min(len(ai_files), len(real_files))
    ai_files = random.sample(ai_files, min_size)
    real_files = random.sample(real_files, min_size)

    print(f"Balanced dataset size per class: {min_size}")

    ai_train, ai_val, ai_test = split(ai_files)
    real_train, real_val, real_test = split(real_files)

    copy(ai_train, "train", "ai")
    copy(ai_val, "val", "ai")
    copy(ai_test, "test", "ai")

    copy(real_train, "train", "real")
    copy(real_val, "val", "real")
    copy(real_test, "test", "real")

    print("Dataset balanced and prepared successfully.")