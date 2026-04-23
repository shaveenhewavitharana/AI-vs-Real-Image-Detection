import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
TEST_DIR = "data/test"
MODEL_PATH = "models/best_model.keras"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model = load_model(MODEL_PATH)

pred_probs = model.predict(test_generator)
preds = (pred_probs > 0.5).astype(int).flatten()

true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(true_labels, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("\nClassification Report:\n")
print(classification_report(true_labels, preds, target_names=class_names, zero_division=0))