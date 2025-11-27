import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

MODEL_FILE = "fashion_classifier.h5"
CLASS_NAMES_FILE = "class_names.txt"
TEST_DIR = "test_images"

OUTPUT_FLAG = "Flag_image.png"
OUTPUT_JSON = "flag_data.json"


# -----------------------------------------------------
# 1. TRAIN MODEL (automatically)
# -----------------------------------------------------
def train_model():
    print("\n[+] Loading Fashion-MNIST Dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Save class names
    with open(CLASS_NAMES_FILE, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # Preprocess
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    # Build model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[+] Training model (10 epochs)...")
    history = model.fit(
        x_train, y_train, epochs=10, batch_size=64,
        validation_split=0.1, verbose=1
    )

    # Evaluate
    print("\n[+] Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
    print(f"📉 Test Loss: {test_loss:.4f}")

    # Save model
    model.save(MODEL_FILE)
    print(f"\n💾 Model saved as {MODEL_FILE}")

    return model



# -----------------------------------------------------
# 2. Load utilities
# -----------------------------------------------------
def load_class_names():
    with open(CLASS_NAMES_FILE, "r") as f:
        return [x.strip() for x in f.readlines()]


def load_image(path, size):
    img = Image.open(path).convert("L")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return tf.convert_to_tensor(arr), img



def predict(model, x):
    probs = model.predict(x, verbose=0)[0]
    return int(np.argmax(probs)), probs.tolist()



# -----------------------------------------------------
# 3. FGSM ATTACK
# -----------------------------------------------------
def fgsm_attack(model, x, label, eps=0.15):

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        y_true = tf.constant([label], dtype=tf.int32)
        loss = loss_fn(y_true, preds)

    grad = tape.gradient(loss, x)
    adv_x = x + eps * tf.sign(grad)
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x



# -----------------------------------------------------
# 4. FLAG DETECTION
# -----------------------------------------------------
def detect_flag(results):
    freq = {}
    for r in results:
        l = r["adv_label"]
        freq[l] = freq.get(l, 0) + 1

    rare_label = min(freq, key=lambda x: freq[x])

    for r in results:
        if r["adv_label"] == rare_label:
            return r, rare_label

    return None, None



# -----------------------------------------------------
# 5. MAIN PIPELINE
# -----------------------------------------------------
def main():

    # TRAIN model first
    model = train_model()

    class_names = load_class_names()
    _, h, w, _ = model.input_shape
    size = (w, h)

    results = []

    print("\n[+] Running FGSM attack on test_images/...")

    for fname in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, fname)

        x, pil_img = load_image(path, size)

        orig_label, _ = predict(model, x)
        x_adv = fgsm_attack(model, x, orig_label)
        adv_label, _ = predict(model, x_adv)

        print(f"{fname}: {orig_label} -> {adv_label}")

        results.append({
            "file": fname,
            "original_label": orig_label,
            "adv_label": adv_label
        })

    # Detect flag
    chosen, rare = detect_flag(results)

    if chosen:
        print(f"\n🎉 FLAG FOUND: {chosen['file']}")
        img = Image.open(os.path.join(TEST_DIR, chosen["file"]))
        img.save(OUTPUT_FLAG)

        output_json = {
            "model": MODEL_FILE,
            "attack_method": "FGSM (eps=0.15)",
            "rare_adv_label": rare,
            "flag_image": chosen["file"],
            "data": results
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(output_json, f, indent=4)

        print(f"\n💾 Saved {OUTPUT_JSON}")
        print(f"💾 Saved {OUTPUT_FLAG}")

    else:
        print("No unique adversarial image found.")



if __name__ == "__main__":
    main()
