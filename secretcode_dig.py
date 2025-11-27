import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_FILE = "fashion_classifier.h5"
CLASS_NAMES_FILE = "class_names.txt"
TEST_DIR = "test_images"

OUTPUT_FLAG = "Flag_image.png"
OUTPUT_JSON = "flag_data.json"
OUTPUT_SECRET = "secret_message.txt"


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


def fgsm_attack(model, x, label, eps=0.15):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        y_true = tf.constant([label], dtype=tf.int32)
        loss = loss_fn(y_true, preds)

    grad = tape.gradient(loss, x)
    adv = x + eps * tf.sign(grad)
    adv = tf.clip_by_value(adv, 0, 1)
    return adv


def detect_flag(results):
    freq = {}
    for r in results:
        l = r["adv_label"]
        freq[l] = freq.get(l, 0) + 1

    # rarest adversarial label
    rare = min(freq, key=lambda x: freq[x])

    for r in results:
        if r["adv_label"] == rare:
            return r, rare

    return None, None


def main():
    print("[+] Loading model...")
    model = load_model(MODEL_FILE)

    with open(CLASS_NAMES_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    _, H, W, _ = model.input_shape
    size = (W, H)

    results = []

    print("\n[+] Running FGSM attack...")
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

    chosen, rare_label = detect_flag(results)

    if chosen:
        print(f"\n🎯 FLAG IMAGE FOUND: {chosen['file']}")

        # Save flag image
        img = Image.open(os.path.join(TEST_DIR, chosen["file"]))
        img.save(OUTPUT_FLAG)

        # Build secret flag string based on chosen image
        base_name = os.path.splitext(chosen["file"])[0]
        secret = f"FLAG{{{base_name}_unique_adversarial_behavior}}"

        # Save secret to txt
        with open(OUTPUT_SECRET, "w") as f:
            f.write(secret)

        print(f"\n🔐 SECRET MESSAGE: {secret}")

        # Save JSON
        output = {
            "model": MODEL_FILE,
            "attack_method": "FGSM eps=0.15",
            "rare_adv_label": rare_label,
            "flag_image": chosen["file"],
            "flag_string": secret,
            "results": results
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(output, f, indent=4)

        print(f"\n💾 Saved {OUTPUT_JSON}")
        print(f"💾 Saved {OUTPUT_FLAG}")
        print(f"💾 Saved {OUTPUT_SECRET}")
    else:
        print("No unique adversarial output found.")


if __name__ == "__main__":
    main()
