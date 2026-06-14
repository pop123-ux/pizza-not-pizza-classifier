"""Run a trained pizza/not-pizza model on one or more image files.

Usage:
    python predict.py path/to/image.jpg [more.jpg ...]
    python predict.py --model pizza_classifier.keras img.jpg
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


IMG_SIZE = (160, 160)


def load_image(path: Path, img_size):
    img = tf.keras.utils.load_img(path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s) to classify.")
    parser.add_argument("--model", type=Path, default=Path("pizza_classifier.keras"))
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability above which an image is labelled 'pizza'.",
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    for path in args.images:
        prob = float(model.predict(load_image(path, IMG_SIZE), verbose=0)[0][0])
        # image_dataset_from_directory sorts classes alphabetically:
        # not_pizza -> 0, pizza -> 1, so a high sigmoid output means "pizza".
        label = "pizza" if prob >= args.threshold else "not_pizza"
        confidence = prob if label == "pizza" else 1 - prob
        print(f"{path.name}: {label} ({confidence:.1%})")


if __name__ == "__main__":
    main()
