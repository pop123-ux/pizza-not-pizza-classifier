"""Train a CNN to classify images as pizza vs. not-pizza.

Dataset: "Pizza or Not Pizza?" (Kaggle, carlosrunner/pizza-not-pizza),
a balanced subset of Food-101 with 983 pizza and 983 non-pizza images.

Expected layout (default; override with --data-dir):
    archive/pizza_not_pizza/
        pizza/
        not_pizza/

Usage:
    python train.py --epochs 15 --batch-size 32
"""
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (160, 160)


def build_datasets(data_dir: Path, img_size, batch_size, val_split, seed):
    """Load images from a directory of class subfolders into train/val datasets."""
    common = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        validation_split=val_split,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)

    class_names = train_ds.class_names
    print("Classes:", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(img_size):
    """A small CNN with on-the-fly augmentation and rescaling baked in."""
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=img_size + (3,)),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="pizza_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("archive/pizza_not_pizza"),
        help="Directory containing 'pizza/' and 'not_pizza/' subfolders.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("pizza_classifier.keras"),
        help="Where to save the trained model.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(
            f"Data directory not found: {args.data_dir}\n"
            "Download the dataset from "
            "https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza "
            "and extract it so that 'pizza/' and 'not_pizza/' live inside the data dir."
        )

    train_ds, val_ds, class_names = build_datasets(
        args.data_dir, IMG_SIZE, args.batch_size, args.val_split, args.seed
    )

    model = build_model(IMG_SIZE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    loss, acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {acc:.3f}  |  loss: {loss:.3f}")

    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
