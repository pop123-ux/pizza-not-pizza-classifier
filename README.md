# Pizza / Not-Pizza — Binary Image Classification

A convolutional neural network (CNN) that classifies images as **pizza** or **not pizza**, built with TensorFlow/Keras.

The model is trained on the [Pizza or Not Pizza?](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza) dataset — a balanced subset of [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) with **983 pizza** and **983 non-pizza** images.

## Project structure

```
.
├── train.py            # Build datasets, train the CNN, evaluate, save the model
├── predict.py          # Classify new images with a trained model
├── requirements.txt
└── archive/            # Dataset (not committed — download from Kaggle)
    └── pizza_not_pizza/
        ├── pizza/
        └── not_pizza/
```

## Setup

```bash
pip install -r requirements.txt
```

### Get the data

The dataset is **not** included in this repo. Download it from Kaggle and extract it so the folder layout matches the structure above:

[kaggle.com/datasets/carlosrunner/pizza-not-pizza](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza?resource=download)

## Train

```bash
python train.py --epochs 15 --batch-size 32
```

The script:

- Loads images with `image_dataset_from_directory` and an 80/20 train/validation split.
- Applies on-the-fly augmentation (random flip, rotation, zoom) and rescaling.
- Trains a small CNN (3 conv blocks → dense head, sigmoid output).
- Uses early stopping on validation loss and saves the best model to `pizza_classifier.keras`.

Key flags: `--data-dir`, `--epochs`, `--batch-size`, `--val-split`, `--output`.

## Predict

```bash
python predict.py path/to/image.jpg
```

Outputs the predicted label and confidence, e.g.:

```
margherita.jpg: pizza (94.2%)
salad.jpg: not_pizza (88.7%)
```

## Model architecture

| Stage | Layers |
|-------|--------|
| Augmentation | RandomFlip → RandomRotation → RandomZoom |
| Preprocess | Rescaling (1/255) |
| Block 1 | Conv2D(32) → MaxPool |
| Block 2 | Conv2D(64) → MaxPool |
| Block 3 | Conv2D(128) → MaxPool |
| Head | Dropout(0.3) → Flatten → Dense(128, relu) → Dense(1, sigmoid) |

Loss: binary cross-entropy · Optimizer: Adam.

## License

MIT

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
