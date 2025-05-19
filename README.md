# Autoencoder-Based Unsupervised Clustering

This repository provides an implementation of an **unsupervised clustering framework** using **autoencoders**. The model learns compact feature representations through an autoencoder and clusters them using a deep clustering mechanism inspired by the **Deep Embedded Clustering (DEC)** approach.

## Overview

Traditional clustering algorithms often struggle with high-dimensional or unstructured data. By leveraging autoencoders, we can first learn meaningful lower-dimensional representations of data and then perform clustering in this latent space.

This implementation supports:

- Unsupervised learning (no labels required)
- Joint optimization of reconstruction and clustering objectives
- DEC-style clustering layer for soft cluster assignments
- Customizable architecture and clustering parameters

## Architecture

- **Encoder**: Maps input data to a lower-dimensional latent space
- **Decoder**: Reconstructs input from latent representation
- **Clustering Layer**: Learns soft assignments and cluster centroids

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn
- matplotlib (optional, for visualization)

Install dependencies:

```bash
pip install -r requirements.txt
python train_autoencoder.py
python train_clustering.py
python main.py
.
├── models/                # Model definitions
│   └── autoencoder.py
├── clustering/            # Clustering logic
│   └── dec.py
├── data/                  # Data loading scripts
├── main.py                # Full pipeline script
├── train_autoencoder.py   # Pretraining autoencoder
├── train_clustering.py    # Clustering training script
├── utils/                 # Helper functions
├── requirements.txt
└── README.md



