# Data-Driven Reconstruction of Thermal Conductivity Profiles

This project develops an adaptive expert ensemble model to solve the ill-posed inverse problem of reconstructing 1D thermal conductivity profiles ($K(z)$) from Time-Domain Thermoreflectance (TDTR) measurements.

## 0. The Problem: An Ill-Posed Inverse Problem

Determining a material's thermal conductivity as a function of depth ($K(z)$) is critical for designing and understanding nanomaterials and thin films. Time-Domain Thermoreflectance (TDTR) is a standard experimental technique that provides a 1D thermal response curve (`ModelRatio` vs. `Tdelay`).

The challenge is the **inverse problem**:
* **Input:** A 1D TDTR signal (a time-series curve).
* **Output:** The underlying 1D thermal conductivity profile ($K(z)$) that produced it.

This reconstruction is notoriously difficult because the mapping from $K(z)$ to the TDTR signal is highly non-linear and ill-posed, meaning multiple complex profiles can produce very similar signals.


## 1. Data
The full, multi-gigabyte dataset used for this project is not stored in this repository. It is publicly available as a Kaggle Dataset.

Kaggle Dataset URL: https://www.kaggle.com/datasets/utkarshsharma2007/projectdata

To run the analysis, please download the CSV files and place them in the data/ directory.


## 2. Our Solution: An Adaptive Expert Ensemble

No single machine learning model excels at all profile types. Kernel Ridge Regression (KRR) is fast and effective for simple, smooth profiles, but fails on complex, non-linear patterns. Deep Neural Networks (DNNs) can capture these non-linearities but are slower and can be unstable.

Our solution is a **hard-gated expert ensemble** that combines the strengths of both models. This system acts as a "smart switch" that routes each sample to the best-suited expert for the job.

The architecture consists of three main components:

### Expert 1: The KRR "Efficiency Expert"
* **Model:** A `sklearn.kernel_ridge.KernelRidge` model.
* **Tuning:** Tuned using `GridSearchCV`, employing a **Laplacian kernel** with `alpha=0.01` and `gamma=0.01`.
* **Role:** Quickly and accurately reconstructs simple, smooth profiles.

### Expert 2: The "Best-NN" Accuracy Expert
* **Model:** This is an "oracle" that uses the best prediction from two different deep learning architectures:
    1.  **Transformer:** A 3-layer `TransformerEncoder` (8 heads) to capture complex, non-local dependencies in the data.
    2.  **ResNet-MLP:** A 3-layer feed-forward network with residual skip connections.
* **Role:** Captures the complex, highly non-linear patterns that KRR misses.

### The Gate: The "Smart Switch"
The "brain" of our ensemble is a **classifier model (`XGBClassifier`)** that decides which expert to use for each sample.

* **Data Pipeline:**
    1.  **Input (X):** The TDTR curve is converted into a **51-dimensional Fourier feature vector**.
    2.  **Output (Y):** The $K(z)$ profile is interpolated onto a **100-point standardized grid**.
* **Gate Features:** To make an informed decision, the gate is trained on **351 "meta-features"**:
    1.  The 51-dim original Fourier input.
    2.  The 100-dim scaled *prediction* from the KRR expert.
    3.  The 100-dim scaled *prediction* from the Best-NN expert.
    4.  The 100-dim *difference vector* (`krr_pred - nn_pred`).
* **Gate Training:** The gate is trained on a binary label: `1` if the NN's true MSE was lower than KRR's, and `0` otherwise. It is trained to optimize for **accuracy**, learning the natural distribution of the training data (where the NN wins ~70-75% of the time).

This ensemble approach allows us to achieve robust and accurate reconstructions across a wide variety of profile types, leveraging KRR's speed for simple cases and the DNN's power for
complex ones.

## 3. Results
*(Placeholder: Add your final plots here, such as the `model_performance_by_dist.png` and a few key reconstruction examples.)*

## 4. Installation and Usage
*(Placeholder: You will add this later.)*
