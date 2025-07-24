
# AI PPG

**AI PPG** is a repository developed to explore and implement AI strategies for the identification of **ectopic beats** in streaming **photoplethysmogram (PPG)** signals. The project focuses on real-time signal processing and heartbeat anomaly detection using both **machine learning** and **deep learning** techniques.

---

## ✨ Highlights

- ✅ Robust **filtering and outlier rejection** strategy tailored for noisy PPG data  
- ✅ Robust **windowing** approach for effective temporal context extraction  
- ✅ Evaluation of multiple AI models:
  - 🧠 **Convolutional Neural Networks (CNN)**
  - 🔁 **Recurrent Neural Networks (RNN)**
  - 📈 **Support Vector Machines (SVM)**
  - 📊 **Logistic Regression**
  - 🌲 **Random Forest**

---

## 🎯 Objective

To provide a reliable and scalable pipeline for the detection of **ectopic beats** in real-time PPG streams, enabling improved cardiovascular monitoring in **wearable** and **mobile health** applications.

---
## 🧼 Preprocessing

To ensure signal quality and consistent input formatting across models, the following preprocessing steps were applied:

- **Filtering**:
  - Applied a forward-backward **Chebyshev Type II bandpass filter** using the [`pyPPG`](https://github.com/godamartonaron/GODA_pyPPG.git) library.
  - Cutoff frequencies: **0.5 Hz – 4.3 Hz**, chosen to preserve the physiological pulse range.
  - Helps reduce high-frequency noise and baseline drift without introducing phase distortion.

- **Cropping**:
  - Signals were segmented into **individual crops**, each containing **one labeled heartbeat**.
  - **Onsets** (start of a heartbeat) were detected using [`pyPPG`](https://github.com/godamartonaron/GODA_pyPPG.git).
  - Each crop spans from one onset to the next, capturing a full pulse cycle.

- **Dataset Preparation**:
  - **Corrupted or distorted crops** were removed using hard thresholding on peak amplitude, based on the interquartile range (IQR).
  - The dataset was split **patient-wise** into training (70%), validation (15%), and test (15%) sets to avoid data leakage.
  - **Class imbalance** was mitigated via **random undersampling** of the dominant class (normal beats), resulting in a balanced distribution across classes.

---
### 🧰  Machine Learning Feature Engineering

In addition, To train ML models only, each crop was described by a vector of handcrafted features. A total of 21 features were extracted for each crop. After a **correlation analysis**, 7 highly correlated features were removed to reduce redundancy, leaving **14 key features** for model training:
:

- **Temporal**: `crop_duration`, `t_peak`, `pulse_width`, `Peak-to-Peak (PTP)`
- **Statistical**: `median`, `std`, `skew`, `kurt`
- **Spectral**: `spectral_entropy`, `std_spectrogram`, `average_energy`
- **Morphological**: `symmetry`, `peak_amplitude`, `Abnormal_to_Normal_ratio`

A **Z-score method** was used to identify and remove outliers from the training and validation sets.

---

## 🧪 Experiments


Both deep learning (DL) and machine learning (ML) models were evaluated on a Binary setting (Normal vs. Abnormal).
To realistically emulate a streaming data scenario, each sample was constructed to preserve the correct temporal consistency present in the original signal sequences. This design ensured that temporal dependencies were maintained throughout both train and evaluation.

- For **ML models**, temporal features were computed on-the-fly for each incoming signal crop, simulating real-time feature extraction.  
- For **recurrent neural networks (RNNs)**, samples were generated using a sliding window approach, where each new crop was appended to the sequence and only the classification for the most recent crop was predicted, maintaining temporal continuity across crops.  
- **Convolutional neural networks (CNNs)**, which inherently do not model temporal dependencies across samples, were applied to classify single crops only.

This setup allows for a comprehensive comparison of different modeling approaches under streaming-like conditions.

---

##  📈 Results

### 🔍 Binary Classification (Normal vs Abnormal)

| Model             | Recall (%) | Precision (%) | F1-score (%) | Accuracy (%) |
|-------------------|------------|---------------|--------------|--------------|
| **CNN+LSTM**        | **95.1**   | **72.1**      | **82.0**     | **94.5**     |
| LSTM              | 93.9       | 69.3          | 79.6         | 93.5         |
| ConvNeXt          | 92.8       | 67.3          | 78.0         | 92.3         |
| ResNet            | 89.2       | 63.6          | 74.3         | 90.1         |
| Logistic Regression | 89.6       | 71.1          | 79.3         | 92.2         |
| SVM (RBF Kernel)  | 82.9       | 70.4          | 76.1         | 89.8         |
| Random Forest     | 71.5       | 71.3          | 71.3         | 90.7         |

---

## 🙏 Acknowledgments

We would like to thank the authors of the [pyPPG](https://github.com/godamartonaron/GODA_pyPPG) repository for their excellent work in PPG signal preprocessing. 