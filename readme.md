Hyperspectral Anomaly Detection Using Autoencoder, KMeans, and One-Class SVM
============================================================================

This repository contains a Python implementation for hyperspectral anomaly detection using a combination of an Autoencoder, KMeans clustering, and One-Class SVM. The primary objective is to detect anomalies in hyperspectral images by leveraging deep learning and machine learning techniques.

Table of Contents
=================
<!--ts-->
  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Methodology](#methodology)
  * [Results](#result)
  * [Contributions](*contributions)
  * [License](#license)
<!--te-->

Introduction
============
Hyperspectral imaging captures a wide spectrum of light across many narrow bands, enabling the identification of materials based on their spectral signatures. Anomaly detection in hyperspectral images is critical for applications such as remote sensing, environmental monitoring, and material identification. This project implements an anomaly detection pipeline using different methods like Autoencoders, KMeans clustering and One-Class SVM for unsupervised anomaly detection.


Installation
============

Prerequisites:

* Python 3.x
* pip (Python package installer)
* Git (for cloning the repository)

Setup: 

* Clone the repository:

```bash
$ git clone https://github.com/Potafe/Hyperspectral-Anomaly-Detection.git
$ cd hyperspectral-anomaly-detection
```


```bash
$ python3 -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


* Install the required packages:

```bash
$ pip install -r requirements.txt
```


Usage
=====
* Prepare your hyperspectral dataset by creating a new data dir, or by downloading the sample dataset from [data](https://drive.google.com/drive/folders/1B_ACSY7oikLaKuAFmBzowqnfi0fHhGVt?usp=drive_link).
* If you're using your own dataset ensure that your dataset is organized properly and the images are in a format that can be read by the code (e.g., .mat, ENVI).
 

* To run the anomaly detection pipeline, execute:

```bash
$ python <file_name>.py
```


Methodology
==========
 ### Autoencoder
  * Purpose: The Autoencoder is used for unsupervised anomaly detection by learning a compressed representation of the hyperspectral image. The reconstruction error is used to identify anomalous pixels.
  * Approach: The Autoencoder is trained on the dataset to minimize reconstruction error, which is then used to detect anomalies as regions with high reconstruction error.
  
  ### KMeans Clustering
  * Purpose: KMeans clustering groups the pixels in the hyperspectral image based on their spectral characteristics. Anomalies are identified as pixels that do not belong to any cluster or are far from cluster centers.
  * Approach: After clustering, a distance metric is applied to detect pixels that are far from the nearest cluster centroid, marking them as anomalies.

  ### One-Class SVM
  * Purpose: One-Class SVM is a supervised method used to learn the boundary of normal data in the feature space. It identifies data points outside this boundary as anomalies.
  * Approach: The One-Class SVM is trained on the majority class (normal data) and used to classify new data points, with anomalies being identified as outliers.

  ### LRX (Low Rank Representation)
* Purpose: The Low Rank Representation (LRX) method is used for anomaly detection by decomposing the hyperspectral image into a low-rank part and a sparse part. The sparse component represents the anomalies in the image.
* Approach: LRX assumes that the background of the hyperspectral image can be modeled as a low-rank matrix, while the anomalies contribute to the sparse matrix. The method uses matrix decomposition techniques to separate the background from the anomalies, with the anomalies being identified in the sparse matrix.
### RX (Reed-Xiaoli Detector)
* Purpose: The Reed-Xiaoli (RX) detector is a statistical anomaly detection method commonly used in hyperspectral imaging. It identifies anomalies by comparing the spectral signature of each pixel against the global statistical properties of the image.
* Approach: The RX detector computes a Mahalanobis distance for each pixel relative to the mean and covariance of the image. Pixels with a high Mahalanobis distance are flagged as anomalies, as they deviate significantly from the background distribution.

Results
======
* Anomaly Maps: Each method generates an anomaly map highlighting pixels identified as anomalous.

Contributions
=============
* Contributions are welcome! Please feel free to submit a Pull Request.

License
======
* This project is licensed under the MIT License - see the [LICENSE](https://github.com/Potafe/Hyperspectral-Anomaly-Detection/blob/main/LICENSE) file for details.
