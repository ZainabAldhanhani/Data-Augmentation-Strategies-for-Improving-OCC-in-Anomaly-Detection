# Data Augmentation Strategies for Improving One-Class Classification in Anomaly Detection
This repository presents our research on enhancing one-class classification for anomaly detection through advanced data augmentation techniques. We integrate methods such as CutPaste, Elastic Deformations, Fourier Domain Adaptation (FDA), and simulated environmental effects into Deep SVDD, OCGAN, and Autoencoder frameworks to enrich training data diversity. The project systematically evaluates each augmentation—and their combinations—on benchmark datasets, measuring improvements in detection accuracy, false positive rates, and model generalization. By introducing a robust augmentation pipeline across multiple OCC architectures, this work contributes scalable strategies for reducing missed anomalies and false alarms in real-world deployment scenarios.  
  

## Overview

One-Class Classification (OCC) models for anomaly detection often suffer from limited data diversity, which impairs generalization and reduces detection accuracy in real-world applications. Conventional OCC approaches such as Deep SVDD, OCGAN, and Autoencoders are particularly sensitive to unseen variations during deployment, resulting in elevated false positive rates and missed anomalies.

This project explores **advanced data augmentation** techniques to enrich training data and improve OCC model robustness. We implement and systematically evaluate the following augmentation methods on benchmark anomaly detection datasets:

- **CutPaste**  
- **Elastic Deformations**  
- **Fourier Domain Adaptation (FDA)**  
- **Simulated Environmental Effects**  

Our goal is to quantify how each technique (and their combinations) affects anomaly detection performance, false positive rates, and model generalization across Deep SVDD, OCGAN, and Autoencoder frameworks.

## Key Features

-  Integration of multiple state-of-the-art augmentation strategies  
-  Automated training and evaluation pipelines for Deep SVDD, OCGAN, and Autoencoder models  
-  Comprehensive ablation studies on augmentation impact  
-  Modular codebase for easy addition of new augmentations or OCC architectures  

## Data Augmentation Techniques

1. **CutPaste**  
   Randomly crops and pastes patches within images to simulate localized defects.  

2. **Elastic Deformations**  
   Applies random grid distortions to mimic non-rigid deformations.  

3. **Fourier Domain Adaptation (FDA)**  
   Swaps low-frequency amplitude components between source and target domains to introduce style variations.  

4. **Simulated Environmental Effects**  
   Adds realistic noise, blur, lighting shifts, and weather artifacts (e.g., rain, fog).  


## Repository Structure

- `Classification/`: Classification VGG16, ResNet50, and ConvNeXt
- `Pairwise-learning/`: Classification with Pairwise-learning for VGG16, ResNet50, and ConvNeXt
- `Visualization/`: View dataset samples
- `README.md`: Overview and setup instructions.
- `requirements.txt`: Required libraries for the project.


## Getting Started

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/occ-data-augmentation.git
   cd occ-data-augmentation
  

2. **Install Requirements**

```bash
pip install -r requirements.txt
```
### Dataset

This project uses the **AI vs Human Generated Dataset** provided by Shutterstock and DeepMedia via Kaggle.

The dataset consists of paired authentic and AI-generated images, with each real image matched to a synthetic version created using advanced generative models. It includes a balanced mix of content types, including images featuring people, to support robust and diverse model training. The data is divided into `train` and `test` folders, with labels provided only for the training set.

You can download the dataset directly from Kaggle:  
 [https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)


To explore and verify the dataset structure, we provide a helper script: `preview_dataset.py`. This script offers a quick overview of the dataset and ensures everything is properly organized after downloading it from Kaggle.

Make sure `train.csv` and the `train_data/` folder are in the same directory. Then run:

```bash
python preview_dataset.py /path/to/your_dataset/train
```

## Classification (Without Pairwise learning)
To train and evaluate a model, use the following commands:

#### For VGG16: 
```bashn
python python Classification/VGG16.py --dataset_path Path
```
#### For ResNet50: 
```bash
python python Classification/ResNet50.py --dataset_path Path
```
#### for ConvNeXt:
```bashn
python python Classification/ConvNext.py --dataset_path Path
```

## Classification (With Pairwise learning)
To train and evaluate a model, use the following commands:
#### For VGG16: 
```bashn
python python Pairwise-learning/VGG16.py --dataset_path Path
```
#### For ResNet50: 
```bash
python python Pairwise-learning/ResNet50.py --dataset_path Path
```
#### for ConvNeXt:
```bashn
python python Pairwise-learning/ConvNext.py --dataset_path Path
```


